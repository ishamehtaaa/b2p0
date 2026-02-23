from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict, cast

from polymarket_bot.http_utils import get_json, websocket_sslopt
from polymarket_bot.models import (
    FeeInfo,
    OrderBookLevel,
    OrderBookSnapshot,
)

LOGGER = logging.getLogger("polymarket_bot")


class FeeRatePayload(TypedDict):
    base_fee: int | float | str


class BookLevelPayload(TypedDict, total=False):
    price: int | float | str
    size: int | float | str


class BookRestPayload(TypedDict):
    timestamp: int | float | str
    bids: list[BookLevelPayload]
    asks: list[BookLevelPayload]


class ClobBookEventPayload(TypedDict, total=False):
    event_type: str
    asset_id: str
    bids: list[BookLevelPayload]
    asks: list[BookLevelPayload]
    timestamp: int | float | str


class PriceChangeEntryPayload(TypedDict, total=False):
    asset_id: str
    side: str
    price: int | float | str
    size: int | float | str


class ClobPriceChangeEventPayload(TypedDict, total=False):
    event_type: str
    price_changes: list[PriceChangeEntryPayload]
    timestamp: int | float | str


@dataclass
class ClobClient:
    base_url: str
    timeout_seconds: float = 10.0

    @staticmethod
    def _as_object(payload: object, *, context: str) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"{context} payload must be a JSON object")
        return cast(dict[str, object], payload)

    @staticmethod
    def _parse_number(value: object, *, field: str, context: str) -> float:
        if isinstance(value, bool):
            raise RuntimeError(f"{context}.{field} cannot be boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise RuntimeError(f"{context}.{field} is empty")
            try:
                return float(text)
            except ValueError as exc:
                raise RuntimeError(
                    f"{context}.{field} must be numeric, got {value!r}"
                ) from exc
        raise RuntimeError(
            f"{context}.{field} has unsupported type {type(value).__name__}"
        )

    @classmethod
    def _parse_int(cls, value: object, *, field: str, context: str) -> int:
        numeric = cls._parse_number(value, field=field, context=context)
        return int(numeric)

    def get_fee_rate(self, token_id: str, side: str = "buy") -> FeeInfo:
        params = {"token_id": token_id, "side": side}
        payload_raw = get_json(
            f"{self.base_url}/fee-rate", params=params, timeout=self.timeout_seconds
        )
        payload = cast(
            FeeRatePayload,
            self._as_object(payload_raw, context="clob.fee_rate"),
        )
        if "base_fee" not in payload:
            raise RuntimeError("clob.fee_rate payload missing base_fee")
        base_fee = self._parse_int(
            payload.get("base_fee"),
            field="base_fee",
            context="clob.fee_rate",
        )
        return FeeInfo(
            token_id=token_id,
            base_fee=base_fee,
            fetched_at=datetime.now(tz=timezone.utc),
        )

    def get_book(self, token_id: str) -> OrderBookSnapshot:
        params = {"token_id": token_id}
        payload_raw = get_json(
            f"{self.base_url}/book", params=params, timeout=self.timeout_seconds
        )
        payload = cast(
            BookRestPayload,
            self._as_object(payload_raw, context="clob.book"),
        )
        bids_raw = payload.get("bids")
        asks_raw = payload.get("asks")
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            raise RuntimeError("clob.book payload must include bids/asks arrays")
        if "timestamp" not in payload:
            raise RuntimeError("clob.book payload missing timestamp")
        timestamp_ms = self._parse_timestamp_ms(
            payload.get("timestamp"),
            context="clob.book",
            allow_missing_now=False,
        )
        bids = self._parse_levels(bids_raw, reverse=True, context="clob.book.bids")
        asks = self._parse_levels(asks_raw, reverse=False, context="clob.book.asks")
        return OrderBookSnapshot(
            token_id=token_id,
            timestamp_ms=timestamp_ms,
            bids=bids,
            asks=asks,
        )


class ClobMarketStream:
    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._books: dict[str, OrderBookSnapshot] = {}
        self._desired_assets: set[str] = set()
        self._subscribed_assets: set[str] = set()
        self._connected = False
        self._ws = None
        self._enabled = False
        self._fatal_error: str | None = None

        try:
            from websocket import WebSocketApp  # type: ignore
        except Exception:
            self._ws_app_cls = None
        else:
            self._ws_app_cls = WebSocketApp
            self._enabled = True

    def start(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                "websocket-client not installed; CLOB stream unavailable"
            )
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._connected = False
            self._fatal_error = None
            self._subscribed_assets = set()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="clob-market-ws", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._ping_thread and self._ping_thread.is_alive():
            self._ping_thread.join(timeout=1.5)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None
        self._ping_thread = None

    def wait_until_ready(self, timeout_seconds: float = 5.0) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with self._lock:
                if self._fatal_error:
                    raise RuntimeError(f"CLOB WS failed: {self._fatal_error}")
                if self._connected:
                    return
            time.sleep(0.05)
        with self._lock:
            if self._fatal_error:
                raise RuntimeError(f"CLOB WS failed: {self._fatal_error}")
        raise RuntimeError("CLOB WS did not connect within startup timeout")

    def assert_healthy(self) -> None:
        with self._lock:
            if self._fatal_error:
                raise RuntimeError(f"CLOB WS failed: {self._fatal_error}")
            if not self._connected:
                raise RuntimeError("CLOB WS not connected")

    def wait_for_assets(
        self, asset_ids: list[str], timeout_seconds: float = 0.8
    ) -> None:
        wanted = {str(x) for x in asset_ids if str(x)}
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            self.assert_healthy()
            with self._lock:
                ready = all(asset_id in self._books for asset_id in wanted)
            if ready:
                return
            time.sleep(0.02)
        missing = sorted(wanted - set(self.get_books(list(wanted)).keys()))
        raise RuntimeError(f"CLOB WS missing books for assets: {','.join(missing[:6])}")

    def set_assets(self, asset_ids: list[str]) -> None:
        self.assert_healthy()
        clean_assets = {str(x) for x in asset_ids if str(x)}
        with self._lock:
            self._desired_assets = clean_assets
        self._sync_subscriptions()

    def get_books(self, token_ids: list[str]) -> dict[str, OrderBookSnapshot]:
        with self._lock:
            result: dict[str, OrderBookSnapshot] = {}
            for token_id in token_ids:
                book = self._books.get(token_id)
                if book is None:
                    continue
                result[token_id] = self._clone_book(book)
            return result

    @staticmethod
    def _clone_book(book: OrderBookSnapshot) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            token_id=book.token_id,
            timestamp_ms=book.timestamp_ms,
            bids=[
                OrderBookLevel(price=level.price, size=level.size)
                for level in book.bids
            ],
            asks=[
                OrderBookLevel(price=level.price, size=level.size)
                for level in book.asks
            ],
        )

    def _run_loop(self) -> None:
        backoff_seconds = 0.5
        while not self._stop_event.is_set():
            try:
                app = self._ws_app_cls(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = app
                app.run_forever(
                    ping_interval=15, ping_timeout=8, sslopt=websocket_sslopt()
                )
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                LOGGER.warning("CLOB WS loop error: %s", exc)
            finally:
                self._ws = None
                with self._lock:
                    self._connected = False
                    self._subscribed_assets = set()

            if self._stop_event.is_set():
                break
            time.sleep(backoff_seconds)
            backoff_seconds = min(5.0, backoff_seconds * 1.8)

    def _on_open(self, ws) -> None:
        with self._lock:
            self._connected = True
            self._fatal_error = None
            desired = sorted(self._desired_assets)
        if desired:
            self._send_json(ws, {"assets_ids": desired, "type": "market"})
            with self._lock:
                self._subscribed_assets = set(desired)
        self._start_ping_loop(ws)

    def _start_ping_loop(self, ws) -> None:
        if self._ping_thread and self._ping_thread.is_alive():
            return

        def _ping() -> None:
            while not self._stop_event.is_set():
                time.sleep(10.0)
                try:
                    with self._send_lock:
                        ws.send("PING")
                except Exception:
                    return

        self._ping_thread = threading.Thread(
            target=_ping, name="clob-market-ws-ping", daemon=True
        )
        self._ping_thread.start()

    def _on_message(self, _ws, message: str) -> None:
        if message in {"PONG", "PING"}:
            return
        try:
            payload = json.loads(message)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self.process_payload(payload)

    def process_payload(self, payload: dict[str, object]) -> None:
        event_type_raw = payload.get("event_type")
        if not isinstance(event_type_raw, str):
            return
        event_type = event_type_raw.strip().lower()
        if event_type == "book":
            self._process_book(cast(ClobBookEventPayload, payload))
            return
        if event_type == "price_change":
            self._process_price_change(cast(ClobPriceChangeEventPayload, payload))
            return

    def _process_book(self, payload: ClobBookEventPayload) -> None:
        asset_id_raw = payload.get("asset_id")
        asset_id = asset_id_raw.strip() if isinstance(asset_id_raw, str) else ""
        if not asset_id:
            return
        bids_raw = payload.get("bids")
        asks_raw = payload.get("asks")
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return
        try:
            bids = self._parse_levels(
                bids_raw, reverse=True, context="clob.ws.book.bids"
            )
            asks = self._parse_levels(
                asks_raw, reverse=False, context="clob.ws.book.asks"
            )
            timestamp_ms = self._parse_timestamp_ms(
                payload.get("timestamp"),
                context="clob.ws.book",
                allow_missing_now=True,
            )
        except Exception:
            return

        with self._lock:
            self._books[asset_id] = OrderBookSnapshot(
                token_id=asset_id,
                timestamp_ms=timestamp_ms,
                bids=bids,
                asks=asks,
            )

    def _process_price_change(self, payload: ClobPriceChangeEventPayload) -> None:
        changes = payload.get("price_changes")
        if not isinstance(changes, list):
            return
        try:
            timestamp_ms = self._parse_timestamp_ms(
                payload.get("timestamp"),
                context="clob.ws.price_change",
                allow_missing_now=True,
            )
        except Exception:
            return
        for change in changes:
            if not isinstance(change, dict):
                continue
            change_obj = cast(PriceChangeEntryPayload, change)
            asset_id_raw = change_obj.get("asset_id")
            asset_id = asset_id_raw.strip() if isinstance(asset_id_raw, str) else ""
            if not asset_id:
                continue
            side_raw = change_obj.get("side")
            side = side_raw.strip().upper() if isinstance(side_raw, str) else ""
            try:
                price = ClobClient._parse_number(
                    change_obj.get("price"),
                    field="price",
                    context="clob.ws.price_change",
                )
                size = ClobClient._parse_number(
                    change_obj.get("size"),
                    field="size",
                    context="clob.ws.price_change",
                )
            except Exception:
                continue
            if price <= 0 or side not in {"BUY", "SELL"}:
                continue
            self._apply_price_change(
                asset_id=asset_id,
                side=side,
                price=price,
                size=size,
                timestamp_ms=timestamp_ms,
            )

    def _apply_price_change(
        self, *, asset_id: str, side: str, price: float, size: float, timestamp_ms: int
    ) -> None:
        with self._lock:
            current = self._books.get(asset_id)
            if current is None:
                current = OrderBookSnapshot(
                    token_id=asset_id, timestamp_ms=timestamp_ms, bids=[], asks=[]
                )

            bid_map = {level.price: level.size for level in current.bids}
            ask_map = {level.price: level.size for level in current.asks}
            levels = bid_map if side == "BUY" else ask_map
            if size <= 0:
                levels.pop(price, None)
            else:
                levels[price] = size

            bids = [
                OrderBookLevel(price=p, size=s) for p, s in bid_map.items() if s > 0
            ]
            asks = [
                OrderBookLevel(price=p, size=s) for p, s in ask_map.items() if s > 0
            ]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            self._books[asset_id] = OrderBookSnapshot(
                token_id=asset_id,
                timestamp_ms=timestamp_ms,
                bids=bids[:200],
                asks=asks[:200],
            )

    def _on_error(self, _ws, error) -> None:
        if self._stop_event.is_set():
            return
        with self._lock:
            self._connected = False
        LOGGER.warning("CLOB WS error: %s", error)

    def _on_close(self, _ws, status_code, msg) -> None:
        with self._lock:
            self._connected = False
            self._subscribed_assets = set()
        if self._stop_event.is_set():
            return
        LOGGER.warning("CLOB WS closed code=%s msg=%s", status_code, msg)

    def _sync_subscriptions(self) -> None:
        ws = self._ws
        if not self._enabled or ws is None:
            raise RuntimeError("CLOB WS unavailable while syncing subscriptions")
        with self._lock:
            if not self._connected:
                raise RuntimeError("CLOB WS not connected while syncing subscriptions")
            desired = set(self._desired_assets)
            subscribed = set(self._subscribed_assets)
        to_sub = sorted(desired - subscribed)
        to_unsub = sorted(subscribed - desired)
        if to_sub:
            self._send_json(ws, {"assets_ids": to_sub, "operation": "subscribe"})
        if to_unsub:
            self._send_json(ws, {"assets_ids": to_unsub, "operation": "unsubscribe"})
        with self._lock:
            self._subscribed_assets |= set(to_sub)
            self._subscribed_assets -= set(to_unsub)

    def _send_json(self, ws, payload: dict[str, object]) -> None:
        try:
            body = json.dumps(payload, separators=(",", ":"))
            with self._send_lock:
                ws.send(body)
        except Exception as exc:
            with self._lock:
                self._connected = False
            LOGGER.warning("CLOB WS send failed: %s", exc)

    @classmethod
    def _parse_timestamp_ms(
        cls, raw: object, *, context: str, allow_missing_now: bool
    ) -> int:
        if raw is None:
            if allow_missing_now:
                return int(time.time() * 1000)
            raise RuntimeError(f"{context}.timestamp is required")
        try:
            return ClobClient._parse_int(raw, field="timestamp", context=context)
        except Exception as exc:
            if allow_missing_now:
                return int(time.time() * 1000)
            raise RuntimeError(f"{context}.timestamp is invalid") from exc

    @classmethod
    def _parse_levels(
        cls, raw_levels: object, *, reverse: bool, context: str
    ) -> list[OrderBookLevel]:
        levels: list[OrderBookLevel] = []
        if not isinstance(raw_levels, list):
            raise RuntimeError(f"{context} must be a list")
        for level in raw_levels:
            if not isinstance(level, dict):
                continue
            level_obj = cast(BookLevelPayload, level)
            try:
                price = ClobClient._parse_number(
                    level_obj.get("price"), field="price", context=context
                )
                size = ClobClient._parse_number(
                    level_obj.get("size"), field="size", context=context
                )
            except Exception:
                continue
            if price <= 0 or size <= 0:
                continue
            levels.append(OrderBookLevel(price=price, size=size))
        levels.sort(key=lambda x: x.price, reverse=reverse)
        return levels[:200]

    def _mark_fatal(self, message: str) -> None:
        with self._lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = message
            self._connected = False
        LOGGER.error("CLOB WS fatal: %s", message)

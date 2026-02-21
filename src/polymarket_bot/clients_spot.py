from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Callable

from polymarket_bot.http_utils import get_json, websocket_sslopt

LOGGER = logging.getLogger("polymarket_bot")


@dataclass
class BtcSpotClient:
    url: str
    timeout_seconds: float = 5.0

    def get_price(self) -> float:
        payload = get_json(self.url, timeout=self.timeout_seconds)

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict) and "amount" in data:
                return float(data["amount"])
            if "price" in payload:
                return float(payload["price"])
        raise RuntimeError("Unable to parse BTC spot price response")


class BtcSpotStream:
    def __init__(
        self,
        ws_url: str,
        rest_client: BtcSpotClient,
        on_price: Callable[[float, float], None],
    ) -> None:
        self.ws_url = ws_url
        self.rest_client = rest_client
        self.on_price = on_price
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_price = 0.0
        self._latest_ts = 0.0
        self._ws_enabled = False
        self._connected = False
        self._fatal_error: str | None = None

        try:
            from websocket import WebSocketApp  # type: ignore
        except Exception:
            self._ws_app_cls = None
        else:
            self._ws_app_cls = WebSocketApp
            self._ws_enabled = True

    def start(self) -> None:
        if not self._ws_enabled:
            raise RuntimeError("websocket-client not installed; spot stream unavailable")
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._connected = False
            self._fatal_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="spot-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def latest(self, max_stale_seconds: float = 1.5) -> tuple[float, str, float]:
        with self._lock:
            fatal_error = self._fatal_error
            connected = self._connected
            price = self._latest_price
            ts = self._latest_ts
        if fatal_error:
            raise RuntimeError(f"Spot WS failed: {fatal_error}")
        now = time.time()
        age = now - ts if ts > 0 else float("inf")
        if connected and price > 0 and age <= max_stale_seconds:
            return price, "ws", age
        if not connected:
            raise RuntimeError("Spot WS not connected")
        raise RuntimeError(f"Spot WS stale (age={age:.2f}s)")

    def wait_until_ready(self, timeout_seconds: float = 5.0) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with self._lock:
                if self._fatal_error:
                    raise RuntimeError(f"Spot WS failed: {self._fatal_error}")
                connected = self._connected
                ts = self._latest_ts
            if connected and ts > 0:
                return
            time.sleep(0.05)
        with self._lock:
            if self._fatal_error:
                raise RuntimeError(f"Spot WS failed: {self._fatal_error}")
        raise RuntimeError("Spot WS did not produce a tick within startup timeout")

    def _run_loop(self) -> None:
        try:
            app = self._ws_app_cls(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            app.run_forever(ping_interval=15, ping_timeout=8, sslopt=websocket_sslopt())
        except Exception as exc:
            self._mark_fatal(str(exc))

    def _on_open(self, _ws) -> None:
        with self._lock:
            self._connected = True

    def _on_message(self, _ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return
        price_raw = payload.get("p") or payload.get("c") or payload.get("price")
        if price_raw is None:
            return
        try:
            price = float(price_raw)
        except Exception:
            return
        if price <= 0:
            return
        ts_ms = payload.get("E") or payload.get("T")
        if isinstance(ts_ms, (int, float)):
            ts = float(ts_ms) / 1000.0
        else:
            ts = time.time()
        self._publish(ts, price)

    def _on_error(self, _ws, error) -> None:
        self._mark_fatal(str(error))

    def _on_close(self, _ws, status_code, msg) -> None:
        with self._lock:
            self._connected = False
        if not self._stop_event.is_set():
            self._mark_fatal(f"closed code={status_code} msg={msg}")

    def _publish(self, ts: float, price: float) -> None:
        with self._lock:
            self._latest_price = price
            self._latest_ts = ts
        self.on_price(ts, price)

    def _mark_fatal(self, message: str) -> None:
        with self._lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = message
            self._connected = False
        LOGGER.error("Spot WS fatal: %s", message)

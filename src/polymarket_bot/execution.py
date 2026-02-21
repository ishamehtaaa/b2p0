from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any
import uuid

from polymarket_bot.config import BotConfig
from polymarket_bot.models import OrderBookSnapshot, OrderIntent, OrderResult, Side, TimeInForce
from polymarket_bot.pricing import per_share_fee

LOGGER = logging.getLogger("polymarket_bot")

ZERO_BYTES32 = "0x" + ("00" * 32)

NEG_RISK_ADAPTER_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_conditionId", "type": "bytes32"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"},
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

CONDITIONAL_TOKENS_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "partition", "type": "uint256[]"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

PROXY_FACTORY_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "enum ProxyWalletLib.CallType",
                        "name": "typeCode",
                        "type": "uint8",
                    },
                    {"internalType": "address payable", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "value", "type": "uint256"},
                    {"internalType": "bytes", "name": "data", "type": "bytes"},
                ],
                "internalType": "struct ProxyWalletLib.ProxyCall[]",
                "name": "calls",
                "type": "tuple[]",
            }
        ],
        "name": "proxy",
        "outputs": [{"internalType": "bytes[]", "name": "returnValues", "type": "bytes[]"}],
        "stateMutability": "payable",
        "type": "function",
    }
]


@dataclass
class OpenPaperOrder:
    order_id: str
    intent: OrderIntent
    created_at: datetime


@dataclass
class SettlementResult:
    status: str
    merged_size: float = 0.0
    redeemed_usdc: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"merged", "redeemed", "ok"}

    @property
    def unsupported(self) -> bool:
        return self.status == "unsupported"


class BaseExecutor:
    def preflight(self) -> None:
        return

    def place_order(self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def cancel_all(self) -> None:
        raise NotImplementedError

    def sweep(self, books: dict[str, OrderBookSnapshot]) -> list[tuple[OrderIntent, OrderResult]]:
        return []

    def register_condition(self, *, condition_id: str, is_neg_risk: bool, market_end_ts: float | None = None) -> None:
        return

    def merge_pairs(
        self,
        *,
        market_id: str,
        primary_token_id: str,
        secondary_token_id: str,
        size: float,
        condition_id: str | None = None,
        is_neg_risk: bool | None = None,
    ) -> SettlementResult:
        return SettlementResult(status="unsupported", raw={"reason": "merge not supported"})

    def redeem_all(self) -> SettlementResult:
        return SettlementResult(status="unsupported", raw={"reason": "redeem not supported"})


class PaperExecutor(BaseExecutor):
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.open_orders: dict[str, OpenPaperOrder] = {}

    def place_order(self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]) -> OrderResult:
        now = datetime.now(tz=timezone.utc)
        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        book = books.get(intent.token_id)
        if book is None:
            return OrderResult(
                order_id=order_id,
                market_id=intent.market_id,
                token_id=intent.token_id,
                side=intent.side,
                price=intent.price,
                size=intent.size,
                status="rejected",
                filled_size=0.0,
                filled_price=0.0,
                fee_paid=0.0,
                engine=intent.engine,
                created_at=now,
                raw={"reason": "book missing"},
            )

        if intent.post_only and self._would_cross(intent, book):
            return OrderResult(
                order_id=order_id,
                market_id=intent.market_id,
                token_id=intent.token_id,
                side=intent.side,
                price=intent.price,
                size=intent.size,
                status="rejected",
                filled_size=0.0,
                filled_price=0.0,
                fee_paid=0.0,
                engine=intent.engine,
                created_at=now,
                raw={"reason": "post-only would cross"},
            )

        if intent.post_only:
            self.open_orders[order_id] = OpenPaperOrder(order_id=order_id, intent=intent, created_at=now)
            return OrderResult(
                order_id=order_id,
                market_id=intent.market_id,
                token_id=intent.token_id,
                side=intent.side,
                price=intent.price,
                size=intent.size,
                status="open",
                filled_size=0.0,
                filled_price=0.0,
                fee_paid=0.0,
                engine=intent.engine,
                created_at=now,
                raw={},
            )

        if intent.side == Side.BUY:
            if book.best_ask <= 0 or intent.price < book.best_ask:
                return OrderResult(
                    order_id=order_id,
                    market_id=intent.market_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size=intent.size,
                    status="rejected",
                    filled_size=0.0,
                    filled_price=0.0,
                    fee_paid=0.0,
                    engine=intent.engine,
                    created_at=now,
                    raw={"reason": "ioc not marketable"},
                )
            fill_price = book.best_ask
        else:
            if book.best_bid <= 0 or intent.price > book.best_bid:
                return OrderResult(
                    order_id=order_id,
                    market_id=intent.market_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size=intent.size,
                    status="rejected",
                    filled_size=0.0,
                    filled_price=0.0,
                    fee_paid=0.0,
                    engine=intent.engine,
                    created_at=now,
                    raw={"reason": "ioc not marketable"},
                )
            fill_price = book.best_bid

        fee = per_share_fee(fill_price, 0) * intent.size
        return OrderResult(
            order_id=order_id,
            market_id=intent.market_id,
            token_id=intent.token_id,
            side=intent.side,
            price=intent.price,
            size=intent.size,
            status="filled",
            filled_size=intent.size,
            filled_price=fill_price,
            fee_paid=fee,
            engine=intent.engine,
            created_at=now,
            raw={},
        )

    def sweep(self, books: dict[str, OrderBookSnapshot]) -> list[tuple[OrderIntent, OrderResult]]:
        now = datetime.now(tz=timezone.utc)
        fills: list[tuple[OrderIntent, OrderResult]] = []
        closed: list[str] = []
        for order_id, open_order in self.open_orders.items():
            book = books.get(open_order.intent.token_id)
            if book is None:
                continue
            if not self._is_fillable(open_order.intent, book):
                continue
            fee = per_share_fee(open_order.intent.price, 0) * open_order.intent.size
            fills.append(
                (
                    open_order.intent,
                    OrderResult(
                        order_id=order_id,
                        market_id=open_order.intent.market_id,
                        token_id=open_order.intent.token_id,
                        side=open_order.intent.side,
                        price=open_order.intent.price,
                        size=open_order.intent.size,
                        status="filled",
                        filled_size=open_order.intent.size,
                        filled_price=open_order.intent.price,
                        fee_paid=fee,
                        engine=open_order.intent.engine,
                        created_at=now,
                        raw={"paper_fill": True},
                    ),
                )
            )
            closed.append(order_id)
        for order_id in closed:
            self.open_orders.pop(order_id, None)
        return fills

    def cancel_order(self, order_id: str) -> bool:
        return self.open_orders.pop(order_id, None) is not None

    def cancel_all(self) -> None:
        self.open_orders.clear()

    def merge_pairs(
        self,
        *,
        market_id: str,
        primary_token_id: str,
        secondary_token_id: str,
        size: float,
        condition_id: str | None = None,
        is_neg_risk: bool | None = None,
    ) -> SettlementResult:
        safe_size = max(0.0, float(size))
        return SettlementResult(
            status="merged",
            merged_size=safe_size,
            raw={
                "paper": True,
                "market_id": market_id,
                "primary_token_id": primary_token_id,
                "secondary_token_id": secondary_token_id,
            },
        )

    def redeem_all(self) -> SettlementResult:
        return SettlementResult(status="ok", raw={"paper": True, "redeemed": 0.0})

    @staticmethod
    def _would_cross(intent: OrderIntent, book: OrderBookSnapshot) -> bool:
        if intent.side == Side.BUY and book.best_ask > 0 and intent.price >= book.best_ask:
            return True
        if intent.side == Side.SELL and book.best_bid > 0 and intent.price <= book.best_bid:
            return True
        return False

    @staticmethod
    def _is_fillable(intent: OrderIntent, book: OrderBookSnapshot) -> bool:
        if intent.side == Side.BUY and book.best_ask > 0 and book.best_ask <= intent.price:
            return True
        if intent.side == Side.SELL and book.best_bid > 0 and book.best_bid >= intent.price:
            return True
        return False


class LiveExecutor(BaseExecutor):
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.client = None
        self._bootstrap_done = False
        self._preflight_done = False
        self._signer_address = ""
        self._funder_address = ""
        self._signature_type = -1
        self._w3 = None
        self._pending_merges: dict[str, dict[str, Any]] = {}
        self._pending_redeems: dict[str, dict[str, Any]] = {}
        self._known_conditions: dict[str, tuple[bool, float]] = {}
        self._redeem_cursor = 0

    @staticmethod
    def _normalize_address(address: str) -> str:
        return address.strip().lower()

    @staticmethod
    def _api_creds_from_env() -> dict[str, str] | None:
        key = (
            os.getenv("POLY_CLOB_API_KEY")
            or os.getenv("POLY_API_KEY")
            or os.getenv("CLOB_API_KEY")
            or os.getenv("PM_API_KEY")
            or ""
        ).strip()
        secret = (
            os.getenv("POLY_CLOB_API_SECRET")
            or os.getenv("POLY_API_SECRET")
            or os.getenv("CLOB_API_SECRET")
            or os.getenv("PM_API_SECRET")
            or ""
        ).strip()
        passphrase = (
            os.getenv("POLY_CLOB_API_PASSPHRASE")
            or os.getenv("POLY_API_PASSPHRASE")
            or os.getenv("CLOB_API_PASSPHRASE")
            or os.getenv("PM_API_PASSPHRASE")
            or ""
        ).strip()
        if key and secret and passphrase:
            return {"key": key, "secret": secret, "passphrase": passphrase}
        return None

    @staticmethod
    def _exception_payload(exc: Exception) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": exc.__class__.__name__,
            "error": str(exc),
        }
        for field in ("status_code", "error_msg", "msg"):
            if hasattr(exc, field):
                payload[field] = getattr(exc, field)
        return payload

    @staticmethod
    def _is_invalid_signature_text(text: str) -> bool:
        lowered = (text or "").lower()
        return "invalid signature" in lowered or "invalid order signature" in lowered

    @classmethod
    def _is_invalid_signature_payload(cls, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        parts = [
            str(payload.get("error") or ""),
            str(payload.get("error_msg") or ""),
            str(payload.get("msg") or ""),
        ]
        return cls._is_invalid_signature_text(" ".join(parts))

    @staticmethod
    def _derive_api_creds_with_retry(client: Any, attempts: int = 4) -> dict[str, Any] | None:
        last_exc: Exception | None = None
        for attempt in range(max(1, attempts)):
            try:
                return client.create_or_derive_api_creds()
            except Exception as exc:  # pragma: no cover - live path
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(0.4 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        return None

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    @classmethod
    def _pick_ci(cls, payload: dict[str, Any] | None, *keys: str) -> Any:
        if not isinstance(payload, dict):
            return None
        lowered = {str(k).lower(): v for k, v in payload.items()}
        for key in keys:
            lk = key.lower()
            if lk in lowered:
                return lowered[lk]
        for nested_key in ("order", "data", "result"):
            nested = lowered.get(nested_key)
            if isinstance(nested, dict):
                found = cls._pick_ci(nested, *keys)
                if found is not None:
                    return found
        return None

    @classmethod
    def _extract_status(cls, payload: dict[str, Any] | None, default: str = "accepted") -> str:
        raw = cls._pick_ci(payload, "status", "state", "orderStatus", "order_status")
        if raw is None:
            return default
        return str(raw).lower()

    @classmethod
    def _extract_order_id(cls, payload: dict[str, Any] | None) -> str:
        raw = cls._pick_ci(payload, "orderID", "orderId", "id")
        if raw is None:
            return uuid.uuid4().hex
        return str(raw)

    @classmethod
    def _extract_fill_fields(
        cls,
        *,
        payload: dict[str, Any] | None,
        fallback_size: float,
        fallback_price: float,
    ) -> tuple[float, float, float]:
        filled_size = cls._as_float(
            cls._pick_ci(
                payload,
                "filledSize",
                "filled_size",
                "matchedSize",
                "matched_size",
                "sizeMatched",
                "size_matched",
                "executedSize",
                "executed_size",
                "filledAmount",
                "filled_amount",
            )
        )
        if filled_size <= 0:
            total_size = cls._as_float(cls._pick_ci(payload, "size", "originalSize", "original_size"))
            remaining = cls._as_float(
                cls._pick_ci(payload, "remainingSize", "remaining_size", "sizeRemaining", "size_remaining")
            )
            if total_size > 0 and remaining >= 0:
                filled_size = max(0.0, total_size - remaining)

        filled_price = cls._as_float(
            cls._pick_ci(
                payload,
                "filledPrice",
                "filled_price",
                "avgPrice",
                "avg_price",
                "averagePrice",
                "average_price",
            )
        )
        fee_paid = cls._as_float(cls._pick_ci(payload, "fee", "feePaid", "fee_paid"))

        status = cls._extract_status(payload, default="")
        if filled_size <= 0 and status in {"filled", "matched", "executed", "completed", "complete"}:
            filled_size = fallback_size
        if filled_size > 0 and filled_price <= 0:
            filled_price = fallback_price
        return filled_size, filled_price, fee_paid

    def _infer_signature_type(self, signer_address: str, funder_address: str) -> int:
        if self.config.poly_signature_type is not None:
            return int(self.config.poly_signature_type)
        if self._normalize_address(funder_address) != self._normalize_address(signer_address):
            # Polymarket proxy wallet / safe style flow.
            return 2
        # EOA wallet.
        return 0

    def _bootstrap_client(self) -> None:
        if self._bootstrap_done:
            return
        self._bootstrap_done = True

        if not self.config.poly_private_key:
            raise RuntimeError("Missing POLY_PRIVATE_KEY for live mode")
        try:
            from py_clob_client.client import ClobClient as PyClobClient  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "py-clob-client is required for live mode. Install it with `pip install py-clob-client`."
            ) from exc
        try:
            from eth_account import Account  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "eth-account is required for live mode. Install it with `pip install eth-account`."
            ) from exc

        signer_address = Account.from_key(self.config.poly_private_key).address
        funder_address = self.config.poly_proxy_address.strip() or signer_address
        signature_type = self._infer_signature_type(signer_address, funder_address)

        self.client = PyClobClient(
            host=self.config.clob_url,
            key=self.config.poly_private_key,
            chain_id=self.config.poly_chain_id,
            signature_type=signature_type,
            funder=funder_address,
        )
        self._signer_address = signer_address
        self._funder_address = funder_address
        self._signature_type = signature_type
        LOGGER.info(
            "live_auth signer=%s funder=%s signature_type=%s",
            self._signer_address,
            self._funder_address,
            self._signature_type,
        )
        if hasattr(self.client, "create_or_derive_api_creds") and hasattr(self.client, "set_api_creds"):
            try:
                creds = self._derive_api_creds_with_retry(self.client)
            except Exception as exc:
                payload = self._exception_payload(exc)
                creds = self._api_creds_from_env()
                if creds is None:
                    raise RuntimeError(
                        "Unable to derive Polymarket API credentials. "
                        f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type} "
                        f"error={payload}"
                    ) from exc
                LOGGER.warning("Using CLOB API creds from environment fallback after derive failure")
            if creds is None:
                creds = self._api_creds_from_env()
                if creds is None:
                    raise RuntimeError(
                        "Unable to derive Polymarket API credentials from wallet key. "
                        f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type}"
                    )
                LOGGER.warning("Using CLOB API creds from environment fallback")
            self.client.set_api_creds(creds)

    def _reinitialize_client_for_signature(self, signature_type: int) -> None:
        if not self.config.poly_private_key:
            raise RuntimeError("Missing POLY_PRIVATE_KEY for live mode")
        try:
            from py_clob_client.client import ClobClient as PyClobClient  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "py-clob-client is required for live mode. Install it with `pip install py-clob-client`."
            ) from exc
        if not self._signer_address:
            try:
                from eth_account import Account  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime dependency path
                raise RuntimeError(
                    "eth-account is required for live mode. Install it with `pip install eth-account`."
                ) from exc
            self._signer_address = Account.from_key(self.config.poly_private_key).address
        if not self._funder_address:
            self._funder_address = self.config.poly_proxy_address.strip() or self._signer_address

        self.client = PyClobClient(
            host=self.config.clob_url,
            key=self.config.poly_private_key,
            chain_id=self.config.poly_chain_id,
            signature_type=int(signature_type),
            funder=self._funder_address,
        )
        if hasattr(self.client, "create_or_derive_api_creds") and hasattr(self.client, "set_api_creds"):
            try:
                creds = self._derive_api_creds_with_retry(self.client)
            except Exception:
                creds = self._api_creds_from_env()
                if creds is None:
                    raise
                LOGGER.warning("Using CLOB API creds from environment fallback after derive failure")
            if creds is None:
                creds = self._api_creds_from_env()
                if creds is None:
                    raise RuntimeError("Unable to derive Polymarket API credentials from wallet key")
                LOGGER.warning("Using CLOB API creds from environment fallback")
            self.client.set_api_creds(creds)

        self._signature_type = int(signature_type)
        self._preflight_done = False
        LOGGER.warning(
            "live_auth_retry signer=%s funder=%s signature_type=%s",
            self._signer_address,
            self._funder_address,
            self._signature_type,
        )
        self.preflight()

    def preflight(self) -> None:
        self._bootstrap_client()
        if self._preflight_done:
            return
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self.client.get_api_keys()
                self._preflight_done = True
                return
            except Exception as exc:  # pragma: no cover - live path
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        if last_exc is not None:
            payload = self._exception_payload(last_exc)
            raise RuntimeError(
                "Live auth preflight failed. "
                "Check POLY_PRIVATE_KEY/POLY_PROXY_ADDRESS or set CLOB API creds env vars. "
                f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type} "
                f"error={payload}"
            ) from last_exc

    def place_order(self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]) -> OrderResult:
        self.preflight()
        now = datetime.now(tz=timezone.utc)
        signature_attempts = [int(self._signature_type)]
        for candidate in (2, 1, 0):
            if candidate not in signature_attempts:
                signature_attempts.append(candidate)

        for attempt_idx, signature_type in enumerate(signature_attempts):
            if attempt_idx > 0:
                try:
                    self._reinitialize_client_for_signature(signature_type)
                except Exception as reinit_exc:
                    payload = self._exception_payload(reinit_exc)
                    if attempt_idx + 1 < len(signature_attempts):
                        LOGGER.warning(
                            "live_auth_retry_failed next_signature_type=%s error=%s",
                            signature_type,
                            payload.get("error", ""),
                        )
                        continue
                    return OrderResult(
                        order_id=f"live-{uuid.uuid4().hex[:12]}",
                        market_id=intent.market_id,
                        token_id=intent.token_id,
                        side=intent.side,
                        price=intent.price,
                        size=intent.size,
                        status="error",
                        filled_size=0.0,
                        filled_price=0.0,
                        fee_paid=0.0,
                        engine=intent.engine,
                        created_at=now,
                        raw=payload,
                    )

            try:
                from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore

                side = "BUY" if intent.side == Side.BUY else "SELL"
                order_args = OrderArgs(
                    price=float(intent.price),
                    size=float(intent.size),
                    side=side,
                    token_id=intent.token_id,
                )
                signed_order = self.client.create_order(order_args)

                order_type = getattr(OrderType, intent.tif.value, getattr(OrderType, "GTC"))
                response = self.client.post_order(signed_order, order_type, post_only=bool(intent.post_only))
                response_payload = response if isinstance(response, dict) else {}
                status = self._extract_status(response_payload, default="accepted")
                order_id = self._extract_order_id(response_payload)

                order_state_payload: dict[str, Any] = {}
                if order_id and intent.tif in {TimeInForce.IOC, TimeInForce.FOK}:
                    try:
                        details = self.client.get_order(order_id)
                        if isinstance(details, dict):
                            order_state_payload = details
                            status = self._extract_status(order_state_payload, default=status)
                    except Exception:
                        pass

                fill_source = order_state_payload if order_state_payload else response_payload
                filled_size, filled_price, fee_paid = self._extract_fill_fields(
                    payload=fill_source,
                    fallback_size=float(intent.size),
                    fallback_price=float(intent.price),
                )
                raw_payload: dict[str, Any] = {"post_order": response_payload, "signature_type": self._signature_type}
                if order_state_payload:
                    raw_payload["order_state"] = order_state_payload
                return OrderResult(
                    order_id=order_id,
                    market_id=intent.market_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size=intent.size,
                    status=status,
                    filled_size=filled_size,
                    filled_price=filled_price,
                    fee_paid=fee_paid,
                    engine=intent.engine,
                    created_at=now,
                    raw=raw_payload,
                )
            except Exception as exc:  # pragma: no cover - live path
                payload = self._exception_payload(exc)
                payload["signature_type"] = self._signature_type
                if self._is_invalid_signature_payload(payload) and attempt_idx + 1 < len(signature_attempts):
                    LOGGER.warning(
                        "order_invalid_signature_retry market=%s current_signature_type=%s next_signature_type=%s",
                        intent.market_id,
                        self._signature_type,
                        signature_attempts[attempt_idx + 1],
                    )
                    continue
                return OrderResult(
                    order_id=f"live-{uuid.uuid4().hex[:12]}",
                    market_id=intent.market_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size=intent.size,
                    status="error",
                    filled_size=0.0,
                    filled_price=0.0,
                    fee_paid=0.0,
                    engine=intent.engine,
                    created_at=now,
                    raw=payload,
                )

        return OrderResult(
            order_id=f"live-{uuid.uuid4().hex[:12]}",
            market_id=intent.market_id,
            token_id=intent.token_id,
            side=intent.side,
            price=intent.price,
            size=intent.size,
            status="error",
            filled_size=0.0,
            filled_price=0.0,
            fee_paid=0.0,
            engine=intent.engine,
            created_at=now,
            raw={"error": "signature retry exhausted"},
        )

    def cancel_order(self, order_id: str) -> bool:
        self._bootstrap_client()
        try:  # pragma: no cover - live path
            self.client.cancel(order_id)
            return True
        except Exception:
            return False

    def cancel_all(self) -> None:
        self._bootstrap_client()
        try:  # pragma: no cover - live path
            self.client.cancel_all()
        except Exception:
            pass

    @staticmethod
    def _normalize_condition_id(condition_id: str) -> str:
        raw = str(condition_id or "").strip().lower()
        if raw.startswith("0x"):
            raw = raw[2:]
        if len(raw) != 64:
            raise RuntimeError("condition_id must be a 32-byte hex value")
        return "0x" + raw

    @staticmethod
    def _to_bytes(data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if data is None:
            return b""
        raw = str(data)
        if raw.startswith("0x"):
            raw = raw[2:]
        return bytes.fromhex(raw)

    @staticmethod
    def _receipt_status(receipt: Any) -> int:
        if receipt is None:
            return 0
        raw = receipt.get("status") if isinstance(receipt, dict) else getattr(receipt, "status", None)
        if raw is None:
            return 0
        if isinstance(raw, str):
            return int(raw, 16) if raw.startswith("0x") else int(raw)
        return int(raw)

    @staticmethod
    def _receipt_summary(receipt: Any) -> dict[str, Any]:
        if receipt is None:
            return {}

        def _as_hex(value: Any) -> str:
            if hasattr(value, "hex"):
                return str(value.hex())
            if isinstance(value, bytes):
                return "0x" + value.hex()
            return str(value)

        out: dict[str, Any] = {
            "status": LiveExecutor._receipt_status(receipt),
        }
        for key in ("transactionHash", "blockHash"):
            value = receipt.get(key) if isinstance(receipt, dict) else getattr(receipt, key, None)
            if value is not None:
                out[key] = _as_hex(value)
        for key in ("blockNumber", "gasUsed", "effectiveGasPrice"):
            value = receipt.get(key) if isinstance(receipt, dict) else getattr(receipt, key, None)
            if value is not None:
                try:
                    out[key] = int(value)
                except Exception:
                    out[key] = str(value)
        return out

    def _web3(self):
        if self._w3 is not None:
            return self._w3
        try:
            from web3 import Web3
            from web3.middleware import geth_poa_middleware
        except Exception as exc:
            raise RuntimeError("web3 is required for merge/redeem. Install with `pip install web3`.") from exc

        provider = Web3.HTTPProvider(
            self.config.polygon_rpc_url,
            request_kwargs={"timeout": max(5.0, self.config.api_timeout_seconds)},
        )
        w3 = Web3(provider)
        try:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception:
            pass
        self._w3 = w3
        return w3

    def _resolve_contract_config(self, *, neg_risk: bool):
        from py_clob_client.config import get_contract_config

        return get_contract_config(int(self.config.poly_chain_id), neg_risk=neg_risk)

    def _read_receipt(self, tx_hash: str) -> Any | None:
        w3 = self._web3()
        try:
            return w3.eth.get_transaction_receipt(tx_hash)
        except Exception as exc:
            text = str(exc).lower()
            if exc.__class__.__name__ == "TransactionNotFound" or "not found" in text:
                return None
            raise

    def _using_proxy_wallet(self) -> bool:
        self.preflight()
        return self._normalize_address(self._funder_address) != self._normalize_address(self._signer_address)

    def _wrap_proxy_call(self, *, target_address: str, calldata: bytes):
        from web3 import Web3

        proxy_factory = str(self.config.poly_proxy_factory_address or "").strip()
        if not proxy_factory:
            raise RuntimeError("POLY_PROXY_FACTORY_ADDRESS is required for proxy-wallet settlement")
        w3 = self._web3()
        factory = w3.eth.contract(address=Web3.to_checksum_address(proxy_factory), abi=PROXY_FACTORY_ABI)
        return factory.functions.proxy([(1, Web3.to_checksum_address(target_address), 0, calldata)])

    def _send_function_tx(self, fn) -> str:
        self.preflight()
        if not self.config.poly_private_key:
            raise RuntimeError("POLY_PRIVATE_KEY missing")
        try:
            from eth_account import Account
        except Exception as exc:
            raise RuntimeError("eth-account is required for merge/redeem. Install with `pip install eth-account`.") from exc

        signer = Account.from_key(self.config.poly_private_key).address
        if self._signer_address and self._normalize_address(signer) != self._normalize_address(self._signer_address):
            raise RuntimeError("signer mismatch with live auth credentials")

        w3 = self._web3()
        nonce = int(w3.eth.get_transaction_count(signer, "pending"))
        gas_price = max(1, int(w3.eth.gas_price))
        tx = fn.build_transaction(
            {
                "from": signer,
                "nonce": nonce,
                "chainId": int(self.config.poly_chain_id),
                "gasPrice": gas_price,
            }
        )
        gas_limit = int(tx.get("gas", 0) or 0)
        if gas_limit <= 0:
            gas_limit = int(w3.eth.estimate_gas(tx))
        tx["gas"] = max(21_000, int(gas_limit * 1.20))

        signed = Account.sign_transaction(tx, self.config.poly_private_key)
        raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        if raw_tx is None:
            raise RuntimeError("unable to access signed raw transaction")
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        return tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)

    def _resolve_pending_merge(self, market_id: str) -> SettlementResult | None:
        pending = self._pending_merges.get(market_id)
        if pending is None:
            return None
        tx_hash = str(pending.get("tx_hash") or "")
        receipt = self._read_receipt(tx_hash)
        if receipt is None:
            return SettlementResult(status="pending", merged_size=0.0, raw=dict(pending))

        self._pending_merges.pop(market_id, None)
        status = self._receipt_status(receipt)
        payload = dict(pending)
        payload["receipt"] = self._receipt_summary(receipt)
        if status == 1:
            return SettlementResult(status="merged", merged_size=float(pending.get("merged_size", 0.0)), raw=payload)
        payload["reason"] = "merge transaction reverted"
        return SettlementResult(status="error", raw=payload)

    def _resolve_pending_redeem(self) -> SettlementResult | None:
        for condition_id, pending in list(self._pending_redeems.items()):
            tx_hash = str(pending.get("tx_hash") or "")
            receipt = self._read_receipt(tx_hash)
            if receipt is None:
                return SettlementResult(status="pending", raw=dict(pending))

            self._pending_redeems.pop(condition_id, None)
            status = self._receipt_status(receipt)
            payload = dict(pending)
            payload["receipt"] = self._receipt_summary(receipt)
            if status == 1:
                return SettlementResult(status="redeemed", redeemed_usdc=0.0, raw=payload)
            payload["reason"] = "redeem transaction reverted"
            return SettlementResult(status="error", raw=payload)
        return None

    def register_condition(self, *, condition_id: str, is_neg_risk: bool, market_end_ts: float | None = None) -> None:
        try:
            normalized = self._normalize_condition_id(condition_id)
        except Exception:
            return
        end_ts = max(0.0, float(market_end_ts)) if market_end_ts is not None else 0.0
        existing = self._known_conditions.get(normalized)
        if existing is None:
            self._known_conditions[normalized] = (bool(is_neg_risk), end_ts)
            return
        existing_neg_risk, existing_end = existing
        self._known_conditions[normalized] = (existing_neg_risk or bool(is_neg_risk), max(existing_end, end_ts))

    def _resolve_neg_risk_flag(self, *, token_id: str, hinted: bool | None) -> bool:
        if hinted is not None:
            return bool(hinted)
        self.preflight()
        try:
            return bool(self.client.get_neg_risk(token_id))
        except Exception:
            return False

    def _build_merge_function(self, *, condition_id: str, amount_raw: int, is_neg_risk: bool):
        from web3 import Web3

        w3 = self._web3()
        config = self._resolve_contract_config(neg_risk=is_neg_risk)
        if is_neg_risk:
            adapter = Web3.to_checksum_address(str(config.exchange))
            contract = w3.eth.contract(address=adapter, abi=NEG_RISK_ADAPTER_ABI)
            return contract.functions.mergePositions(condition_id, int(amount_raw)), adapter

        conditional = Web3.to_checksum_address(str(config.conditional_tokens))
        collateral = Web3.to_checksum_address(str(config.collateral))
        contract = w3.eth.contract(address=conditional, abi=CONDITIONAL_TOKENS_ABI)
        return (
            contract.functions.mergePositions(collateral, ZERO_BYTES32, condition_id, [1, 2], int(amount_raw)),
            conditional,
        )

    def _build_redeem_function(self, *, condition_id: str):
        from web3 import Web3

        w3 = self._web3()
        config = self._resolve_contract_config(neg_risk=False)
        conditional = Web3.to_checksum_address(str(config.conditional_tokens))
        collateral = Web3.to_checksum_address(str(config.collateral))
        contract = w3.eth.contract(address=conditional, abi=CONDITIONAL_TOKENS_ABI)
        return contract.functions.redeemPositions(collateral, ZERO_BYTES32, condition_id, [1, 2]), conditional

    def merge_pairs(
        self,
        *,
        market_id: str,
        primary_token_id: str,
        secondary_token_id: str,
        size: float,
        condition_id: str | None = None,
        is_neg_risk: bool | None = None,
    ) -> SettlementResult:
        del secondary_token_id
        safe_size = max(0.0, float(size))
        if safe_size <= 0:
            return SettlementResult(status="error", raw={"reason": "merge size must be > 0"})

        pending = self._resolve_pending_merge(market_id)
        if pending is not None:
            return pending
        if not condition_id:
            return SettlementResult(status="unsupported", raw={"reason": "missing condition_id"})

        try:
            normalized_condition = self._normalize_condition_id(condition_id)
        except Exception as exc:
            return SettlementResult(status="error", raw=self._exception_payload(exc))

        neg_risk = self._resolve_neg_risk_flag(token_id=primary_token_id, hinted=is_neg_risk)
        amount_raw = max(1, int(round(safe_size * 1_000_000.0)))
        try:
            merge_fn, merge_target = self._build_merge_function(
                condition_id=normalized_condition,
                amount_raw=amount_raw,
                is_neg_risk=neg_risk,
            )
            route = "direct"
            submit_fn = merge_fn
            if self._using_proxy_wallet():
                route = "proxy_factory"
                submit_fn = self._wrap_proxy_call(
                    target_address=merge_target,
                    calldata=self._to_bytes(merge_fn._encode_transaction_data()),
                )
            tx_hash = self._send_function_tx(submit_fn)
        except Exception as exc:
            return SettlementResult(status="error", raw=self._exception_payload(exc))

        merged_size = amount_raw / 1_000_000.0
        pending_payload = {
            "tx_hash": tx_hash,
            "market_id": market_id,
            "condition_id": normalized_condition,
            "is_neg_risk": neg_risk,
            "route": route,
            "target": merge_target,
            "merged_size": merged_size,
        }
        self._pending_merges[market_id] = pending_payload
        self.register_condition(condition_id=normalized_condition, is_neg_risk=neg_risk)

        time.sleep(0.25)
        resolved = self._resolve_pending_merge(market_id)
        if resolved is not None:
            return resolved
        return SettlementResult(status="pending", merged_size=0.0, raw=pending_payload)

    def redeem_all(self) -> SettlementResult:
        pending = self._resolve_pending_redeem()
        if pending is not None:
            return pending

        now_ts = time.time()
        candidates = [
            condition_id
            for condition_id, (is_neg_risk, end_ts) in self._known_conditions.items()
            if (not is_neg_risk) and (end_ts <= 0.0 or now_ts >= (end_ts + 120.0))
        ]
        if not candidates:
            return SettlementResult(status="ok", raw={"reason": "no redeem-eligible conditions"})

        candidates.sort()
        idx = self._redeem_cursor % len(candidates)
        self._redeem_cursor += 1
        condition_id = candidates[idx]

        try:
            redeem_fn, redeem_target = self._build_redeem_function(condition_id=condition_id)
            route = "direct"
            submit_fn = redeem_fn
            if self._using_proxy_wallet():
                route = "proxy_factory"
                submit_fn = self._wrap_proxy_call(
                    target_address=redeem_target,
                    calldata=self._to_bytes(redeem_fn._encode_transaction_data()),
                )
            tx_hash = self._send_function_tx(submit_fn)
        except Exception as exc:
            return SettlementResult(
                status="error",
                raw={
                    "condition_id": condition_id,
                    **self._exception_payload(exc),
                },
            )

        pending_payload = {
            "tx_hash": tx_hash,
            "condition_id": condition_id,
            "route": route,
            "target": redeem_target,
        }
        self._pending_redeems[condition_id] = pending_payload

        time.sleep(0.25)
        resolved = self._resolve_pending_redeem()
        if resolved is not None:
            return resolved
        return SettlementResult(status="pending", raw=pending_payload)

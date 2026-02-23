from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from polymarket_bot.config import BotConfig
from polymarket_bot.models import (
    OrderBookSnapshot,
    OrderIntent,
    OrderResult,
    Side,
    TimeInForce,
)
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
            {
                "internalType": "bytes32",
                "name": "parentCollectionId",
                "type": "bytes32",
            },
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
            {
                "internalType": "bytes32",
                "name": "parentCollectionId",
                "type": "bytes32",
            },
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
                    {
                        "internalType": "address payable",
                        "name": "to",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "value", "type": "uint256"},
                    {"internalType": "bytes", "name": "data", "type": "bytes"},
                ],
                "internalType": "struct ProxyWalletLib.ProxyCall[]",
                "name": "calls",
                "type": "tuple[]",
            }
        ],
        "name": "proxy",
        "outputs": [
            {"internalType": "bytes[]", "name": "returnValues", "type": "bytes[]"}
        ],
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


@dataclass(frozen=True)
class ParsedFill:
    filled_size: float
    filled_price: float
    fee_paid: float


class ClobOrderPayload(TypedDict, total=False):
    id: str
    orderID: str
    status: str
    filledSize: str | int | float
    filledPrice: str | int | float
    avgPrice: str | int | float
    fee: str | int | float


class PendingMergePayload(TypedDict):
    tx_hash: str
    market_id: str
    condition_id: str
    is_neg_risk: bool
    route: str
    target: str
    merged_size: float


class PendingRedeemPayload(TypedDict):
    tx_hash: str
    condition_id: str
    route: str
    target: str


class BalanceAllowancePayload(TypedDict, total=False):
    balance: str | int | float
    allowances: dict[str, str | int | float]


class MakerOrderPayload(TypedDict, total=False):
    order_id: str
    maker_address: str
    matched_amount: str | int | float
    price: str | int | float
    fee_rate_bps: str | int | float
    asset_id: str
    side: str


class TradePayload(TypedDict, total=False):
    id: str
    match_time: str | int | float
    maker_orders: list[MakerOrderPayload]


@dataclass(frozen=True)
class MakerTradeFill:
    trade_id: str
    order_id: str
    token_id: str
    side: Side
    size: float
    price: float
    fee_rate_bps: int
    match_time: int


class BaseExecutor:
    def place_order(
        self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]
    ) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def cancel_all(self) -> None:
        raise NotImplementedError

    def sweep(
        self, books: dict[str, OrderBookSnapshot]
    ) -> list[tuple[OrderIntent, OrderResult]]:
        return []

    def get_order_result(
        self, order_id: str, intent: OrderIntent
    ) -> OrderResult | None:
        return None

    def get_collateral_balance(self) -> float | None:
        return None

    def get_token_balance(self, token_id: str) -> float | None:
        del token_id
        return None

    def get_recent_maker_trade_fills(self, *, after_ts: int) -> list[MakerTradeFill]:
        del after_ts
        return []

    def register_condition(
        self,
        *,
        condition_id: str,
        is_neg_risk: bool,
        market_end_ts: float | None = None,
    ) -> None:
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
        return SettlementResult(
            status="unsupported", raw={"reason": "merge not supported"}
        )

    def redeem_all(self) -> SettlementResult:
        return SettlementResult(
            status="unsupported", raw={"reason": "redeem not supported"}
        )


class PaperExecutor(BaseExecutor):
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.open_orders: dict[str, OpenPaperOrder] = {}

    def place_order(
        self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]
    ) -> OrderResult:
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
            self.open_orders[order_id] = OpenPaperOrder(
                order_id=order_id, intent=intent, created_at=now
            )
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

    def sweep(
        self, books: dict[str, OrderBookSnapshot]
    ) -> list[tuple[OrderIntent, OrderResult]]:
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

    def get_order_result(
        self, order_id: str, intent: OrderIntent
    ) -> OrderResult | None:
        open_order = self.open_orders.get(order_id)
        if open_order is None:
            return None
        now = datetime.now(tz=timezone.utc)
        return OrderResult(
            order_id=order_id,
            market_id=open_order.intent.market_id,
            token_id=open_order.intent.token_id,
            side=open_order.intent.side,
            price=open_order.intent.price,
            size=open_order.intent.size,
            status="open",
            filled_size=0.0,
            filled_price=0.0,
            fee_paid=0.0,
            engine=open_order.intent.engine,
            created_at=now,
            raw={"paper_open": True},
        )

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
        if (
            intent.side == Side.BUY
            and book.best_ask > 0
            and intent.price >= book.best_ask
        ):
            return True
        if (
            intent.side == Side.SELL
            and book.best_bid > 0
            and intent.price <= book.best_bid
        ):
            return True
        return False

    @staticmethod
    def _is_fillable(intent: OrderIntent, book: OrderBookSnapshot) -> bool:
        if (
            intent.side == Side.BUY
            and book.best_ask > 0
            and book.best_ask <= intent.price
        ):
            return True
        if (
            intent.side == Side.SELL
            and book.best_bid > 0
            and book.best_bid >= intent.price
        ):
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
        self._pending_merges: dict[str, PendingMergePayload] = {}
        self._pending_redeems: dict[str, PendingRedeemPayload] = {}
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
    def _derive_api_creds(client: Any) -> dict[str, Any] | None:
        return client.create_or_derive_api_creds()

    @staticmethod
    def _ensure_payload_object(payload: Any, *, context: str) -> ClobOrderPayload:
        if not isinstance(payload, dict):
            raise RuntimeError(f"{context} payload must be a JSON object")
        return cast(ClobOrderPayload, payload)

    @staticmethod
    def _parse_number(value: Any, *, field: str, context: str) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            raise RuntimeError(f"{context}.{field} cannot be boolean")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return 0.0
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
    def _parse_status(cls, payload: ClobOrderPayload, *, default: str) -> str:
        raw = payload.get("status")
        if raw is None:
            return default
        if not isinstance(raw, str):
            raise RuntimeError(f"order payload status must be string, got {type(raw).__name__}")
        text = raw.strip().lower()
        return text if text else default

    @classmethod
    def _parse_order_id(cls, payload: ClobOrderPayload) -> str:
        raw = payload.get("orderID")
        if raw is None:
            raw = payload.get("id")
        if raw is None:
            raise RuntimeError("post_order response missing order id")
        order_id = str(raw).strip()
        if not order_id:
            raise RuntimeError("post_order response has empty order id")
        return order_id

    @classmethod
    def _parse_fill(cls, payload: ClobOrderPayload, *, context: str) -> ParsedFill:
        filled_size = cls._parse_number(
            payload.get("filledSize"), field="filledSize", context=context
        )
        # Prefer avgPrice from CLOB order state; use filledPrice only when explicitly provided.
        filled_price = cls._parse_number(
            payload.get("avgPrice"), field="avgPrice", context=context
        )
        if filled_price <= 0 and ("filledPrice" in payload):
            filled_price = cls._parse_number(
                payload.get("filledPrice"), field="filledPrice", context=context
            )
        fee_paid = cls._parse_number(payload.get("fee"), field="fee", context=context)
        if filled_size < 0:
            raise RuntimeError(f"{context}.filledSize cannot be negative")
        if filled_price < 0:
            raise RuntimeError(f"{context}.avgPrice/filledPrice cannot be negative")
        return ParsedFill(
            filled_size=filled_size,
            filled_price=filled_price,
            fee_paid=fee_paid,
        )

    def _infer_signature_type(self, signer_address: str, funder_address: str) -> int:
        if self.config.poly_signature_type is not None:
            return int(self.config.poly_signature_type)
        if self._normalize_address(funder_address) != self._normalize_address(
            signer_address
        ):
            # Default contract-wallet flow to Polymarket proxy signatures.
            # Gnosis safe signatures should be explicitly overridden via
            # POLY_SIGNATURE_TYPE=2 when needed.
            return 1
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
        if hasattr(self.client, "create_or_derive_api_creds") and hasattr(
            self.client, "set_api_creds"
        ):
            try:
                creds = self._derive_api_creds(self.client)
            except Exception as exc:
                payload = self._exception_payload(exc)
                creds = self._api_creds_from_env()
                if creds is None:
                    raise RuntimeError(
                        "Unable to derive Polymarket API credentials. "
                        f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type} "
                        f"error={payload}"
                    ) from exc
                LOGGER.warning(
                    "Using CLOB API creds from environment fallback after derive failure"
                )
            if creds is None:
                creds = self._api_creds_from_env()
                if creds is None:
                    raise RuntimeError(
                        "Unable to derive Polymarket API credentials from wallet key. "
                        f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type}"
                    )
                LOGGER.warning("Using CLOB API creds from environment fallback")
            self.client.set_api_creds(creds)

    def preflight(self) -> None:
        self._bootstrap_client()
        if self._preflight_done:
            return
        try:
            self.client.get_api_keys()
            self._preflight_done = True
        except Exception as exc:  # pragma: no cover - live path
            payload = self._exception_payload(exc)
            raise RuntimeError(
                "Live auth preflight failed (no retries). "
                "Check POLY_PRIVATE_KEY/POLY_PROXY_ADDRESS or set CLOB API creds env vars. "
                f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type} "
                f"error={payload}"
            ) from exc

    def _require_preflight(self) -> None:
        self._bootstrap_client()
        if self._preflight_done:
            return
        raise RuntimeError(
            "Live executor not authenticated. Run preflight() once at startup before trading."
        )

    @classmethod
    def _parse_balance_allowance(
        cls, payload: object, *, context: str
    ) -> float:
        data = cls._ensure_payload_object(payload, context=context)
        if "balance" not in data:
            raise RuntimeError(f"{context} payload missing balance")
        raw_balance = data.get("balance")
        value = cls._parse_number(raw_balance, field="balance", context=context)
        value = cls._normalize_balance_units(raw_balance, value)
        if value < 0:
            raise RuntimeError(f"{context}.balance cannot be negative")
        return value

    @staticmethod
    def _normalize_balance_units(raw_value: object, parsed_value: float) -> float:
        if parsed_value <= 0:
            return 0.0
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return 0.0
            lowered = text.lower()
            if "." in text or "e" in lowered:
                return parsed_value
            # CLOB balance_allowance integer balances are 6-decimal base units.
            return parsed_value / 1_000_000.0
        if isinstance(raw_value, int):
            # CLOB balance_allowance integer balances are 6-decimal base units.
            return parsed_value / 1_000_000.0
        return parsed_value

    def get_collateral_balance(self) -> float:
        self._require_preflight()
        try:
            from py_clob_client.clob_types import (  # type: ignore
                AssetType,
                BalanceAllowanceParams,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "py-clob-client clob_types unavailable for balance allowance call"
            ) from exc

        try:
            payload = self.client.get_balance_allowance(  # type: ignore[union-attr]
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=self._signature_type,
                )
            )
        except Exception as exc:  # pragma: no cover - live path
            raise RuntimeError(
                f"get_balance_allowance collateral failed: {self._exception_payload(exc)}"
            ) from exc
        return self._parse_balance_allowance(
            payload, context="get_balance_allowance.collateral"
        )

    def get_token_balance(self, token_id: str) -> float:
        self._require_preflight()
        cleaned_token_id = str(token_id or "").strip()
        if not cleaned_token_id:
            raise RuntimeError("token_id is required for token balance call")
        try:
            from py_clob_client.clob_types import (  # type: ignore
                AssetType,
                BalanceAllowanceParams,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "py-clob-client clob_types unavailable for balance allowance call"
            ) from exc

        try:
            payload = self.client.get_balance_allowance(  # type: ignore[union-attr]
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=cleaned_token_id,
                    signature_type=self._signature_type,
                )
            )
        except Exception as exc:  # pragma: no cover - live path
            raise RuntimeError(
                "get_balance_allowance conditional failed "
                f"token_id={cleaned_token_id}: {self._exception_payload(exc)}"
            ) from exc
        return self._parse_balance_allowance(
            payload, context="get_balance_allowance.conditional"
        )

    @staticmethod
    def _ensure_trade_payload_object(payload: object) -> TradePayload | None:
        if not isinstance(payload, dict):
            return None
        return cast(TradePayload, payload)

    def _parse_maker_trade_fills(
        self,
        payload: object,
        *,
        after_ts: int,
    ) -> list[MakerTradeFill]:
        if not isinstance(payload, list):
            raise RuntimeError("get_trades response must be a list")
        wallet_addresses = {
            self._normalize_address(self._signer_address),
            self._normalize_address(self._funder_address),
        }
        wallet_addresses = {addr for addr in wallet_addresses if addr}
        fills: list[MakerTradeFill] = []
        seen: set[tuple[str, str]] = set()
        min_ts = max(0, int(after_ts))
        for raw_trade in payload:
            trade = self._ensure_trade_payload_object(raw_trade)
            if trade is None:
                continue
            trade_id = str(trade.get("id") or "").strip()
            if not trade_id:
                continue
            match_time = int(
                self._parse_number(
                    trade.get("match_time"),
                    field="match_time",
                    context="trade",
                )
            )
            if match_time < min_ts:
                continue
            maker_orders = trade.get("maker_orders")
            if not isinstance(maker_orders, list):
                continue
            for raw_order in maker_orders:
                if not isinstance(raw_order, dict):
                    continue
                maker_address = self._normalize_address(
                    str(raw_order.get("maker_address") or "")
                )
                if maker_address not in wallet_addresses:
                    continue
                order_id = str(raw_order.get("order_id") or "").strip()
                token_id = str(raw_order.get("asset_id") or "").strip()
                side_text = str(raw_order.get("side") or "").strip().lower()
                if not order_id or not token_id:
                    continue
                if side_text == "buy":
                    side = Side.BUY
                elif side_text == "sell":
                    side = Side.SELL
                else:
                    continue
                size = self._parse_number(
                    raw_order.get("matched_amount"),
                    field="matched_amount",
                    context="trade.maker_order",
                )
                price = self._parse_number(
                    raw_order.get("price"),
                    field="price",
                    context="trade.maker_order",
                )
                fee_rate_bps = int(
                    self._parse_number(
                        raw_order.get("fee_rate_bps"),
                        field="fee_rate_bps",
                        context="trade.maker_order",
                    )
                )
                if size <= 0 or price <= 0:
                    continue
                dedupe_key = (trade_id, order_id)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                fills.append(
                    MakerTradeFill(
                        trade_id=trade_id,
                        order_id=order_id,
                        token_id=token_id,
                        side=side,
                        size=size,
                        price=price,
                        fee_rate_bps=max(0, fee_rate_bps),
                        match_time=match_time,
                    )
                )
        fills.sort(key=lambda item: (item.match_time, item.trade_id, item.order_id))
        return fills

    def get_recent_maker_trade_fills(self, *, after_ts: int) -> list[MakerTradeFill]:
        self._require_preflight()
        query_after = max(0, int(after_ts))
        try:
            from py_clob_client.clob_types import TradeParams  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "py-clob-client clob_types unavailable for trade fill call"
            ) from exc
        maker_address = self._funder_address or self._signer_address
        try:
            payload = self.client.get_trades(  # type: ignore[union-attr]
                TradeParams(
                    maker_address=maker_address,
                    after=query_after,
                ),
                next_cursor="MA==",
            )
        except Exception as exc:  # pragma: no cover - live path
            raise RuntimeError(f"get_trades failed: {self._exception_payload(exc)}") from exc
        return self._parse_maker_trade_fills(payload, after_ts=query_after)

    def place_order(self, intent: OrderIntent, books: dict[str, OrderBookSnapshot]) -> OrderResult:
        del books
        now = datetime.now(tz=timezone.utc)
        self._require_preflight()
        try:
            from py_clob_client.clob_types import (  # type: ignore
                OrderArgs,
                OrderType,
            )

            side = "BUY" if intent.side == Side.BUY else "SELL"
            order_args = OrderArgs(
                price=float(intent.price),
                size=float(intent.size),
                side=side,
                token_id=intent.token_id,
            )
            signed_order = self.client.create_order(order_args)

            order_type = getattr(OrderType, intent.tif.value, getattr(OrderType, "GTC"))
            response = self.client.post_order(
                signed_order, order_type, post_only=bool(intent.post_only)
            )
            response_payload = self._ensure_payload_object(
                response, context="post_order"
            )
            status = self._parse_status(response_payload, default="accepted")
            order_id = self._parse_order_id(response_payload)

            order_state_payload: ClobOrderPayload | None = None
            if order_id and intent.tif in {TimeInForce.IOC, TimeInForce.FOK}:
                try:
                    details = self.client.get_order(order_id)
                    order_state_payload = self._ensure_payload_object(
                        details, context="get_order"
                    )
                    status = self._parse_status(order_state_payload, default=status)
                except Exception:
                    order_state_payload = None

            fill_source = order_state_payload or response_payload
            fill = self._parse_fill(
                fill_source,
                context="order_state" if order_state_payload is not None else "post_order",
            )
            filled_size = fill.filled_size
            filled_price = fill.filled_price
            fee_paid = fill.fee_paid
            if (
                intent.tif in {TimeInForce.IOC, TimeInForce.FOK}
                and order_id
                and filled_size <= 0
                and status in {"filled", "matched", "executed", "completed", "complete"}
            ):
                for _ in range(2):
                    time.sleep(0.12)
                    try:
                        details_retry = self.client.get_order(order_id)
                    except Exception:
                        continue
                    try:
                        details_retry_payload = self._ensure_payload_object(
                            details_retry, context="get_order"
                        )
                    except Exception:
                        continue
                    retry_status = self._parse_status(details_retry_payload, default=status)
                    retry_fill = self._parse_fill(
                        details_retry_payload, context="order_state"
                    )
                    if retry_fill.filled_size > 0:
                        order_state_payload = details_retry_payload
                        status = retry_status
                        filled_size = retry_fill.filled_size
                        filled_price = retry_fill.filled_price
                        fee_paid = retry_fill.fee_paid
                        break

            raw_payload: dict[str, Any] = {
                "post_order": response_payload,
                "signature_type": self._signature_type,
            }
            if order_state_payload is not None:
                raw_payload["order_state"] = order_state_payload
            raw_payload["fill_source"] = (
                "order_state" if order_state_payload is not None else "post_order"
            )
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
            if self._is_invalid_signature_payload(payload):
                raise RuntimeError(
                    "Live auth failed during order placement (no retries). "
                    f"signer={self._signer_address} funder={self._funder_address} signature_type={self._signature_type} "
                    f"error={payload}"
                ) from exc
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

    def get_order_result(self, order_id: str, intent: OrderIntent) -> OrderResult | None:
        self._require_preflight()
        now = datetime.now(tz=timezone.utc)
        try:  # pragma: no cover - live path
            details = self.client.get_order(order_id)
        except Exception:
            return None
        try:
            details_payload = self._ensure_payload_object(details, context="get_order")
        except Exception:
            return None
        status = self._parse_status(details_payload, default="live")
        fill = self._parse_fill(details_payload, context="get_order")
        return OrderResult(
            order_id=order_id,
            market_id=intent.market_id,
            token_id=intent.token_id,
            side=intent.side,
            price=intent.price,
            size=intent.size,
            status=status,
            filled_size=fill.filled_size,
            filled_price=fill.filled_price,
            fee_paid=fill.fee_paid,
            engine=intent.engine,
            created_at=now,
            raw={
                "order_state": details_payload,
                "reconciled": True,
                "signature_type": self._signature_type,
                "fill_source": "get_order",
            },
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
    def _receipt_map(receipt: Any) -> dict[str, Any]:
        if isinstance(receipt, dict):
            return receipt
        if hasattr(receipt, "items"):
            try:
                return dict(receipt.items())
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError("unreadable receipt mapping") from exc
        raise RuntimeError(f"unsupported receipt type: {type(receipt).__name__}")

    @staticmethod
    def _parse_receipt_int(value: Any, *, field: str) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise RuntimeError(f"receipt field {field} is empty")
            try:
                return int(text, 16) if text.startswith("0x") else int(text)
            except ValueError as exc:
                raise RuntimeError(
                    f"receipt field {field} is not an integer: {value!r}"
                ) from exc
        raise RuntimeError(
            f"receipt field {field} has unsupported type: {type(value).__name__}"
        )

    @staticmethod
    def _parse_receipt_hex(value: Any, *, field: str) -> str:
        if isinstance(value, (bytes, bytearray)):
            return "0x" + bytes(value).hex()
        hex_method = getattr(value, "hex", None)
        if callable(hex_method):
            raw = hex_method()
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            return text if text.startswith("0x") else f"0x{text}"
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise RuntimeError(f"receipt field {field} is empty")
            return text
        raise RuntimeError(
            f"receipt field {field} has unsupported type: {type(value).__name__}"
        )

    @classmethod
    def _receipt_payload(cls, receipt: Any) -> dict[str, Any]:
        raw = cls._receipt_map(receipt)
        if "status" not in raw:
            raise RuntimeError("receipt missing status")

        payload: dict[str, Any] = {
            "status": cls._parse_receipt_int(raw.get("status"), field="status"),
        }

        for key in ("transactionHash", "blockHash"):
            if key not in raw or raw.get(key) is None:
                continue
            payload[key] = cls._parse_receipt_hex(raw.get(key), field=key)

        for key in ("blockNumber", "gasUsed", "effectiveGasPrice"):
            if key not in raw or raw.get(key) is None:
                continue
            payload[key] = cls._parse_receipt_int(raw.get(key), field=key)

        return payload

    def _web3(self):
        if self._w3 is not None:
            return self._w3
        try:
            from web3 import Web3
            from web3.middleware import geth_poa_middleware
        except Exception as exc:
            raise RuntimeError(
                "web3 is required for merge/redeem. Install with `pip install web3`."
            ) from exc

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
        self._require_preflight()
        return self._normalize_address(self._funder_address) != self._normalize_address(
            self._signer_address
        )

    def _wrap_proxy_call(self, *, target_address: str, calldata: bytes):
        from web3 import Web3

        proxy_factory = str(self.config.poly_proxy_factory_address or "").strip()
        if not proxy_factory:
            raise RuntimeError(
                "POLY_PROXY_FACTORY_ADDRESS is required for proxy-wallet settlement"
            )
        w3 = self._web3()
        factory = w3.eth.contract(
            address=Web3.to_checksum_address(proxy_factory), abi=PROXY_FACTORY_ABI
        )
        return factory.functions.proxy(
            [(1, Web3.to_checksum_address(target_address), 0, calldata)]
        )

    def _send_function_tx(self, fn) -> str:
        self._require_preflight()
        if not self.config.poly_private_key:
            raise RuntimeError("POLY_PRIVATE_KEY missing")
        try:
            from eth_account import Account
        except Exception as exc:
            raise RuntimeError(
                "eth-account is required for merge/redeem. Install with `pip install eth-account`."
            ) from exc

        signer = Account.from_key(self.config.poly_private_key).address
        if self._signer_address and self._normalize_address(
            signer
        ) != self._normalize_address(self._signer_address):
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
        raw_tx = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        if raw_tx is None:
            raise RuntimeError("unable to access signed raw transaction")
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        return tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)

    def _resolve_pending_merge(self, market_id: str) -> SettlementResult | None:
        pending: PendingMergePayload | None = self._pending_merges.get(market_id)
        if pending is None:
            return None
        tx_hash = str(pending.get("tx_hash") or "")
        receipt = self._read_receipt(tx_hash)
        if receipt is None:
            return SettlementResult(
                status="pending", merged_size=0.0, raw=dict(pending)
            )

        payload = dict(pending)
        try:
            receipt_payload = self._receipt_payload(receipt)
        except Exception as exc:
            payload["receipt_parse_error"] = self._exception_payload(exc)
            return SettlementResult(
                status="pending",
                merged_size=0.0,
                raw=payload,
            )
        self._pending_merges.pop(market_id, None)
        status = int(receipt_payload["status"])
        payload["receipt"] = receipt_payload
        if status == 1:
            return SettlementResult(
                status="merged",
                merged_size=float(pending.get("merged_size", 0.0)),
                raw=payload,
            )
        payload["reason"] = "merge transaction reverted"
        return SettlementResult(status="error", raw=payload)

    def _resolve_pending_redeem(self) -> SettlementResult | None:
        for condition_id, pending in list(self._pending_redeems.items()):
            tx_hash = str(pending.get("tx_hash") or "")
            receipt = self._read_receipt(tx_hash)
            if receipt is None:
                return SettlementResult(status="pending", raw=dict(pending))

            payload = dict(pending)
            try:
                receipt_payload = self._receipt_payload(receipt)
            except Exception as exc:
                payload["receipt_parse_error"] = self._exception_payload(exc)
                return SettlementResult(status="pending", raw=payload)
            self._pending_redeems.pop(condition_id, None)
            status = int(receipt_payload["status"])
            payload["receipt"] = receipt_payload
            if status == 1:
                return SettlementResult(
                    status="redeemed", redeemed_usdc=0.0, raw=payload
                )
            payload["reason"] = "redeem transaction reverted"
            return SettlementResult(status="error", raw=payload)
        return None

    def register_condition(
        self,
        *,
        condition_id: str,
        is_neg_risk: bool,
        market_end_ts: float | None = None,
    ) -> None:
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
        self._known_conditions[normalized] = (
            existing_neg_risk or bool(is_neg_risk),
            max(existing_end, end_ts),
        )

    def _resolve_neg_risk_flag(self, *, token_id: str, hinted: bool | None) -> bool:
        if hinted is not None:
            return bool(hinted)
        self._require_preflight()
        try:
            return bool(self.client.get_neg_risk(token_id))
        except Exception:
            return False

    def _build_merge_function(
        self, *, condition_id: str, amount_raw: int, is_neg_risk: bool
    ):
        from web3 import Web3

        w3 = self._web3()
        config = self._resolve_contract_config(neg_risk=is_neg_risk)
        if is_neg_risk:
            adapter = Web3.to_checksum_address(str(config.exchange))
            contract = w3.eth.contract(address=adapter, abi=NEG_RISK_ADAPTER_ABI)
            return contract.functions.mergePositions(
                condition_id, int(amount_raw)
            ), adapter

        conditional = Web3.to_checksum_address(str(config.conditional_tokens))
        collateral = Web3.to_checksum_address(str(config.collateral))
        contract = w3.eth.contract(address=conditional, abi=CONDITIONAL_TOKENS_ABI)
        return (
            contract.functions.mergePositions(
                collateral, ZERO_BYTES32, condition_id, [1, 2], int(amount_raw)
            ),
            conditional,
        )

    def _build_redeem_function(self, *, condition_id: str):
        from web3 import Web3

        w3 = self._web3()
        config = self._resolve_contract_config(neg_risk=False)
        conditional = Web3.to_checksum_address(str(config.conditional_tokens))
        collateral = Web3.to_checksum_address(str(config.collateral))
        contract = w3.eth.contract(address=conditional, abi=CONDITIONAL_TOKENS_ABI)
        return contract.functions.redeemPositions(
            collateral, ZERO_BYTES32, condition_id, [1, 2]
        ), conditional

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
            return SettlementResult(
                status="error", raw={"reason": "merge size must be > 0"}
            )

        pending = self._resolve_pending_merge(market_id)
        if pending is not None:
            return pending
        if not condition_id:
            return SettlementResult(
                status="unsupported", raw={"reason": "missing condition_id"}
            )

        try:
            normalized_condition = self._normalize_condition_id(condition_id)
        except Exception as exc:
            return SettlementResult(status="error", raw=self._exception_payload(exc))

        neg_risk = self._resolve_neg_risk_flag(
            token_id=primary_token_id, hinted=is_neg_risk
        )
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
        pending_payload: PendingMergePayload = {
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
            return SettlementResult(
                status="ok", raw={"reason": "no redeem-eligible conditions"}
            )

        candidates.sort()
        idx = self._redeem_cursor % len(candidates)
        self._redeem_cursor += 1
        condition_id = candidates[idx]

        try:
            redeem_fn, redeem_target = self._build_redeem_function(
                condition_id=condition_id
            )
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

        pending_payload: PendingRedeemPayload = {
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

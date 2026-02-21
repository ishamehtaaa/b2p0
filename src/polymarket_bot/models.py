from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any


class Timeframe(str, Enum):
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR = "1h"
    UNKNOWN = "unknown"

    @staticmethod
    def from_tag(tag_id: int) -> "Timeframe":
        if tag_id == 102892:
            return Timeframe.FIVE_MIN
        if tag_id == 102467:
            return Timeframe.FIFTEEN_MIN
        if tag_id == 102175:
            return Timeframe.ONE_HOUR
        return Timeframe.UNKNOWN


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def parse_ts(value: str | None) -> datetime:
    if value is None or value == "":
        return utc_now()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def parse_float(raw: Any, default: float = 0.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_token_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                data = json.loads(stripped)
                return [str(x) for x in data]
            except json.JSONDecodeError:
                return []
        if stripped:
            return [stripped]
    return []


@dataclass
class MarketInfo:
    market_id: str
    slug: str
    question: str
    condition_id: str
    tag_id: int
    timeframe: Timeframe
    start_time: datetime
    end_time: datetime
    liquidity_num: float
    volume_24h: float
    best_bid: float
    best_ask: float
    spread: float
    outcomes: list[str]
    token_ids: list[str]
    primary_token_id: str
    secondary_token_id: str
    primary_label: str
    secondary_label: str
    is_neg_risk: bool = False
    order_min_size: float = 5.0

    @property
    def seconds_to_end(self) -> float:
        return (self.end_time - utc_now()).total_seconds()

    @property
    def seconds_to_start(self) -> float:
        return (self.start_time - utc_now()).total_seconds()

    @property
    def market_mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        if self.best_ask > 0:
            return self.best_ask
        if self.best_bid > 0:
            return self.best_bid
        return 0.5


@dataclass
class OrderBookLevel:
    price: float
    size: float

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    token_id: str
    timestamp_ms: int
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread(self) -> float:
        if self.best_bid <= 0 or self.best_ask <= 0:
            return 0.0
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        if self.best_ask > 0:
            return self.best_ask
        if self.best_bid > 0:
            return self.best_bid
        return 0.5

    def top_depth_usdc(self, levels: int = 2) -> tuple[float, float]:
        bid_depth = sum(level.notional for level in self.bids[:levels])
        ask_depth = sum(level.notional for level in self.asks[:levels])
        return bid_depth, ask_depth

    def top_size_imbalance(self, levels: int = 2) -> float:
        bid_size = sum(level.size for level in self.bids[:levels])
        ask_size = sum(level.size for level in self.asks[:levels])
        total = bid_size + ask_size
        if total <= 0:
            return 0.0
        return (bid_size - ask_size) / total


@dataclass
class FeeInfo:
    token_id: str
    base_fee: int
    fetched_at: datetime


@dataclass
class MarketSnapshot:
    market: MarketInfo
    primary_book: OrderBookSnapshot
    secondary_book: OrderBookSnapshot
    primary_fee: FeeInfo
    secondary_fee: FeeInfo

    @property
    def primary_mid(self) -> float:
        return self.primary_book.mid

    @property
    def secondary_mid(self) -> float:
        return self.secondary_book.mid

    @property
    def imbalance(self) -> float:
        return self.primary_book.top_size_imbalance(levels=2)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    GTC = "GTC"
    FOK = "FOK"
    IOC = "IOC"


@dataclass
class OrderIntent:
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    tif: TimeInForce
    post_only: bool
    engine: str
    expected_edge: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class OrderResult:
    order_id: str
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    status: str
    filled_size: float
    filled_price: float
    fee_paid: float
    engine: str
    created_at: datetime
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.filled_size > 0

    @property
    def is_error(self) -> bool:
        return self.status in {"rejected", "error"}


@dataclass
class PositionState:
    token_id: str
    market_id: str
    size: float = 0.0
    average_price: float = 0.0
    mark_price: float = 0.5

    @property
    def notional(self) -> float:
        return abs(self.size * self.mark_price)

    @property
    def unrealized_pnl(self) -> float:
        if self.size == 0:
            return 0.0
        return self.size * (self.mark_price - self.average_price)


@dataclass
class RiskState:
    current_equity: float
    daily_drawdown_pct: float
    total_exposure: float
    halted: bool
    halted_reason: str
    consecutive_exec_errors: int

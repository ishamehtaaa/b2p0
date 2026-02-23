from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from polymarket_bot.models import OrderIntent

QuoteKey: TypeAlias = tuple[str, str, str, str, str, str]


@dataclass
class QuoteOrderState:
    intent: OrderIntent
    key: QuoteKey | None = None
    cum_filled_size: float = 0.0
    cum_fee_paid: float = 0.0
    cum_notional: float = 0.0
    retired_ts: float | None = None


@dataclass
class AlphaOpenLeg:
    market_id: str
    entry_token_id: str
    opposite_token_id: str
    entry_side: str
    timeframe: str
    seconds_to_end_at_entry: float
    entry_price: float
    remaining_size: float
    opened_ts: float


@dataclass
class MarketBuyExecutionState:
    last_side: str = ""
    run_len: int = 0
    last_primary_price: float = 0.0
    last_secondary_price: float = 0.0
    updated_ts: float = 0.0

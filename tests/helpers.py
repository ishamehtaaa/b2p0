from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.config import load_config
from polymarket_bot.models import (  # noqa: E402
    FeeInfo,
    MarketInfo,
    MarketSnapshot,
    OrderBookLevel,
    OrderBookSnapshot,
    Timeframe,
)


def test_config(**kwargs):
    cfg = load_config()
    return replace(cfg, **kwargs)


def build_snapshot(
    timeframe: Timeframe,
    slug: str = "btc-updown-5m-test",
    spread: float = 0.02,
    primary_fee_bps: int = 0,
    secondary_fee_bps: int = 0,
    primary_bid: float = 0.49,
    primary_ask: float = 0.51,
    secondary_bid: float = 0.49,
    secondary_ask: float = 0.51,
) -> MarketSnapshot:
    start = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
    end = datetime.now(tz=timezone.utc) + timedelta(minutes=30)
    market = MarketInfo(
        market_id="m1",
        slug=slug,
        question="Bitcoin Up or Down",
        condition_id="0x" + ("11" * 32),
        tag_id=102175 if timeframe == Timeframe.ONE_HOUR else 102892,
        timeframe=timeframe,
        start_time=start,
        end_time=end,
        liquidity_num=10000.0,
        volume_24h=1000.0,
        best_bid=primary_bid,
        best_ask=primary_ask,
        spread=spread,
        outcomes=["Up", "Down"],
        token_ids=["token-up", "token-down"],
        primary_token_id="token-up",
        secondary_token_id="token-down",
        primary_label="Up",
        secondary_label="Down",
        is_neg_risk=False,
        order_min_size=5.0,
    )
    primary_book = OrderBookSnapshot(
        token_id="token-up",
        timestamp_ms=0,
        bids=[OrderBookLevel(price=primary_bid, size=1000.0)],
        asks=[OrderBookLevel(price=primary_ask, size=1000.0)],
    )
    secondary_book = OrderBookSnapshot(
        token_id="token-down",
        timestamp_ms=0,
        bids=[OrderBookLevel(price=secondary_bid, size=1000.0)],
        asks=[OrderBookLevel(price=secondary_ask, size=1000.0)],
    )
    return MarketSnapshot(
        market=market,
        primary_book=primary_book,
        secondary_book=secondary_book,
        primary_fee=FeeInfo(token_id="token-up", base_fee=primary_fee_bps, fetched_at=datetime.now(tz=timezone.utc)),
        secondary_fee=FeeInfo(
            token_id="token-down",
            base_fee=secondary_fee_bps,
            fetched_at=datetime.now(tz=timezone.utc),
        ),
    )

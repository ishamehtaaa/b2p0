from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TypedDict, cast
import re
from urllib.error import HTTPError

from polymarket_bot.http_utils import get_json
from polymarket_bot.learning import PairTimingLearner
from polymarket_bot.models import Timeframe
from polymarket_bot.pricing import clamp


class ActivityEvent(TypedDict, total=False):
    timestamp: int | float | str
    conditionId: str
    type: str
    side: str
    price: object
    transactionHash: str
    asset: str
    outcome: str
    outcomeIndex: object
    title: str
    slug: str
    eventSlug: str


@dataclass(frozen=True)
class SeedTrade:
    condition_id: str
    market_id: str
    entry_side: str
    timeframe: Timeframe
    seconds_to_end: float
    price: float
    timestamp: int
    transaction_hash: str
    asset: str


@dataclass
class LearningSeedResult:
    pages_fetched: int = 0
    events_seen: int = 0
    trade_events_seen: int = 0
    trades_imported: int = 0
    trades_skipped: int = 0
    duplicate_trades: int = 0
    seeded_pair_events: int = 0
    seeded_proxy_events: int = 0
    skipped_unknown_timeframe: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "pages_fetched": self.pages_fetched,
            "events_seen": self.events_seen,
            "trade_events_seen": self.trade_events_seen,
            "trades_imported": self.trades_imported,
            "trades_skipped": self.trades_skipped,
            "duplicate_trades": self.duplicate_trades,
            "seeded_pair_events": self.seeded_pair_events,
            "seeded_proxy_events": self.seeded_proxy_events,
            "skipped_unknown_timeframe": self.skipped_unknown_timeframe,
        }


_FIVE_MIN_PATTERN = re.compile(r"\b(5m|5\s*min(?:ute)?s?)\b")
_FIFTEEN_MIN_PATTERN = re.compile(r"\b(15m|15\s*min(?:ute)?s?)\b")
_ONE_HOUR_PATTERN = re.compile(r"\b(1h|1\s*hour|hourly)\b")


def _window_seconds(timeframe: Timeframe) -> int:
    if timeframe == Timeframe.FIVE_MIN:
        return 300
    if timeframe == Timeframe.FIFTEEN_MIN:
        return 900
    if timeframe == Timeframe.ONE_HOUR:
        return 3600
    return 0


def _seconds_to_end_from_timestamp(timestamp: int, timeframe: Timeframe) -> float:
    window = _window_seconds(timeframe)
    if window <= 0:
        return 0.0
    remainder = timestamp % window
    if remainder == 0:
        return 0.0
    return float(window - remainder)


def _parse_timestamp(raw: int | float | str | object) -> int | None:
    if isinstance(raw, int):
        ts = raw
    elif isinstance(raw, float):
        ts = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            ts = int(float(text))
        except ValueError:
            return None
    else:
        return None
    if ts <= 0:
        return None
    return ts


def _parse_price(raw: object) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


def _parse_entry_side(event: ActivityEvent) -> str:
    outcome = str(event.get("outcome") or "").strip().lower()
    if outcome:
        if outcome in {"up", "yes"} or " up" in outcome or outcome.startswith("up "):
            return "primary"
        if outcome in {"down", "no"} or " down" in outcome or outcome.startswith("down "):
            return "secondary"

    raw_outcome_index = event.get("outcomeIndex")
    outcome_index: int | None
    if isinstance(raw_outcome_index, int):
        outcome_index = raw_outcome_index
    elif isinstance(raw_outcome_index, float):
        outcome_index = int(raw_outcome_index)
    elif isinstance(raw_outcome_index, str):
        text = raw_outcome_index.strip()
        if not text:
            outcome_index = None
        else:
            try:
                outcome_index = int(text)
            except ValueError:
                outcome_index = None
    else:
        outcome_index = None

    if outcome_index == 0:
        return "primary"
    if outcome_index == 1:
        return "secondary"
    return ""


def _infer_timeframe(event: ActivityEvent) -> Timeframe:
    title = str(event.get("title") or "")
    slug = str(event.get("slug") or "")
    event_slug = str(event.get("eventSlug") or "")
    text = f"{title} {slug} {event_slug}".lower()
    if _FIVE_MIN_PATTERN.search(text):
        return Timeframe.FIVE_MIN
    if _FIFTEEN_MIN_PATTERN.search(text):
        return Timeframe.FIFTEEN_MIN
    if _ONE_HOUR_PATTERN.search(text):
        return Timeframe.ONE_HOUR
    return Timeframe.UNKNOWN


def _iter_activity(
    *,
    data_api_url: str,
    user: str,
    page_size: int,
    max_pages: int,
) -> tuple[list[ActivityEvent], int]:
    events: list[ActivityEvent] = []
    pages_fetched = 0
    for page_index in range(max_pages):
        try:
            payload = get_json(
                f"{data_api_url}/activity",
                params={
                    "user": user,
                    "limit": str(page_size),
                    "offset": str(page_index * page_size),
                },
                timeout=3.0,
            )
        except Exception as exc:
            message = str(exc)
            is_http_400 = isinstance(exc, HTTPError) and exc.code == 400
            if "HTTP Error 400" in message:
                is_http_400 = True
            if page_index > 0 and is_http_400:
                break
            raise
        pages_fetched += 1
        if not isinstance(payload, list):
            break
        if not payload:
            break
        for item in payload:
            if isinstance(item, dict):
                events.append(cast(ActivityEvent, item))
        if len(payload) < page_size:
            break
    return events, pages_fetched


def _extract_seed_trades(
    events: list[ActivityEvent],
    *,
    forced_timeframe: Timeframe | None,
) -> tuple[list[SeedTrade], LearningSeedResult]:
    result = LearningSeedResult(events_seen=len(events))
    seen_keys: set[str] = set()
    trades: list[SeedTrade] = []

    for event in events:
        event_type = str(event.get("type") or "").strip().upper()
        if event_type != "TRADE":
            continue
        result.trade_events_seen += 1
        side = str(event.get("side") or "").strip().upper()
        if side != "BUY":
            result.trades_skipped += 1
            continue

        condition_id = str(event.get("conditionId") or "").strip().lower()
        if not condition_id:
            result.trades_skipped += 1
            continue

        entry_side = _parse_entry_side(event)
        if entry_side not in {"primary", "secondary"}:
            result.trades_skipped += 1
            continue

        timestamp = _parse_timestamp(event.get("timestamp"))
        if timestamp is None:
            result.trades_skipped += 1
            continue

        price = _parse_price(event.get("price"))
        if not (0.0 < price < 1.0):
            result.trades_skipped += 1
            continue

        timeframe = forced_timeframe if forced_timeframe is not None else _infer_timeframe(event)
        if timeframe == Timeframe.UNKNOWN:
            result.skipped_unknown_timeframe += 1
            result.trades_skipped += 1
            continue

        tx_hash = str(event.get("transactionHash") or "").strip().lower()
        asset = str(event.get("asset") or "").strip().lower()
        dedupe_key = f"{tx_hash}:{asset}" if tx_hash else (
            f"{condition_id}:{timestamp}:{entry_side}:{price:.6f}"
        )
        if dedupe_key in seen_keys:
            result.duplicate_trades += 1
            continue
        seen_keys.add(dedupe_key)

        trades.append(
            SeedTrade(
                condition_id=condition_id,
                market_id=condition_id,
                entry_side=entry_side,
                timeframe=timeframe,
                seconds_to_end=_seconds_to_end_from_timestamp(timestamp, timeframe),
                price=price,
                timestamp=timestamp,
                transaction_hash=tx_hash,
                asset=asset,
            )
        )

    result.trades_imported = len(trades)
    return trades, result


def seed_pair_learning_from_trader_history(
    *,
    learner: PairTimingLearner,
    data_api_url: str,
    user: str,
    page_size: int = 100,
    max_pages: int = 40,
    forced_timeframe: Timeframe | None = None,
    pair_gap_seconds: float = 180.0,
    proxy_buffer: float = 0.01,
    proxy_delay_seconds: float = 20.0,
    success_cost_threshold: float = 1.005,
) -> LearningSeedResult:
    safe_page_size = max(1, min(200, int(page_size)))
    safe_max_pages = max(1, int(max_pages))
    safe_pair_gap = max(0.0, float(pair_gap_seconds))
    safe_proxy_buffer = clamp(float(proxy_buffer), 0.0, 0.20)
    safe_proxy_delay = max(0.0, float(proxy_delay_seconds))

    events, pages_fetched = _iter_activity(
        data_api_url=data_api_url,
        user=user,
        page_size=safe_page_size,
        max_pages=safe_max_pages,
    )
    trades, result = _extract_seed_trades(
        events,
        forced_timeframe=forced_timeframe,
    )
    result.pages_fetched = pages_fetched

    groups: dict[tuple[str, Timeframe], list[SeedTrade]] = defaultdict(list)
    for trade in trades:
        groups[(trade.condition_id, trade.timeframe)].append(trade)

    success_threshold = float(success_cost_threshold)

    for (_condition_id, _timeframe), grouped in groups.items():
        ordered = sorted(grouped, key=lambda item: item.timestamp)
        unmatched_primary: deque[SeedTrade] = deque()
        unmatched_secondary: deque[SeedTrade] = deque()

        for trade in ordered:
            opposite_queue = (
                unmatched_secondary if trade.entry_side == "primary" else unmatched_primary
            )
            while opposite_queue and (trade.timestamp - opposite_queue[0].timestamp) > safe_pair_gap:
                opposite_queue.popleft()
            if opposite_queue:
                opposite = opposite_queue.pop()
                pair_cost = trade.price + opposite.price
                hedge_delay = float(abs(trade.timestamp - opposite.timestamp))
                success = pair_cost <= success_threshold
                learner.observe(
                    market_id=trade.market_id,
                    timeframe=trade.timeframe,
                    side=trade.entry_side,
                    seconds_to_end=trade.seconds_to_end,
                    pair_price_cost=pair_cost,
                    hedge_delay_seconds=hedge_delay,
                    success=success,
                    source="seed_history_pair",
                )
                learner.observe(
                    market_id=opposite.market_id,
                    timeframe=opposite.timeframe,
                    side=opposite.entry_side,
                    seconds_to_end=opposite.seconds_to_end,
                    pair_price_cost=pair_cost,
                    hedge_delay_seconds=hedge_delay,
                    success=success,
                    source="seed_history_pair",
                )
                result.seeded_pair_events += 2
                continue

            if trade.entry_side == "primary":
                unmatched_primary.append(trade)
            else:
                unmatched_secondary.append(trade)

        for trade in list(unmatched_primary) + list(unmatched_secondary):
            opposite_proxy = clamp((1.0 - trade.price) + safe_proxy_buffer, 0.01, 0.99)
            pair_cost = trade.price + opposite_proxy
            learner.observe(
                market_id=trade.market_id,
                timeframe=trade.timeframe,
                side=trade.entry_side,
                seconds_to_end=trade.seconds_to_end,
                pair_price_cost=pair_cost,
                hedge_delay_seconds=safe_proxy_delay,
                success=pair_cost <= success_threshold,
                source="seed_history_proxy",
            )
            result.seeded_proxy_events += 1

    return result

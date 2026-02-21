from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from polymarket_bot.http_utils import get_json
from polymarket_bot.models import MarketInfo, Timeframe, parse_float, parse_token_ids, parse_ts


def _safe_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _choose_primary_secondary(outcomes: list[str], token_ids: list[str]) -> tuple[str, str, str, str]:
    if not token_ids:
        return "", "", "primary", "secondary"
    if len(token_ids) == 1:
        return token_ids[0], token_ids[0], "primary", "secondary"

    pairs = list(zip(outcomes, token_ids))
    if not pairs:
        return token_ids[0], token_ids[1], "primary", "secondary"

    def score(label: str) -> int:
        normalized = label.lower()
        if "up" in normalized or "yes" in normalized:
            return 3
        if "down" in normalized or "no" in normalized:
            return 1
        return 2

    pairs_sorted = sorted(pairs, key=lambda x: score(x[0]), reverse=True)
    primary_label, primary_token = pairs_sorted[0]
    secondary_label, secondary_token = pairs_sorted[-1]
    return primary_token, secondary_token, primary_label, secondary_label


def _window_seconds(timeframe: Timeframe) -> int:
    if timeframe == Timeframe.FIVE_MIN:
        return 300
    if timeframe == Timeframe.FIFTEEN_MIN:
        return 900
    if timeframe == Timeframe.ONE_HOUR:
        return 3600
    return 0


@dataclass
class GammaClient:
    base_url: str
    timeout_seconds: float = 10.0

    def fetch_markets(self, tag_id: int, limit: int = 2000) -> list[dict]:
        params = {
            "tag_id": str(tag_id),
            "active": "true",
            "closed": "false",
            "limit": str(limit),
        }
        payload = get_json(f"{self.base_url}/markets", params=params, timeout=self.timeout_seconds)
        if not isinstance(payload, list):
            return []
        return payload

    def fetch_btc_markets(self, tags: Iterable[int], limit_per_tag: int = 2000) -> list[MarketInfo]:
        markets: list[MarketInfo] = []
        now = datetime.now(tz=timezone.utc)
        for tag in tags:
            timeframe = Timeframe.from_tag(tag)
            for item in self.fetch_markets(tag_id=tag, limit=limit_per_tag):
                question = str(item.get("question") or item.get("title") or "")
                if "bitcoin" not in question.lower():
                    continue
                raw_start_time = parse_ts(item.get("startDate"))
                end_time = parse_ts(item.get("endDate"))

                # Gamma "startDate" can represent listing/open time, not the
                # contract window start. Derive the effective start from end-time
                # and tenor so pre-start entries are impossible.
                tenor_seconds = _window_seconds(timeframe)
                if tenor_seconds > 0:
                    derived_start_time = end_time - timedelta(seconds=tenor_seconds)
                    start_time = max(raw_start_time, derived_start_time)
                else:
                    start_time = raw_start_time

                if end_time <= start_time:
                    continue
                if start_time > now:
                    continue
                if end_time <= now:
                    continue

                token_ids = parse_token_ids(item.get("clobTokenIds"))
                outcomes = _safe_list(item.get("outcomes"))
                primary_token, secondary_token, primary_label, secondary_label = _choose_primary_secondary(
                    outcomes, token_ids
                )
                markets.append(
                    MarketInfo(
                        market_id=str(item.get("id", "")),
                        slug=str(item.get("slug") or item.get("marketSlug") or ""),
                        question=question,
                        condition_id=str(item.get("conditionId") or ""),
                        tag_id=tag,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time,
                        liquidity_num=parse_float(item.get("liquidityNum")),
                        volume_24h=parse_float(item.get("volume24hr")),
                        best_bid=parse_float(item.get("bestBid")),
                        best_ask=parse_float(item.get("bestAsk")),
                        spread=parse_float(item.get("spread")),
                        outcomes=outcomes,
                        token_ids=token_ids,
                        primary_token_id=primary_token,
                        secondary_token_id=secondary_token,
                        primary_label=primary_label,
                        secondary_label=secondary_label,
                        is_neg_risk=_parse_bool(item.get("negRisk")),
                        order_min_size=parse_float(item.get("orderMinSize"), 5.0),
                    )
                )
        markets.sort(key=lambda m: m.end_time)
        return markets

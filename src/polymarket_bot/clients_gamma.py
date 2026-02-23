from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, TypedDict, cast

from polymarket_bot.http_utils import get_json
from polymarket_bot.models import MarketInfo, Timeframe, parse_float, parse_token_ids, parse_ts


class GammaMarketPayload(TypedDict, total=False):
    id: str | int
    slug: str
    question: str
    conditionId: str
    startDate: str
    endDate: str
    clobTokenIds: list[str] | str
    outcomes: list[str]
    liquidityNum: object
    volume24hr: object
    bestBid: object
    bestAsk: object
    spread: object
    negRisk: object
    orderMinSize: object


def _safe_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


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

    def fetch_markets(self, tag_id: int, limit: int = 2000) -> list[GammaMarketPayload]:
        params = {
            "tag_id": str(tag_id),
            "active": "true",
            "closed": "false",
            "limit": str(limit),
        }
        payload = get_json(f"{self.base_url}/markets", params=params, timeout=self.timeout_seconds)
        if not isinstance(payload, list):
            raise RuntimeError("Gamma /markets response must be a JSON array")
        out: list[GammaMarketPayload] = []
        for item in payload:
            if isinstance(item, dict):
                out.append(cast(GammaMarketPayload, item))
        return out

    def fetch_btc_markets(self, tags: Iterable[int], limit_per_tag: int = 2000) -> list[MarketInfo]:
        markets: list[MarketInfo] = []
        now = datetime.now(tz=timezone.utc)
        for tag in tags:
            timeframe = Timeframe.from_tag(tag)
            for item in self.fetch_markets(tag_id=tag, limit=limit_per_tag):
                question_raw = item.get("question")
                if not isinstance(question_raw, str):
                    continue
                question = question_raw.strip()
                if not question or "bitcoin" not in question.lower():
                    continue

                market_id_raw = item.get("id")
                market_id = str(market_id_raw).strip() if market_id_raw is not None else ""
                if not market_id:
                    continue

                condition_id_raw = item.get("conditionId")
                condition_id = (
                    condition_id_raw.strip() if isinstance(condition_id_raw, str) else ""
                )
                if not condition_id:
                    continue

                start_raw = item.get("startDate")
                end_raw = item.get("endDate")
                if not isinstance(start_raw, str) or not isinstance(end_raw, str):
                    continue
                if not start_raw.strip() or not end_raw.strip():
                    continue
                try:
                    raw_start_time = parse_ts(start_raw)
                    end_time = parse_ts(end_raw)
                except Exception:
                    continue

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
                if len(token_ids) < 2:
                    continue
                outcomes = _safe_list(item.get("outcomes"))
                primary_token, secondary_token, primary_label, secondary_label = _choose_primary_secondary(
                    outcomes, token_ids
                )
                slug_raw = item.get("slug")
                slug = slug_raw.strip() if isinstance(slug_raw, str) else ""
                markets.append(
                    MarketInfo(
                        market_id=market_id,
                        slug=slug,
                        question=question,
                        condition_id=condition_id,
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

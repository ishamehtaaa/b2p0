from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from polymarket_bot.models import Timeframe
from polymarket_bot.pricing import clamp
from polymarket_bot.storage import Storage


@dataclass
class PairBucketStat:
    timeframe: str
    entry_side: str
    sec_bucket: int
    samples: int
    successes: int
    ewma_pair_price: float
    ewma_hedge_delay: float

    @property
    def success_rate(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.successes / self.samples


class PairTimingLearner:
    """
    Learns side/timing buckets for alpha-first pair entries:
      - success: completed pair price (entry + hedge) under $1
      - failure: stale unhedged alpha entries that drift above $1
    """

    def __init__(self, storage: Storage, ewma_alpha: float = 0.22) -> None:
        self.storage = storage
        self.ewma_alpha = clamp(float(ewma_alpha), 0.05, 0.75)
        self._stats: dict[tuple[str, str, int], PairBucketStat] = {}
        self._load_existing()

    @staticmethod
    def _bucket_step(timeframe: Timeframe) -> int:
        if timeframe == Timeframe.FIVE_MIN:
            return 20
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 45
        if timeframe == Timeframe.ONE_HOUR:
            return 180
        return 60

    def bucket_for(self, timeframe: Timeframe, seconds_to_end: float) -> int:
        step = max(1, self._bucket_step(timeframe))
        sec = int(max(0.0, seconds_to_end))
        return int((sec // step) * step)

    @staticmethod
    def _normalize_side(side: str) -> str:
        lowered = str(side or "").strip().lower()
        if lowered in {"primary", "secondary"}:
            return lowered
        return ""

    @staticmethod
    def _normalize_timeframe(value: str | Timeframe) -> Timeframe:
        if isinstance(value, Timeframe):
            return value
        raw = str(value or "").strip().lower()
        for timeframe in (Timeframe.FIVE_MIN, Timeframe.FIFTEEN_MIN, Timeframe.ONE_HOUR):
            if raw == timeframe.value:
                return timeframe
        return Timeframe.UNKNOWN

    def _load_existing(self) -> None:
        for row in self.storage.load_pair_learning_stats():
            timeframe = str(row.get("timeframe") or "")
            entry_side = self._normalize_side(str(row.get("entry_side") or ""))
            if not timeframe or not entry_side:
                continue
            sec_bucket = int(row.get("sec_bucket") or 0)
            stat = PairBucketStat(
                timeframe=timeframe,
                entry_side=entry_side,
                sec_bucket=sec_bucket,
                samples=int(row.get("samples") or 0),
                successes=int(row.get("successes") or 0),
                ewma_pair_price=float(row.get("ewma_pair_price") or 1.02),
                ewma_hedge_delay=float(row.get("ewma_hedge_delay") or 60.0),
            )
            self._stats[(timeframe, entry_side, sec_bucket)] = stat

    def _matching(self, *, timeframe: Timeframe, side: str, sec_bucket: int) -> Iterable[PairBucketStat]:
        tf = timeframe.value
        exact = self._stats.get((tf, side, sec_bucket))
        if exact is not None:
            yield exact

        # Nearby buckets are still informative for short-horizon markets.
        step = self._bucket_step(timeframe)
        for near in (sec_bucket - step, sec_bucket + step):
            neighbor = self._stats.get((tf, side, near))
            if neighbor is not None:
                yield neighbor

    def estimate(
        self,
        *,
        timeframe: Timeframe,
        side: str,
        seconds_to_end: float,
    ) -> tuple[float | None, float | None, int]:
        normalized_side = self._normalize_side(side)
        if timeframe == Timeframe.UNKNOWN or not normalized_side:
            return None, None, 0
        bucket = self.bucket_for(timeframe, seconds_to_end)
        candidates = list(self._matching(timeframe=timeframe, side=normalized_side, sec_bucket=bucket))
        if not candidates:
            return self._aggregate_side(timeframe=timeframe, side=normalized_side)

        total_weight = float(sum(max(1, item.samples) for item in candidates))
        if total_weight <= 0:
            return None, None, 0
        exp_pair = sum(item.ewma_pair_price * max(1, item.samples) for item in candidates) / total_weight
        exp_success = sum(item.success_rate * max(1, item.samples) for item in candidates) / total_weight
        exp_samples = int(sum(item.samples for item in candidates))
        return exp_pair, exp_success, exp_samples

    def _aggregate_side(self, *, timeframe: Timeframe, side: str) -> tuple[float | None, float | None, int]:
        stats = [
            item
            for (tf, sd, _bucket), item in self._stats.items()
            if tf == timeframe.value and sd == side and item.samples > 0
        ]
        if not stats:
            return None, None, 0
        total_weight = float(sum(item.samples for item in stats))
        exp_pair = sum(item.ewma_pair_price * item.samples for item in stats) / total_weight
        exp_success = sum(item.success_rate * item.samples for item in stats) / total_weight
        exp_samples = int(sum(item.samples for item in stats))
        return exp_pair, exp_success, exp_samples

    def observe(
        self,
        *,
        market_id: str,
        timeframe: Timeframe,
        side: str,
        seconds_to_end: float,
        pair_price_cost: float,
        hedge_delay_seconds: float,
        success: bool,
        source: str,
    ) -> None:
        normalized_side = self._normalize_side(side)
        if timeframe == Timeframe.UNKNOWN or not normalized_side:
            return
        bucket = self.bucket_for(timeframe, seconds_to_end)
        key = (timeframe.value, normalized_side, bucket)

        clamped_pair = clamp(float(pair_price_cost), 0.01, 1.99)
        clamped_delay = clamp(float(hedge_delay_seconds), 0.0, 3600.0)
        prior = self._stats.get(key)
        if prior is None:
            updated = PairBucketStat(
                timeframe=timeframe.value,
                entry_side=normalized_side,
                sec_bucket=bucket,
                samples=1,
                successes=1 if success else 0,
                ewma_pair_price=clamped_pair,
                ewma_hedge_delay=clamped_delay,
            )
        else:
            alpha = self.ewma_alpha
            updated = PairBucketStat(
                timeframe=timeframe.value,
                entry_side=normalized_side,
                sec_bucket=bucket,
                samples=prior.samples + 1,
                successes=prior.successes + (1 if success else 0),
                ewma_pair_price=((1.0 - alpha) * prior.ewma_pair_price) + (alpha * clamped_pair),
                ewma_hedge_delay=((1.0 - alpha) * prior.ewma_hedge_delay) + (alpha * clamped_delay),
            )

        self._stats[key] = updated
        self.storage.record_pair_learning_event(
            market_id=market_id,
            timeframe=timeframe.value,
            entry_side=normalized_side,
            sec_bucket=bucket,
            pair_price_cost=clamped_pair,
            hedge_delay_seconds=clamped_delay,
            success=success,
            source=source,
        )
        self.storage.upsert_pair_learning_stat(
            timeframe=updated.timeframe,
            entry_side=updated.entry_side,
            sec_bucket=updated.sec_bucket,
            samples=updated.samples,
            successes=updated.successes,
            ewma_pair_price=updated.ewma_pair_price,
            ewma_hedge_delay=updated.ewma_hedge_delay,
        )

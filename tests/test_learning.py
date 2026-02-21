from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.learning import PairTimingLearner
from polymarket_bot.models import Timeframe


class _DummyStorage:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.stats: list[dict[str, object]] = []

    def load_pair_learning_stats(self):
        return self.stats

    def record_pair_learning_event(self, **kwargs):
        self.events.append(kwargs)

    def upsert_pair_learning_stat(self, **kwargs):
        for i, stat in enumerate(self.stats):
            if (
                stat.get("timeframe") == kwargs.get("timeframe")
                and stat.get("entry_side") == kwargs.get("entry_side")
                and int(stat.get("sec_bucket", 0)) == int(kwargs.get("sec_bucket", 0))
            ):
                self.stats[i] = dict(kwargs)
                return
        self.stats.append(dict(kwargs))


class PairTimingLearnerTests(unittest.TestCase):
    def test_observe_and_estimate_bucket(self) -> None:
        storage = _DummyStorage()
        learner = PairTimingLearner(storage)
        for idx in range(12):
            learner.observe(
                market_id=f"m-{idx}",
                timeframe=Timeframe.FIVE_MIN,
                side="primary",
                seconds_to_end=210,
                pair_price_cost=0.992 + (0.001 * (idx % 2)),
                hedge_delay_seconds=8.0 + idx,
                success=True,
                source="bot_fill",
            )
        cost, success_rate, samples = learner.estimate(
            timeframe=Timeframe.FIVE_MIN,
            side="primary",
            seconds_to_end=205,
        )
        self.assertIsNotNone(cost)
        self.assertIsNotNone(success_rate)
        self.assertGreaterEqual(samples, 12)
        self.assertLess(float(cost), 1.0)
        self.assertGreater(float(success_rate), 0.80)

    def test_estimate_falls_back_to_timeframe_side_aggregate(self) -> None:
        storage = _DummyStorage()
        learner = PairTimingLearner(storage)
        learner.observe(
            market_id="m-a",
            timeframe=Timeframe.FIFTEEN_MIN,
            side="secondary",
            seconds_to_end=800,
            pair_price_cost=0.998,
            hedge_delay_seconds=20.0,
            success=True,
            source="bot_fill",
        )
        learner.observe(
            market_id="m-b",
            timeframe=Timeframe.FIFTEEN_MIN,
            side="secondary",
            seconds_to_end=600,
            pair_price_cost=1.004,
            hedge_delay_seconds=24.0,
            success=False,
            source="stale_timeout",
        )
        cost, success_rate, samples = learner.estimate(
            timeframe=Timeframe.FIFTEEN_MIN,
            side="secondary",
            seconds_to_end=120,  # No exact bucket stats yet.
        )
        self.assertIsNotNone(cost)
        self.assertIsNotNone(success_rate)
        self.assertGreaterEqual(samples, 2)
        self.assertGreater(float(cost), 0.99)
        self.assertLess(float(cost), 1.01)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.main import SpotTracker


class SpotTrackerTests(unittest.TestCase):
    def test_rnjd_probability_skews_up_on_positive_path(self) -> None:
        tracker = SpotTracker()
        price = 100_000.0
        ts = 1_000.0
        for i in range(260):
            ts += 0.2
            price *= 1.0 + 0.00015
            if i % 40 == 0:
                price *= 1.0008
            tracker.update(ts, price)
        p = tracker.rnjd_probability(300)
        self.assertGreater(p, 0.5)

    def test_rnjd_probability_neutral_on_chop(self) -> None:
        tracker = SpotTracker()
        base = 100_000.0
        ts = 2_000.0
        for i in range(260):
            ts += 0.2
            wiggle = 0.00015 if i % 2 == 0 else -0.00015
            tracker.update(ts, base * (1.0 + wiggle))
        p = tracker.rnjd_probability(300)
        self.assertGreater(p, 0.35)
        self.assertLess(p, 0.65)


if __name__ == "__main__":
    unittest.main()

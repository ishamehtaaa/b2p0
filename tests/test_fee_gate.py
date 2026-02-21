from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.helpers import build_snapshot, test_config

from polymarket_bot.engines.engine_1h_maker import HourMakerEngine
from polymarket_bot.models import Timeframe


class FeeGateTests(unittest.TestCase):
    def test_maker_enabled_when_fee_is_zero(self) -> None:
        cfg = test_config(fee_disable_maker_bps=800)
        snapshot = build_snapshot(timeframe=Timeframe.ONE_HOUR, primary_fee_bps=0)
        engine = HourMakerEngine(cfg)
        intents = engine.generate(snapshot, inventory_shares=0.0, fair_probability=0.5)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].side.value, "buy")

    def test_maker_posts_ask_when_inventory_exists(self) -> None:
        cfg = test_config(fee_disable_maker_bps=800)
        snapshot = build_snapshot(timeframe=Timeframe.ONE_HOUR, primary_fee_bps=0)
        engine = HourMakerEngine(cfg)
        intents = engine.generate(snapshot, inventory_shares=100.0, fair_probability=0.5)
        self.assertEqual(len(intents), 2)

    def test_maker_disabled_when_fee_is_high(self) -> None:
        cfg = test_config(fee_disable_maker_bps=800)
        snapshot = build_snapshot(timeframe=Timeframe.ONE_HOUR, primary_fee_bps=1000)
        engine = HourMakerEngine(cfg)
        intents = engine.generate(snapshot, inventory_shares=0.0, fair_probability=0.5)
        self.assertEqual(len(intents), 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.engines.engine_short_dir import ShortDirectionalEngine
from polymarket_bot.pricing import directional_ev_per_share
from polymarket_bot.models import TimeInForce, Timeframe
from tests.helpers import build_snapshot, test_config


class EVTests(unittest.TestCase):
    def test_directional_ev_negative_with_high_fee(self) -> None:
        ev = directional_ev_per_share(
            fair_probability=0.52,
            entry_price=0.50,
            base_fee_bps=1000,
            slippage_buffer=0.002,
        )
        self.assertLessEqual(ev, 0)

    def test_directional_engine_rejects_when_threshold_or_ev_not_met(self) -> None:
        cfg = test_config(
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=1000,
            secondary_fee_bps=1000,
            primary_bid=0.49,
            primary_ask=0.50,
            secondary_bid=0.49,
            secondary_ask=0.50,
        )
        intents = engine.generate(snapshot, fair_probability=0.505)
        self.assertEqual(intents, [])

    def test_directional_engine_uses_ioc_for_clear_edge(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
            max_market_exposure_pct=0.10,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=1000,
            secondary_fee_bps=1000,
            primary_bid=0.40,
            primary_ask=0.42,
            secondary_bid=0.56,
            secondary_ask=0.58,
        )
        intents = engine.generate(snapshot, fair_probability=0.60)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].tif, TimeInForce.IOC)
        self.assertFalse(intents[0].post_only)

    def test_directional_engine_rebalance_intent_when_inventory_skewed(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
            max_market_exposure_pct=0.10,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=0,
            secondary_fee_bps=0,
            primary_bid=0.48,
            primary_ask=0.50,
            secondary_bid=0.49,
            secondary_ask=0.51,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.484,
            primary_inventory=30.0,
            secondary_inventory=0.0,
        )
        self.assertTrue(intents)
        rebalance = [
            i
            for i in intents
            if i.metadata.get("intent_type") in {"rebalance", "rebalance_forced"}
        ]
        self.assertTrue(rebalance)
        self.assertEqual(rebalance[0].token_id, snapshot.market.secondary_token_id)
        self.assertEqual(rebalance[0].side.value, "buy")

    def test_directional_engine_emits_take_profit_exit(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
            max_market_exposure_pct=0.10,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=0,
            secondary_fee_bps=0,
            primary_bid=0.60,
            primary_ask=0.62,
            secondary_bid=0.35,
            secondary_ask=0.70,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.55,
            primary_inventory=20.0,
            primary_avg_entry=0.50,
        )
        exits = [i for i in intents if i.side.value == "sell" and i.metadata.get("intent_type") == "take_profit_exit"]
        self.assertTrue(exits)
        self.assertEqual(exits[0].token_id, snapshot.market.primary_token_id)

    def test_directional_engine_dual_alpha_when_both_sides_positive(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
            max_market_exposure_pct=0.20,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=0,
            secondary_fee_bps=0,
            primary_bid=0.48,
            primary_ask=0.49,
            secondary_bid=0.42,
            secondary_ask=0.43,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.53,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        dual = [i for i in intents if i.metadata.get("intent_type") == "dual_alpha"]
        self.assertEqual(len(dual), 2)
        self.assertTrue(all(i.side.value == "buy" for i in dual))

    def test_directional_engine_forced_rebalance_priority(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_threshold=0.035,
            directional_slippage_buffer=0.002,
            directional_notional=20.0,
            max_market_exposure_pct=0.20,
        )
        engine = ShortDirectionalEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_fee_bps=0,
            secondary_fee_bps=0,
            primary_bid=0.50,
            primary_ask=0.51,
            secondary_bid=0.48,
            secondary_ask=0.49,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.40,
            primary_inventory=80.0,
            secondary_inventory=0.0,
        )
        forced = [i for i in intents if i.metadata.get("intent_type") == "rebalance_forced"]
        self.assertTrue(forced)
        self.assertEqual(forced[0].token_id, snapshot.market.secondary_token_id)


if __name__ == "__main__":
    unittest.main()

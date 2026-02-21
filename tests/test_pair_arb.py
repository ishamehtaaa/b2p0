from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.engines.engine_pair_arb import PairArbEngine
from polymarket_bot.models import Timeframe
from tests.helpers import build_snapshot, test_config


class PairArbEngineTests(unittest.TestCase):
    def test_generates_alpha_entry_on_underpriced_side(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.40,
            primary_ask=0.42,
            secondary_bid=0.52,
            secondary_ask=0.54,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.60,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertEqual(len(intents), 2)
        self.assertTrue(all(i.side.value == "buy" for i in intents))
        self.assertEqual(
            {i.token_id for i in intents},
            {snapshot.market.primary_token_id, snapshot.market.secondary_token_id},
        )
        self.assertEqual({i.engine for i in intents}, {"engine_pair_arb"})
        self.assertIn("pair_entry_primary", {str(i.metadata.get("intent_type")) for i in intents})
        self.assertIn("pair_completion", {str(i.metadata.get("intent_type")) for i in intents})

    def test_skips_alpha_entry_when_no_edge(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.52,
            primary_ask=0.53,
            secondary_bid=0.52,
            secondary_ask=0.53,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertEqual(intents, [])

    def test_learned_timing_can_lower_edge_gate(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.48,
            primary_ask=0.49,
            secondary_bid=0.49,
            secondary_ask=0.50,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.553,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            learned_primary_pair_price=0.995,
            learned_primary_success_rate=0.62,
            learned_primary_samples=20,
            learned_secondary_pair_price=1.02,
            learned_secondary_success_rate=0.40,
            learned_secondary_samples=20,
        )
        self.assertEqual(len(intents), 2)
        self.assertIn("pair_entry_primary", {str(i.metadata.get("intent_type")) for i in intents})
        self.assertIn("pair_completion", {str(i.metadata.get("intent_type")) for i in intents})

    def test_skips_alpha_entry_when_pair_completion_is_not_profitable(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.51,
            primary_ask=0.52,
            secondary_bid=0.51,
            secondary_ask=0.52,
            primary_fee_bps=1000,
            secondary_fee_bps=1000,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.85,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertEqual(intents, [])

    def test_allows_alpha_entry_when_projected_completion_cost_is_profitable(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.48,
            primary_ask=0.49,
            secondary_bid=0.46,
            secondary_ask=0.54,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.80,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(str(intents[0].metadata.get("intent_type")), "alpha_entry")

    def test_skips_alpha_entry_when_pair_cost_is_too_high_even_with_projection(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.72,
            primary_ask=0.75,
            secondary_bid=0.70,
            secondary_ask=0.74,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.99,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertEqual(intents, [])

    def test_soft_equalizer_uses_ladder_above_target_pair_cost(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.10,
            primary_ask=0.11,
            secondary_bid=0.90,
            secondary_ask=0.91,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.30,
            primary_inventory=80.0,
            secondary_inventory=20.0,
            primary_avg_entry=0.10,
            secondary_avg_entry=0.90,
            now=datetime.now(tz=timezone.utc),
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].metadata.get("intent_type"), "equalize")
        self.assertGreaterEqual(intents[0].size, snapshot.market.order_min_size)

    def test_unbalanced_inventory_prioritizes_equalizer_over_new_alpha(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.44,
            primary_ask=0.45,
            secondary_bid=0.54,
            secondary_ask=0.55,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.60,
            primary_inventory=68.0,
            secondary_inventory=32.0,
            primary_avg_entry=0.45,
            secondary_avg_entry=0.55,
            now=datetime.now(tz=timezone.utc),
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].metadata.get("intent_type"), "equalize")

    def test_skips_prestart_market(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.44,
            primary_ask=0.45,
            secondary_bid=0.54,
            secondary_ask=0.55,
        )
        now = datetime.now(tz=timezone.utc)
        snapshot.market.start_time = now + timedelta(minutes=4)
        snapshot.market.end_time = now + timedelta(minutes=9)
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.60,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            now=now,
        )
        self.assertEqual(intents, [])

    def test_skips_alpha_entry_when_too_close_to_end(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.01,
            primary_ask=0.03,
            secondary_bid=0.97,
            secondary_ask=0.99,
        )
        now = datetime.now(tz=timezone.utc)
        snapshot.market.start_time = now - timedelta(minutes=3)
        snapshot.market.end_time = now + timedelta(seconds=90)
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.55,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            now=now,
        )
        self.assertEqual(intents, [])

    def test_forced_equalizer_when_naked_ratio_is_high(self) -> None:
        cfg = test_config(
            bankroll_usdc=200.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.44,
            primary_ask=0.45,
            secondary_bid=0.54,
            secondary_ask=0.55,
        )
        now = datetime.now(tz=timezone.utc)
        snapshot.market.start_time = now - timedelta(minutes=4)
        snapshot.market.end_time = now + timedelta(seconds=35)
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=130.0,
            secondary_inventory=10.0,
            primary_avg_entry=0.46,
            secondary_avg_entry=0.54,
            now=now,
        )
        self.assertEqual(len(intents), 1)
        intent = intents[0]
        self.assertEqual(intent.token_id, snapshot.market.secondary_token_id)
        self.assertEqual(intent.side.value, "buy")
        self.assertEqual(intent.metadata.get("intent_type"), "equalize_forced")
        self.assertEqual(intent.engine, "engine_pair_arb")

    def test_alpha_entry_notional_stays_within_market_cap_on_low_price(self) -> None:
        cfg = test_config(
            bankroll_usdc=20.0,
            directional_notional=25.0,
            max_market_exposure_pct=0.60,  # cap = 12
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.ONE_HOUR,
            primary_bid=0.03,
            primary_ask=0.04,
            secondary_bid=0.91,
            secondary_ask=0.93,
        )
        now = datetime.now(tz=timezone.utc)
        snapshot.market.start_time = now - timedelta(minutes=2)
        snapshot.market.end_time = now + timedelta(minutes=20)
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.30,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            now=now,
        )
        self.assertEqual(len(intents), 2)
        self.assertLessEqual(sum(i.notional for i in intents), 6.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()

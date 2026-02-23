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
        self.assertTrue(all(i.post_only for i in intents))
        self.assertTrue(all(i.tif.value == "GTC" for i in intents))
        by_token = {i.token_id: i for i in intents}
        self.assertAlmostEqual(by_token[snapshot.market.primary_token_id].price, 0.41, places=2)
        self.assertAlmostEqual(by_token[snapshot.market.secondary_token_id].price, 0.53, places=2)

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

    def test_prefers_resting_pair_when_maker_cost_is_profitable(self) -> None:
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
        self.assertEqual(len(intents), 2)
        self.assertIn("pair_entry_primary", {str(i.metadata.get("intent_type")) for i in intents})
        self.assertIn("pair_completion", {str(i.metadata.get("intent_type")) for i in intents})
        self.assertTrue(all(i.post_only for i in intents))
        self.assertTrue(all(i.tif.value == "GTC" for i in intents))

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

    def test_soft_equalizer_keeps_pair_ladder_active_above_target_pair_cost(self) -> None:
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
        intent_types = {str(i.metadata.get("intent_type")) for i in intents}
        self.assertIn("equalize", intent_types)
        self.assertNotIn("alpha_entry", intent_types)
        self.assertIn("pair_entry_primary", intent_types)
        self.assertIn("pair_completion", intent_types)
        equalizers = [i for i in intents if i.metadata.get("intent_type") == "equalize"]
        self.assertEqual(len(equalizers), 1)
        self.assertGreaterEqual(equalizers[0].size, snapshot.market.order_min_size)
        self.assertTrue(equalizers[0].post_only)
        self.assertEqual(equalizers[0].tif.value, "GTC")

    def test_unbalanced_inventory_keeps_equalizer_and_disables_alpha(self) -> None:
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
        intent_types = {str(i.metadata.get("intent_type")) for i in intents}
        self.assertIn("equalize", intent_types)
        self.assertNotIn("alpha_entry", intent_types)
        self.assertIn("pair_entry_primary", intent_types)
        self.assertIn("pair_completion", intent_types)

    def test_deep_mode_hard_skew_forces_rebalance_only(self) -> None:
        cfg = test_config(
            bankroll_usdc=300.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.80,
            directional_slippage_buffer=0.002,
            single_5m_deep_mode=True,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.54,
            primary_ask=0.55,
            secondary_bid=0.44,
            secondary_ask=0.45,
        )
        now = datetime.now(tz=timezone.utc)
        snapshot.market.start_time = now - timedelta(minutes=2)
        snapshot.market.end_time = now + timedelta(seconds=170)
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.58,
            primary_inventory=165.0,
            secondary_inventory=124.0,
            primary_avg_entry=0.54,
            secondary_avg_entry=0.44,
            now=now,
        )
        self.assertEqual(len(intents), 1)
        intent = intents[0]
        self.assertEqual(intent.metadata.get("intent_type"), "equalize_forced")
        self.assertEqual(intent.token_id, snapshot.market.secondary_token_id)
        self.assertFalse(intent.post_only)
        self.assertEqual(intent.tif.value, "IOC")

    def test_deep_mode_soft_skew_keeps_ladder_plus_equalizer(self) -> None:
        cfg = test_config(
            bankroll_usdc=300.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.80,
            directional_slippage_buffer=0.002,
            single_5m_deep_mode=True,
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
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.56,
            primary_inventory=114.0,
            secondary_inventory=100.0,
            primary_avg_entry=0.45,
            secondary_avg_entry=0.55,
            now=now,
        )
        intent_types = {str(i.metadata.get("intent_type")) for i in intents}
        self.assertIn("equalize", intent_types)
        self.assertIn("pair_entry_primary", intent_types)
        self.assertIn("pair_completion", intent_types)
        self.assertNotIn("alpha_entry", intent_types)

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
        snapshot.market.end_time = now + timedelta(seconds=60)
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
        self.assertFalse(intent.post_only)
        self.assertEqual(intent.tif.value, "IOC")

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
        self.assertTrue(all(i.post_only for i in intents))
        self.assertTrue(all(i.tif.value == "GTC" for i in intents))

    def test_wide_spread_pair_entry_emits_multiple_maker_levels(self) -> None:
        cfg = test_config(
            bankroll_usdc=120.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.20,
            primary_ask=0.24,
            secondary_bid=0.76,
            secondary_ask=0.80,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertGreaterEqual(len(intents), 4)
        self.assertTrue(all(i.post_only for i in intents))
        primary_intents = [i for i in intents if i.token_id == snapshot.market.primary_token_id]
        secondary_intents = [i for i in intents if i.token_id == snapshot.market.secondary_token_id]
        self.assertGreaterEqual(len(primary_intents), 2)
        self.assertGreaterEqual(len(secondary_intents), 2)
        self.assertGreater(len({i.price for i in primary_intents}), 1)
        self.assertGreater(len({i.price for i in secondary_intents}), 1)
        self.assertTrue(all(str(i.metadata.get("quote_level_id", "")).startswith("pair_") for i in intents))
        self.assertTrue(all(str(i.metadata.get("pair_group_id", "")).startswith(snapshot.market.market_id) for i in intents))

    def test_motion_swing_expands_5m_pair_ladder_depth(self) -> None:
        cfg = test_config(
            bankroll_usdc=120.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.20,
            primary_ask=0.22,
            secondary_bid=0.78,
            secondary_ask=0.80,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            primary_motion={"ask_low": 0.16, "ask_high": 0.22, "ask_swing": 0.06, "samples": 14.0},
            secondary_motion={"ask_low": 0.74, "ask_high": 0.80, "ask_swing": 0.06, "samples": 14.0},
        )
        self.assertGreaterEqual(len(intents), 6)
        self.assertTrue(all(i.post_only for i in intents))
        self.assertTrue(all(i.tif.value == "GTC" for i in intents))
        self.assertTrue(all(float(i.metadata.get("primary_ask_swing", 0.0)) >= 0.05 for i in intents))
        self.assertTrue(all(float(i.metadata.get("secondary_ask_swing", 0.0)) >= 0.05 for i in intents))

    def test_high_naked_ratio_keeps_paired_quotes_when_equalizer_unavailable(self) -> None:
        cfg = test_config(
            bankroll_usdc=100.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.60,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.05,
            primary_ask=0.06,
            secondary_bid=0.94,
            secondary_ask=0.95,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=80.0,
            secondary_inventory=20.0,
            primary_avg_entry=0.10,
            secondary_avg_entry=0.90,
            now=datetime.now(tz=timezone.utc),
        )
        # Even when equalization is expensive/unavailable, keep paired resting
        # quotes live so we continue capturing complete-set opportunities.
        intent_types = {str(i.metadata.get("intent_type")) for i in intents}
        self.assertIn("pair_entry_primary", intent_types)
        self.assertIn("pair_completion", intent_types)
        self.assertNotIn("alpha_entry", intent_types)

    def test_fast_fluctuation_mode_enables_queue_hold_metadata(self) -> None:
        cfg = test_config(
            bankroll_usdc=120.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.20,
            primary_ask=0.22,
            secondary_bid=0.76,
            secondary_ask=0.78,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            primary_motion={
                "ask_swing": 0.05,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.22,
                "mid_flip_rate": 0.18,
                "samples": 18.0,
            },
            secondary_motion={
                "ask_swing": 0.05,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.21,
                "mid_flip_rate": 0.17,
                "samples": 18.0,
            },
        )
        self.assertGreaterEqual(len(intents), 6)
        self.assertTrue(all(i.post_only for i in intents))
        self.assertTrue(all(bool(i.metadata.get("hold_queue")) for i in intents))
        self.assertTrue(all(float(i.metadata.get("min_quote_dwell_seconds", 0.0)) > 0.0 for i in intents))
        self.assertTrue(all(float(i.metadata.get("quote_refresh_seconds", 0.0)) > 1.6 for i in intents))

    def test_anticipatory_pair_index_plan_includes_cross_levels(self) -> None:
        plan = PairArbEngine._pair_ladder_index_plan(
            primary_levels=4,
            secondary_levels=4,
            anticipatory=True,
            extreme=False,
        )
        self.assertIn((0, 0), plan)
        self.assertIn((0, 1), plan)
        self.assertIn((1, 0), plan)
        self.assertTrue(any(abs(primary_idx - secondary_idx) == 1 for primary_idx, secondary_idx in plan))

    def test_fast_5m_anticipatory_mode_emits_cross_level_pair_quotes(self) -> None:
        cfg = test_config(
            bankroll_usdc=220.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.30,
            primary_ask=0.34,
            secondary_bid=0.66,
            secondary_ask=0.70,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            primary_motion={
                "ask_swing": 0.06,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.22,
                "mid_flip_rate": 0.18,
                "samples": 20.0,
            },
            secondary_motion={
                "ask_swing": 0.06,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.21,
                "mid_flip_rate": 0.17,
                "samples": 20.0,
            },
        )
        pair_primary = [i for i in intents if i.metadata.get("intent_type") == "pair_entry_primary"]
        self.assertGreaterEqual(len(pair_primary), 3)
        self.assertTrue(all(bool(i.metadata.get("anticipatory_mode")) for i in pair_primary))
        self.assertTrue(
            any(
                int(i.metadata.get("primary_ladder_index", 0))
                != int(i.metadata.get("secondary_ladder_index", 0))
                for i in pair_primary
            )
        )
        self.assertTrue(
            all(
                max(
                    float(i.metadata.get("primary_gap_ticks", 9.0)),
                    float(i.metadata.get("secondary_gap_ticks", 9.0)),
                )
                <= 2.0
                for i in pair_primary
            )
        )

    def test_fair_value_signal_sets_directional_bias_when_dislocation_is_large(self) -> None:
        cfg = test_config(
            bankroll_usdc=120.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.40,
            primary_ask=0.41,
            secondary_bid=0.58,
            secondary_ask=0.59,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        bullish = engine._fair_value_signal(
            snapshot=snapshot,
            fair_up=0.53,
            primary_motion=None,
            secondary_motion=None,
            fluctuation_regime={},
        )
        bearish = engine._fair_value_signal(
            snapshot=snapshot,
            fair_up=0.31,
            primary_motion=None,
            secondary_motion=None,
            fluctuation_regime={},
        )
        self.assertEqual(bullish.get("bias_side"), "primary")
        self.assertEqual(bearish.get("bias_side"), "secondary")
        self.assertGreater(float(bullish.get("dislocation", 0.0)), 0.0)
        self.assertLess(float(bearish.get("dislocation", 0.0)), 0.0)

    def test_deep_mode_quotes_hold_queue_longer(self) -> None:
        cfg = test_config(
            bankroll_usdc=300.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
            single_5m_deep_mode=True,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.24,
            primary_ask=0.26,
            secondary_bid=0.74,
            secondary_ask=0.76,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            primary_motion={
                "ask_swing": 0.05,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.20,
                "mid_flip_rate": 0.16,
                "samples": 16.0,
            },
            secondary_motion={
                "ask_swing": 0.05,
                "ask_swing_short": 0.03,
                "ask_flip_rate": 0.20,
                "mid_flip_rate": 0.16,
                "samples": 16.0,
            },
        )
        pair_intents = [i for i in intents if str(i.metadata.get("intent_type")) in {"pair_entry_primary", "pair_completion"}]
        self.assertGreaterEqual(len(pair_intents), 2)
        self.assertTrue(all(bool(i.metadata.get("hold_queue")) for i in pair_intents))
        self.assertTrue(all(float(i.metadata.get("min_quote_dwell_seconds", 0.0)) >= 2.0 for i in pair_intents))
        self.assertTrue(
            all(
                float(i.metadata.get("quote_max_age_seconds", 0.0))
                > float(i.metadata.get("quote_refresh_seconds", 0.0))
                for i in pair_intents
            )
        )

    def test_high_bankroll_fast_mode_uses_smaller_clips_and_faster_refresh(self) -> None:
        baseline_cfg = test_config(
            bankroll_usdc=120.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        fast_cfg = test_config(
            bankroll_usdc=300.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        baseline_engine = PairArbEngine(baseline_cfg)
        fast_engine = PairArbEngine(fast_cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.39,
            primary_ask=0.40,
            secondary_bid=0.59,
            secondary_ask=0.60,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        baseline_intents = baseline_engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        fast_intents = fast_engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
        )
        self.assertGreaterEqual(len(baseline_intents), 2)
        self.assertGreaterEqual(len(fast_intents), 2)
        baseline_primary = next(i for i in baseline_intents if i.metadata.get("intent_type") == "pair_entry_primary")
        fast_primary = next(i for i in fast_intents if i.metadata.get("intent_type") == "pair_entry_primary")
        self.assertLess(fast_primary.size, baseline_primary.size)
        self.assertLess(
            float(fast_primary.metadata.get("quote_refresh_seconds", 9.9)),
            float(baseline_primary.metadata.get("quote_refresh_seconds", 0.0)),
        )
        self.assertTrue(bool(fast_primary.metadata.get("fast_iteration_mode")))

    def test_pair_cost_governor_blocks_expensive_pairs_when_rolling_avg_is_high(self) -> None:
        cfg = test_config(
            bankroll_usdc=150.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.50,
            primary_ask=0.51,
            secondary_bid=0.51,
            secondary_ask=0.52,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            rolling_pair_cost_avg=1.020,
            rolling_pair_cost_samples=120,
        )
        self.assertEqual(intents, [])

    def test_pair_cost_governor_allows_opportunistic_overpay_when_rolling_avg_is_low(self) -> None:
        cfg = test_config(
            bankroll_usdc=150.0,
            directional_notional=20.0,
            max_market_exposure_pct=0.90,
            directional_slippage_buffer=0.002,
        )
        engine = PairArbEngine(cfg)
        snapshot = build_snapshot(
            timeframe=Timeframe.FIVE_MIN,
            primary_bid=0.50,
            primary_ask=0.51,
            secondary_bid=0.51,
            secondary_ask=0.52,
            primary_fee_bps=0,
            secondary_fee_bps=0,
        )
        intents = engine.generate(
            snapshot=snapshot,
            fair_probability=0.50,
            primary_inventory=0.0,
            secondary_inventory=0.0,
            rolling_pair_cost_avg=0.970,
            rolling_pair_cost_samples=120,
        )
        self.assertGreaterEqual(len(intents), 2)
        self.assertIn("pair_entry_primary", {str(i.metadata.get("intent_type")) for i in intents})
        self.assertIn("pair_completion", {str(i.metadata.get("intent_type")) for i in intents})


if __name__ == "__main__":
    unittest.main()

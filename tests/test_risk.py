from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.models import OrderIntent, OrderResult, Side, TimeInForce
from polymarket_bot.risk import RiskManager
from tests.helpers import test_config


class RiskTests(unittest.TestCase):
    def test_exposure_cap_no_longer_rejects_large_order(self) -> None:
        cfg = test_config(bankroll_usdc=1000.0, max_total_exposure_pct=0.20, max_market_exposure_pct=0.04)
        risk = RiskManager(cfg)
        intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.5,
            size=200.0,  # notional = 100, exceeds market cap (40)
            tif=TimeInForce.GTC,
            post_only=False,
            engine="test",
            expected_edge=0.01,
        )
        decision = risk.can_place(intent)
        self.assertTrue(decision.allowed)

    def test_daily_drawdown_no_longer_halts(self) -> None:
        cfg = test_config(bankroll_usdc=1000.0, max_daily_dd_pct=0.01)
        risk = RiskManager(cfg)
        fill = OrderResult(
            order_id="o1",
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.9,
            size=50.0,
            status="filled",
            filled_size=50.0,
            filled_price=0.9,
            fee_paid=0.0,
            engine="test",
            created_at=datetime.now(tz=timezone.utc),
            raw={},
        )
        risk.apply_fill(fill)
        risk.mark_to_market({"t1": 0.0})
        risk.check_kill_switches(datetime.now(tz=timezone.utc))
        self.assertFalse(risk.halted)

    def test_stale_feed_no_longer_halts(self) -> None:
        cfg = test_config(stale_feed_seconds=5.0)
        risk = RiskManager(cfg)
        now = datetime.now(tz=timezone.utc)
        risk.update_data_heartbeat(now)
        risk.check_kill_switches(now + timedelta(seconds=6))
        self.assertFalse(risk.halted)

    def test_sell_allowed_even_when_halted(self) -> None:
        cfg = test_config(bankroll_usdc=1000.0)
        risk = RiskManager(cfg)
        risk.halt("test")
        intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.SELL,
            price=0.5,
            size=10.0,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="test",
            expected_edge=0.0,
        )
        decision = risk.can_place(intent)
        self.assertTrue(decision.allowed)

    def test_buy_blocked_when_halted(self) -> None:
        cfg = test_config(bankroll_usdc=1000.0, max_daily_dd_pct=0.10)
        risk = RiskManager(cfg)
        risk.halt("fatal")
        intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.5,
            size=10.0,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="test",
            expected_edge=0.0,
        )
        decision = risk.can_place(intent)
        self.assertFalse(decision.allowed)

    def test_buy_blocked_when_cash_budget_exceeded(self) -> None:
        cfg = test_config(bankroll_usdc=10.0)
        risk = RiskManager(cfg)
        intent = OrderIntent(
            market_id="m1",
            token_id="t1",
            side=Side.BUY,
            price=0.70,
            size=20.0,  # notional 14 > cash 10
            tif=TimeInForce.IOC,
            post_only=False,
            engine="test",
            expected_edge=0.0,
            metadata={"fee_bps": 1000},
        )
        decision = risk.can_place(intent)
        self.assertFalse(decision.allowed)
        self.assertIn("cash budget exceeded", decision.reason)

    def test_opposite_outcomes_count_as_hedged_market_exposure(self) -> None:
        cfg = test_config(bankroll_usdc=100.0, max_total_exposure_pct=0.95, max_market_exposure_pct=0.60)
        risk = RiskManager(cfg)
        buy_up = OrderResult(
            order_id="o-up",
            market_id="m1",
            token_id="up",
            side=Side.BUY,
            price=0.60,
            size=50.0,
            status="filled",
            filled_size=50.0,
            filled_price=0.60,
            fee_paid=0.0,
            engine="test",
            created_at=datetime.now(tz=timezone.utc),
            raw={},
        )
        buy_down = OrderResult(
            order_id="o-down",
            market_id="m1",
            token_id="down",
            side=Side.BUY,
            price=0.40,
            size=50.0,
            status="filled",
            filled_size=50.0,
            filled_price=0.40,
            fee_paid=0.0,
            engine="test",
            created_at=datetime.now(tz=timezone.utc),
            raw={},
        )
        risk.apply_fill(buy_up)
        risk.apply_fill(buy_down)
        risk.mark_to_market({"up": 0.60, "down": 0.40})
        self.assertAlmostEqual(risk.market_exposure.get("m1", 0.0), 30.0, places=6)
        self.assertAlmostEqual(risk.total_exposure(), 30.0, places=6)

    def test_apply_merge_reduces_both_legs_and_adds_cash(self) -> None:
        cfg = test_config(bankroll_usdc=100.0)
        risk = RiskManager(cfg)
        risk.apply_fill(
            OrderResult(
                order_id="b1",
                market_id="m1",
                token_id="up",
                side=Side.BUY,
                price=0.45,
                size=20.0,
                status="filled",
                filled_size=20.0,
                filled_price=0.45,
                fee_paid=0.0,
                engine="test",
                created_at=datetime.now(tz=timezone.utc),
                raw={},
            )
        )
        risk.apply_fill(
            OrderResult(
                order_id="b2",
                market_id="m1",
                token_id="down",
                side=Side.BUY,
                price=0.50,
                size=15.0,
                status="filled",
                filled_size=15.0,
                filled_price=0.50,
                fee_paid=0.0,
                engine="test",
                created_at=datetime.now(tz=timezone.utc),
                raw={},
            )
        )
        cash_before = risk.cash
        merged = risk.apply_merge(
            market_id="m1",
            primary_token_id="up",
            secondary_token_id="down",
            pair_size=12.0,
            payout_per_pair=1.0,
        )
        self.assertAlmostEqual(merged, 12.0, places=6)
        self.assertAlmostEqual(risk.positions["up"].size, 8.0, places=6)
        self.assertAlmostEqual(risk.positions["down"].size, 3.0, places=6)
        self.assertAlmostEqual(risk.cash - cash_before, 12.0, places=6)

    def test_sync_exchange_inventory_overrides_sizes_and_cash(self) -> None:
        cfg = test_config(bankroll_usdc=100.0)
        risk = RiskManager(cfg)
        risk.apply_fill(
            OrderResult(
                order_id="o1",
                market_id="m1",
                token_id="up",
                side=Side.BUY,
                price=0.40,
                size=10.0,
                status="filled",
                filled_size=10.0,
                filled_price=0.40,
                fee_paid=0.0,
                engine="test",
                created_at=datetime.now(tz=timezone.utc),
                raw={},
            )
        )
        result = risk.sync_exchange_inventory(
            cash=81.25,
            token_sizes={"up": 35.0, "down": 35.0},
            token_market_ids={"up": "m1", "down": "m1"},
        )
        self.assertAlmostEqual(risk.cash, 81.25, places=6)
        self.assertAlmostEqual(risk.positions["up"].size, 35.0, places=6)
        self.assertAlmostEqual(risk.positions["down"].size, 35.0, places=6)
        self.assertAlmostEqual(result.cash_after, 81.25, places=6)
        self.assertIn("up", result.token_size_changes)
        self.assertIn("down", result.token_size_changes)
        self.assertEqual(result.unknown_market_tokens, ())

    def test_sync_exchange_inventory_tracks_unknown_market_tokens(self) -> None:
        cfg = test_config(bankroll_usdc=100.0)
        risk = RiskManager(cfg)
        result = risk.sync_exchange_inventory(
            cash=100.0,
            token_sizes={"orphan-token": 5.0},
            token_market_ids={},
        )
        self.assertEqual(result.unknown_market_tokens, ("orphan-token",))
        self.assertNotIn("orphan-token", risk.positions)


if __name__ == "__main__":
    unittest.main()

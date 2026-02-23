from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polymarket_bot.main import BotRuntime
from polymarket_bot.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderIntent,
    OrderResult,
    PositionState,
    Side,
    TimeInForce,
)
from polymarket_bot.risk import RiskManager
from polymarket_bot.runtime_state import QuoteOrderState
from tests.helpers import test_config


class _DummyStorage:
    def __init__(self) -> None:
        self.orders: list[tuple[OrderIntent, OrderResult, str]] = []
        self.risk_states: list[object] = []
        self.risk_events: list[tuple[str, dict]] = []

    def record_order(self, intent: OrderIntent, result: OrderResult, mode: str) -> None:
        self.orders.append((intent, result, mode))

    def record_risk_event(self, kind: str, details: dict) -> None:
        self.risk_events.append((kind, details))

    def record_risk_state(self, state) -> None:
        self.risk_states.append(state)


class _SequencedExecutor:
    def __init__(self, responses: dict[str, list[OrderResult]]) -> None:
        self._responses = responses

    def get_order_result(self, order_id: str, intent: OrderIntent) -> OrderResult | None:
        _ = intent
        queue = self._responses.get(order_id)
        if not queue:
            return None
        return queue.pop(0)

    def cancel_all(self) -> None:
        return


class _NoopPostOnlyExecutor:
    def __init__(self) -> None:
        self.cancel_calls = 0
        self.place_calls = 0

    def place_order(self, intent: OrderIntent, books: dict) -> OrderResult:
        _ = books
        self.place_calls += 1
        return OrderResult(
            order_id=f"oid-new-{self.place_calls}",
            market_id=intent.market_id,
            token_id=intent.token_id,
            side=intent.side,
            price=intent.price,
            size=intent.size,
            status="open",
            filled_size=0.0,
            filled_price=0.0,
            fee_paid=0.0,
            engine=intent.engine,
            created_at=datetime.now(tz=timezone.utc),
            raw={},
        )

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        self.cancel_calls += 1
        return True

    def cancel_all(self) -> None:
        return


class _CrossRejectExecutor:
    def __init__(self) -> None:
        self.place_calls = 0
        self.cancel_calls = 0

    def place_order(self, intent: OrderIntent, books: dict) -> OrderResult:
        _ = books
        self.place_calls += 1
        return OrderResult(
            order_id=f"oid-cross-{self.place_calls}",
            market_id=intent.market_id,
            token_id=intent.token_id,
            side=intent.side,
            price=intent.price,
            size=intent.size,
            status="error",
            filled_size=0.0,
            filled_price=0.0,
            fee_paid=0.0,
            engine=intent.engine,
            created_at=datetime.now(tz=timezone.utc),
            raw={"error": "invalid post-only order: order crosses book"},
        )

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        self.cancel_calls += 1
        return True

    def cancel_all(self) -> None:
        return


class _InventoryExecutor:
    def __init__(self, cash: float, token_balances: dict[str, float]) -> None:
        self.cash = float(cash)
        self.token_balances = dict(token_balances)
        self.cash_calls = 0
        self.token_calls: list[str] = []

    def get_collateral_balance(self) -> float:
        self.cash_calls += 1
        return self.cash

    def get_token_balance(self, token_id: str) -> float:
        self.token_calls.append(token_id)
        return float(self.token_balances.get(token_id, 0.0))


class _TradeFill:
    def __init__(
        self,
        *,
        trade_id: str,
        order_id: str,
        token_id: str,
        side: Side,
        size: float,
        price: float,
        match_time: int,
    ) -> None:
        self.trade_id = trade_id
        self.order_id = order_id
        self.token_id = token_id
        self.side = side
        self.size = size
        self.price = price
        self.match_time = match_time


class _TradeFillExecutor:
    def __init__(self, fills: list[_TradeFill]) -> None:
        self.fills = fills
        self.calls = 0

    def get_recent_maker_trade_fills(self, *, after_ts: int) -> list[_TradeFill]:
        _ = after_ts
        self.calls += 1
        return list(self.fills)


class RuntimeReconciliationTests(unittest.TestCase):
    def _build_runtime(self) -> BotRuntime:
        cfg = test_config(mode="live", bankroll_usdc=100.0)
        runtime = BotRuntime.__new__(BotRuntime)
        runtime.config = cfg
        runtime.risk = RiskManager(cfg)
        runtime.storage = _DummyStorage()
        runtime.quote_order_ids = {}
        runtime.quote_order_notional = {}
        runtime.quote_order_plan = {}
        runtime.quote_order_state = {}
        runtime._last_ioc_submission = {}
        runtime._open_alpha_legs = {}
        runtime._market_buy_exec_state = {}
        runtime._pair_cost_history = deque(maxlen=320)
        runtime._pair_cost_open_legs = {}
        runtime._seen_trade_fills = set()
        runtime._last_inventory_sync_ts = 0.0
        runtime._last_trade_sync_ts = 0
        runtime._cycle_counter = 1
        runtime._keep_running = True
        runtime._record_alpha_fill_for_learning = lambda *_args, **_kwargs: None
        runtime._match_hedge_fill_for_learning = lambda *_args, **_kwargs: None
        runtime._build_immediate_equalizer = lambda **_kwargs: None
        runtime._execute_intents = lambda _intents, _books: (0, 0, 0)
        return runtime

    def test_runtime_preflight_delegates_live_executor(self) -> None:
        runtime = self._build_runtime()
        calls = {"count": 0}

        class _Exec:
            @staticmethod
            def preflight() -> None:
                calls["count"] += 1

        runtime.executor = _Exec()
        runtime.preflight()
        self.assertEqual(calls["count"], 1)

        runtime.config = test_config(mode="paper", bankroll_usdc=100.0)
        runtime.preflight()
        self.assertEqual(calls["count"], 1)

    def test_quote_key_includes_level_id(self) -> None:
        intent_l1 = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.25,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"intent_type": "pair_entry_primary", "quote_level_id": "pair_primary_l1"},
        )
        intent_l2 = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.24,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"intent_type": "pair_entry_primary", "quote_level_id": "pair_primary_l2"},
        )
        self.assertNotEqual(BotRuntime._quote_key(intent_l1), BotRuntime._quote_key(intent_l2))

    def test_truncate_intents_preserves_complete_pair_groups(self) -> None:
        runtime = self._build_runtime()

        def _intent(
            *,
            token_id: str,
            intent_type: str,
            edge: float,
            group: str = "",
        ) -> OrderIntent:
            metadata = {"intent_type": intent_type}
            if group:
                metadata["pair_group_id"] = group
            return OrderIntent(
                market_id="m1",
                token_id=token_id,
                side=Side.BUY,
                price=0.45,
                size=5.0,
                tif=TimeInForce.GTC,
                post_only=True,
                engine="engine_pair_arb",
                expected_edge=edge,
                metadata=metadata,
            )

        intents = [
            _intent(token_id="token-eq", intent_type="equalize", edge=1.0),
            _intent(token_id="token-up-a", intent_type="pair_entry_primary", edge=0.8, group="m1:pair_l1"),
            _intent(token_id="token-down-a", intent_type="pair_completion", edge=0.8, group="m1:pair_l1"),
            _intent(token_id="token-up-b", intent_type="pair_entry_primary", edge=0.7, group="m1:pair_l2"),
            _intent(token_id="token-down-b", intent_type="pair_completion", edge=0.7, group="m1:pair_l2"),
            _intent(token_id="token-alpha", intent_type="alpha_entry", edge=0.6),
        ]

        selected = runtime._truncate_intents_preserving_pair_groups(intents, max_per_cycle=4)
        self.assertEqual(len(selected), 4)
        selected_ids = {i.token_id for i in selected}
        self.assertIn("token-eq", selected_ids)
        self.assertIn("token-up-a", selected_ids)
        self.assertIn("token-down-a", selected_ids)
        self.assertIn("token-alpha", selected_ids)
        self.assertNotIn("token-up-b", selected_ids)
        self.assertNotIn("token-down-b", selected_ids)

    def test_live_reconciliation_applies_delta_fills(self) -> None:
        runtime = self._build_runtime()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.25,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"intent_type": "pair_entry_primary", "quote_level_id": "pair_primary_l1"},
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-1"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-1"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        runtime.executor = _SequencedExecutor(
            {
                "oid-1": [
                    OrderResult(
                        order_id="oid-1",
                        market_id=intent.market_id,
                        token_id=intent.token_id,
                        side=intent.side,
                        price=intent.price,
                        size=intent.size,
                        status="live",
                        filled_size=4.0,
                        filled_price=0.25,
                        fee_paid=0.02,
                        engine=intent.engine,
                        created_at=datetime.now(tz=timezone.utc),
                        raw={"order_state": {"status": "live"}},
                    ),
                    OrderResult(
                        order_id="oid-1",
                        market_id=intent.market_id,
                        token_id=intent.token_id,
                        side=intent.side,
                        price=intent.price,
                        size=intent.size,
                        status="filled",
                        filled_size=6.0,
                        filled_price=0.25,
                        fee_paid=0.03,
                        engine=intent.engine,
                        created_at=datetime.now(tz=timezone.utc),
                        raw={"order_state": {"status": "filled"}},
                    ),
                ]
            }
        )

        fills_first = runtime._process_live_quote_reconciliation({})
        self.assertEqual(fills_first, 1)
        self.assertAlmostEqual(runtime.risk.positions[intent.token_id].size, 4.0, places=6)
        self.assertAlmostEqual(runtime.risk.cash, 98.98, places=6)
        self.assertAlmostEqual(runtime.quote_order_notional[key], 1.5, places=6)
        self.assertEqual(len(runtime.storage.orders), 1)
        self.assertAlmostEqual(runtime.storage.orders[0][1].filled_size, 4.0, places=6)

        fills_second = runtime._process_live_quote_reconciliation({})
        self.assertEqual(fills_second, 1)
        self.assertAlmostEqual(runtime.risk.positions[intent.token_id].size, 6.0, places=6)
        self.assertAlmostEqual(runtime.risk.cash, 98.47, places=6)
        self.assertNotIn(key, runtime.quote_order_ids)
        self.assertNotIn("oid-1", runtime.quote_order_state)
        self.assertEqual(len(runtime.storage.orders), 2)
        self.assertAlmostEqual(runtime.storage.orders[1][1].filled_size, 2.0, places=6)

    def test_replaced_quote_remains_reconciled_until_terminal(self) -> None:
        runtime = self._build_runtime()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.36,
            size=5.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"intent_type": "pair_entry_primary", "quote_level_id": "pair_primary_l1"},
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-late"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-late"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        # Simulate replacing/canceling a quote: it should leave active quote maps
        # but remain in reconciliation until terminal.
        runtime._clear_quote_slot(key, order_id="oid-late", keep_reconcile=True)
        self.assertNotIn(key, runtime.quote_order_ids)
        self.assertNotIn(key, runtime.quote_order_notional)
        self.assertNotIn(key, runtime.quote_order_plan)
        self.assertIn("oid-late", runtime.quote_order_state)
        self.assertIsNone(runtime.quote_order_state["oid-late"].key)

        runtime.executor = _SequencedExecutor(
            {
                "oid-late": [
                    OrderResult(
                        order_id="oid-late",
                        market_id=intent.market_id,
                        token_id=intent.token_id,
                        side=intent.side,
                        price=intent.price,
                        size=intent.size,
                        status="matched",
                        filled_size=5.0,
                        filled_price=0.36,
                        fee_paid=0.0,
                        engine=intent.engine,
                        created_at=datetime.now(tz=timezone.utc),
                        raw={"order_state": {"status": "matched"}},
                    )
                ]
            }
        )

        fills = runtime._process_live_quote_reconciliation({})
        self.assertEqual(fills, 1)
        self.assertAlmostEqual(runtime.risk.positions[intent.token_id].size, 5.0, places=6)
        self.assertNotIn("oid-late", runtime.quote_order_state)

    def test_hold_queue_waits_for_min_dwell_before_refresh(self) -> None:
        runtime = self._build_runtime()
        runtime.executor = _NoopPostOnlyExecutor()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.24,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "quote_level_id": "pair_primary_l1",
                "quote_refresh_seconds": 0.5,
                "quote_max_age_seconds": 2.0,
                "hold_queue": True,
                "min_quote_dwell_seconds": 4.0,
                "picked_side": "primary",
            },
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-existing"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-existing"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        executed, fills, errors = BotRuntime._execute_intents(
            runtime,
            [intent],
            {},
            prune_unplanned_quotes=False,
        )
        self.assertEqual(executed, 0)
        self.assertEqual(fills, 0)
        self.assertEqual(errors, 0)
        self.assertEqual(runtime.executor.place_calls, 0)
        self.assertEqual(runtime.executor.cancel_calls, 0)
        self.assertEqual(runtime.quote_order_ids.get(key), "oid-existing")

    def test_empty_cycle_preserves_recent_5m_quotes(self) -> None:
        runtime = self._build_runtime()
        runtime.executor = _NoopPostOnlyExecutor()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.24,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "quote_level_id": "pair_primary_l1",
                "quote_refresh_seconds": 1.2,
                "quote_max_age_seconds": 6.0,
                "hold_queue": True,
                "min_quote_dwell_seconds": 3.0,
                "picked_side": "primary",
                "timeframe": "5m",
                "seconds_to_end": 220.0,
            },
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-existing"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-existing"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        executed, fills, errors = BotRuntime._execute_intents(
            runtime,
            [],
            {},
            prune_unplanned_quotes=True,
        )
        self.assertEqual(executed, 0)
        self.assertEqual(fills, 0)
        self.assertEqual(errors, 0)
        self.assertEqual(runtime.executor.cancel_calls, 0)
        self.assertIn(key, runtime.quote_order_ids)

    def test_empty_cycle_cancels_stale_5m_quotes(self) -> None:
        runtime = self._build_runtime()
        runtime.executor = _NoopPostOnlyExecutor()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.24,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "quote_level_id": "pair_primary_l1",
                "quote_refresh_seconds": 1.2,
                "quote_max_age_seconds": 2.0,
                "hold_queue": True,
                "min_quote_dwell_seconds": 3.0,
                "picked_side": "primary",
                "timeframe": "5m",
                "seconds_to_end": 220.0,
            },
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-existing"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (
            intent.price,
            intent.size,
            time.time() - 20.0,
        )
        runtime.quote_order_state["oid-existing"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        executed, fills, errors = BotRuntime._execute_intents(
            runtime,
            [],
            {},
            prune_unplanned_quotes=True,
        )
        self.assertEqual(executed, 0)
        self.assertEqual(fills, 0)
        self.assertEqual(errors, 0)
        self.assertEqual(runtime.executor.cancel_calls, 1)
        self.assertNotIn(key, runtime.quote_order_ids)

    def test_post_only_cross_local_is_sanitized_before_place(self) -> None:
        runtime = self._build_runtime()
        runtime.executor = _NoopPostOnlyExecutor()
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.61,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "quote_level_id": "pair_primary_l1",
                "picked_side": "primary",
                "timeframe": "5m",
                "seconds_to_end": 220.0,
            },
        )
        book = OrderBookSnapshot(
            token_id="token-up",
            timestamp_ms=0,
            bids=[OrderBookLevel(price=0.60, size=100.0)],
            asks=[OrderBookLevel(price=0.61, size=100.0)],
        )
        executed, fills, errors = BotRuntime._execute_intents(
            runtime,
            [intent],
            {"token-up": book},
            prune_unplanned_quotes=False,
        )
        self.assertEqual(executed, 1)
        self.assertEqual(fills, 0)
        self.assertEqual(errors, 0)
        self.assertEqual(runtime.executor.place_calls, 1)
        key = runtime._quote_key(intent)
        self.assertIn(key, runtime.quote_order_plan)
        placed_price = float(runtime.quote_order_plan[key][0])
        self.assertLess(placed_price, book.best_ask)
        self.assertAlmostEqual(placed_price, 0.60, places=2)
        order_id = runtime.quote_order_ids.get(key)
        self.assertIsNotNone(order_id)
        state = runtime.quote_order_state.get(str(order_id))
        self.assertIsNotNone(state)
        assert state is not None
        sanitized_intent = state.intent
        self.assertTrue(bool(sanitized_intent.metadata.get("post_only_sanitized")))
        self.assertAlmostEqual(float(sanitized_intent.metadata.get("post_only_sanitized_from", 0.0)), 0.61, places=2)

    def test_live_inventory_sync_updates_cash_and_position_sizes(self) -> None:
        runtime = self._build_runtime()
        runtime.risk.cash = 100.0
        runtime.risk.positions["token-up"] = PositionState(
            token_id="token-up",
            market_id="m1",
            size=2.0,
            average_price=0.42,
        )
        runtime.quote_order_state["oid-down"] = QuoteOrderState(
            key=None,
            intent=OrderIntent(
                market_id="m1",
                token_id="token-down",
                side=Side.BUY,
                price=0.45,
                size=10.0,
                tif=TimeInForce.GTC,
                post_only=True,
                engine="engine_pair_arb",
                expected_edge=0.0,
                metadata={
                    "intent_type": "pair_entry_secondary",
                    "quote_level_id": "pair_secondary_l1",
                },
            ),
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )
        runtime.executor = _InventoryExecutor(
            cash=81.0,
            token_balances={"token-up": 35.0, "token-down": 35.0},
        )

        runtime._sync_live_inventory_from_exchange(now_ts=time.time())

        self.assertAlmostEqual(runtime.risk.cash, 81.0, places=6)
        self.assertAlmostEqual(runtime.risk.positions["token-up"].size, 35.0, places=6)
        self.assertAlmostEqual(
            runtime.risk.positions["token-down"].size, 35.0, places=6
        )
        self.assertEqual(runtime.executor.cash_calls, 1)
        self.assertEqual(set(runtime.executor.token_calls), {"token-up", "token-down"})

    def test_inventory_sync_error_records_risk_event(self) -> None:
        runtime = self._build_runtime()

        class _FailingInventoryExecutor:
            @staticmethod
            def get_collateral_balance() -> float:
                raise RuntimeError("auth failed")

            @staticmethod
            def get_token_balance(_token_id: str) -> float:
                return 0.0

        runtime.executor = _FailingInventoryExecutor()
        runtime._sync_live_inventory_from_exchange(now_ts=time.time())
        kinds = [kind for kind, _ in runtime.storage.risk_events]
        self.assertIn("inventory_sync_error", kinds)

    def test_cash_pacing_uses_live_equity_baseline(self) -> None:
        runtime = self._build_runtime()
        runtime.risk.cash = 483.31
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.50,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"timeframe": "5m", "seconds_to_end": 240.0},
        )
        headroom = runtime._buy_cash_pacing_headroom(
            intent,
            intent_type="pair_entry_primary",
        )
        self.assertIsNotNone(headroom)
        assert headroom is not None
        self.assertGreater(headroom, 0.0)

    def test_live_trade_reconcile_records_fill_and_updates_quote_state(self) -> None:
        runtime = self._build_runtime()
        runtime.risk.positions["token-up"] = PositionState(
            token_id="token-up",
            market_id="m1",
            size=0.0,
            average_price=0.0,
        )
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.20,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={"intent_type": "pair_completion", "quote_level_id": "l1"},
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-1"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-1"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )
        runtime._last_trade_sync_ts = 100
        runtime.executor = _TradeFillExecutor(
            [
                _TradeFill(
                    trade_id="trade-1",
                    order_id="oid-1",
                    token_id="token-up",
                    side=Side.BUY,
                    size=4.0,
                    price=0.20,
                    match_time=101,
                )
            ]
        )

        count = runtime._process_live_trade_reconciliation({})
        self.assertEqual(count, 1)
        self.assertEqual(len(runtime.storage.orders), 1)
        state = runtime.quote_order_state.get("oid-1")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertAlmostEqual(float(state.cum_filled_size), 4.0, places=6)
        self.assertAlmostEqual(runtime.quote_order_notional[key], 1.2, places=6)

        count_second = runtime._process_live_trade_reconciliation({})
        self.assertEqual(count_second, 0)

    def test_live_trade_reconcile_reuses_tracked_intent_for_immediate_hedge(self) -> None:
        runtime = self._build_runtime()
        captured_parent_types: list[str] = []

        def _capture_hedge(**kwargs):
            parent_intent = kwargs["parent_intent"]
            captured_parent_types.append(
                str(parent_intent.metadata.get("intent_type") or "")
            )
            return None

        runtime._build_immediate_equalizer = _capture_hedge
        intent = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.20,
            size=10.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "opposite_token_id": "token-down",
                "quote_level_id": "pair_primary_l1",
                "timeframe": "5m",
                "seconds_to_end": 180.0,
            },
        )
        key = runtime._quote_key(intent)
        runtime.quote_order_ids[key] = "oid-1"
        runtime.quote_order_notional[key] = intent.notional
        runtime.quote_order_plan[key] = (intent.price, intent.size, time.time())
        runtime.quote_order_state["oid-1"] = QuoteOrderState(
            key=key,
            intent=intent,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )
        runtime._last_trade_sync_ts = 100
        runtime.executor = _TradeFillExecutor(
            [
                _TradeFill(
                    trade_id="trade-1",
                    order_id="oid-1",
                    token_id="token-up",
                    side=Side.BUY,
                    size=3.0,
                    price=0.20,
                    match_time=101,
                )
            ]
        )

        count = runtime._process_live_trade_reconciliation({})
        self.assertEqual(count, 1)
        self.assertEqual(captured_parent_types, ["pair_entry_primary"])
        self.assertEqual(len(runtime.storage.orders), 1)
        self.assertEqual(
            runtime.storage.orders[0][0].metadata.get("intent_type"),
            "pair_entry_primary",
        )

    def test_pair_group_cross_reject_cancels_existing_sibling_quotes(self) -> None:
        runtime = self._build_runtime()
        runtime.executor = _CrossRejectExecutor()
        existing = OrderIntent(
            market_id="m1",
            token_id="token-up",
            side=Side.BUY,
            price=0.20,
            size=5.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_entry_primary",
                "quote_level_id": "pair_primary_l1",
                "pair_group_id": "m1:pair_l1",
                "picked_side": "primary",
                "timeframe": "5m",
                "seconds_to_end": 220.0,
            },
        )
        existing_key = runtime._quote_key(existing)
        runtime.quote_order_ids[existing_key] = "oid-existing"
        runtime.quote_order_notional[existing_key] = existing.notional
        runtime.quote_order_plan[existing_key] = (
            existing.price,
            existing.size,
            time.time(),
        )
        runtime.quote_order_state["oid-existing"] = QuoteOrderState(
            key=existing_key,
            intent=existing,
            cum_filled_size=0.0,
            cum_fee_paid=0.0,
            cum_notional=0.0,
        )

        incoming = OrderIntent(
            market_id="m1",
            token_id="token-down",
            side=Side.BUY,
            price=0.80,
            size=5.0,
            tif=TimeInForce.GTC,
            post_only=True,
            engine="engine_pair_arb",
            expected_edge=0.0,
            metadata={
                "intent_type": "pair_completion",
                "quote_level_id": "pair_secondary_l1",
                "pair_group_id": "m1:pair_l1",
                "picked_side": "secondary",
                "timeframe": "5m",
                "seconds_to_end": 220.0,
            },
        )
        executed, fills, errors = BotRuntime._execute_intents(
            runtime,
            [incoming],
            {},
            prune_unplanned_quotes=False,
        )
        self.assertEqual(executed, 1)
        self.assertEqual(fills, 0)
        self.assertEqual(errors, 0)
        self.assertEqual(runtime.executor.cancel_calls, 1)
        self.assertNotIn(existing_key, runtime.quote_order_ids)


if __name__ == "__main__":
    unittest.main()

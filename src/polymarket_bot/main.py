from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from polymarket_bot.clients_clob import ClobClient, ClobMarketStream
from polymarket_bot.clients_gamma import GammaClient
from polymarket_bot.clients_spot import BtcSpotClient, BtcSpotStream
from polymarket_bot.config import BotConfig, load_config
from polymarket_bot.engines.engine_pair_arb import PairArbEngine
from polymarket_bot.execution import BaseExecutor, LiveExecutor, PaperExecutor
from polymarket_bot.learning_seed import LearningSeedResult, seed_pair_learning_from_trader_history
from polymarket_bot.learning import PairTimingLearner
from polymarket_bot.models import (
    FeeInfo,
    MarketInfo,
    MarketSnapshot,
    OrderBookSnapshot,
    OrderIntent,
    OrderResult,
    PositionState,
    Side,
    Timeframe,
    TimeInForce,
)
from polymarket_bot.pricing import clamp, per_share_fee, round_tick
from polymarket_bot.risk import RiskManager
from polymarket_bot.runtime_state import (
    AlphaOpenLeg,
    MarketBuyExecutionState,
    QuoteKey,
    QuoteOrderState,
)
from polymarket_bot.runtime_support import BookMotionTracker, FeeCache, SpotTracker
from polymarket_bot.storage import Storage

LOGGER = logging.getLogger("polymarket_bot")
_DURATION_5M_TAG = 102892
_DURATION_15M_TAG = 102467
_DURATION_1H_TAG = 102175


class BotRuntime:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.storage = Storage(config.database_path)
        self.gamma = GammaClient(
            config.gamma_url, timeout_seconds=config.api_timeout_seconds
        )
        self.clob = ClobClient(
            config.clob_url, timeout_seconds=config.api_timeout_seconds
        )
        self.clob_stream = ClobMarketStream(config.clob_ws_url)
        self.spot_rest = BtcSpotClient(
            config.btc_spot_url, timeout_seconds=config.api_timeout_seconds
        )
        self.risk = RiskManager(config)
        self.fee_cache = FeeCache(config.fee_poll_interval_seconds)
        self.spot_tracker = SpotTracker()
        self.book_motion = BookMotionTracker()
        self.pair_learner = PairTimingLearner(self.storage)
        self.spot_stream = BtcSpotStream(
            ws_url=config.btc_spot_ws_url,
            rest_client=self.spot_rest,
            on_price=self._on_spot_tick,
        )
        self.engine_pair = PairArbEngine(config)
        self.quote_order_ids: dict[QuoteKey, str] = {}
        self.quote_order_notional: dict[QuoteKey, float] = {}
        self.quote_order_plan: dict[
            QuoteKey, tuple[float, float, float]
        ] = {}
        self.quote_order_state: dict[str, QuoteOrderState] = {}
        self._seen_trade_fills: set[tuple[str, str]] = set()
        self._last_ioc_submission: dict[tuple[str, ...], tuple[float, float]] = {}
        self._last_inventory_sync_ts = 0.0
        self._last_inventory_report_ts = 0.0
        self._last_exchange_cash: float | None = None
        self._last_exchange_token_sizes: dict[str, float] = {}
        self._last_trade_sync_ts = int(time.time())
        self._last_settlement_ts = 0.0
        self._last_redeem_ts = 0.0
        self._settlement_method_missing = False
        self._redeem_method_missing = False
        self._active_market_by_tf: dict[Timeframe, str] = {}
        self._single_5m_pause_until_ts = 0.0
        self._cycle_counter = 0
        self._last_pair_arb_summary: (
            tuple[int, tuple[tuple[str, int], ...], str, str, int] | None
        ) = None
        self._last_pair_idle_diag: dict[str, tuple[str, int, int]] = {}
        self._no_snapshot_streak = 0
        self._open_alpha_legs: dict[str, list[AlphaOpenLeg]] = {}
        self._market_buy_exec_state: dict[str, MarketBuyExecutionState] = {}
        self._pair_cost_history: deque[tuple[float, float, float]] = deque(maxlen=320)
        self._pair_cost_open_legs: dict[
            str, dict[str, deque[tuple[float, float, float]]]
        ] = {}

        self.executor: BaseExecutor
        if config.live_mode:
            self.executor = LiveExecutor(config)
        else:
            self.executor = PaperExecutor(config)

        self._keep_running = True

    def _on_spot_tick(self, ts: float, price: float) -> None:
        self.spot_tracker.update(ts, price)

    def stop(self) -> None:
        self._keep_running = False

    def preflight(self) -> None:
        if not self.config.live_mode:
            return
        preflight_fn = getattr(self.executor, "preflight", None)
        if preflight_fn is None:
            raise RuntimeError("Live executor does not implement preflight")
        preflight_fn()

    def run(self) -> None:
        self.clob_stream.start()
        self.clob_stream.wait_until_ready(timeout_seconds=6.0)
        self.spot_stream.start()
        self.spot_stream.wait_until_ready(timeout_seconds=6.0)
        while self._keep_running:
            started = time.time()
            self._cycle()
            elapsed = time.time() - started
            sleep_seconds = max(0.0, self.config.poll_interval_seconds - elapsed)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if self.config.live_mode:
            self._finalize_live_quote_reconciliation()

    def close(self) -> None:
        if self.config.live_mode:
            try:
                self._finalize_live_quote_reconciliation(timeout_seconds=2.0)
            except Exception as exc:
                LOGGER.warning("close_reconcile_failed error=%s", exc)
        self.clob_stream.stop()
        self.spot_stream.stop()
        self.storage.close()

    def _cycle(self) -> None:
        cycle_started = time.time()
        self._cycle_counter += 1
        now = datetime.now(tz=timezone.utc)
        self.risk.update_data_heartbeat(now)

        try:
            spot_price, spot_source, spot_age = self.spot_stream.latest(
                max_stale_seconds=max(1.5, self.config.stale_feed_seconds)
            )
        except RuntimeError as exc:
            message = str(exc)
            lowered = message.lower()
            if "stale" in lowered or "not connected" in lowered:
                LOGGER.warning(
                    "cycle=%s spot_unavailable=%s; skipping cycle",
                    self._cycle_counter,
                    message,
                )
                return
            raise
        self.spot_tracker.update(now.timestamp(), spot_price)
        spot_return_60s = self.spot_tracker.return_over_seconds(60)

        all_markets = self.gamma.fetch_btc_markets(
            self.config.enabled_tags, self.config.max_markets_per_tag
        )
        markets = self._select_markets(all_markets)
        self._sync_live_inventory_from_exchange(now_ts=now.timestamp(), markets=markets)
        self._log_inventory_comparison(markets, now_ts=now.timestamp())
        self._sync_quotes_with_selected_markets(markets)
        self._log_market_switches(markets)
        LOGGER.info(
            "cycle=%s markets=%s spot=%.2f ret_60s=%.4f spot_src=%s spot_age=%.2fs",
            self._cycle_counter,
            len(markets),
            spot_price,
            spot_return_60s,
            spot_source,
            spot_age,
        )
        snapshots, books = self._build_snapshots(markets)
        if not snapshots:
            self._no_snapshot_streak += 1
            merged_pairs, redeem_calls = self._settle_pairs_and_redeem([], now=now)
            self.storage.record_risk_state(self.risk.state())
            if self._single_5m_pause_active(now.timestamp()):
                LOGGER.info(
                    "cycle=%s redeem_pause active snapshots=0 merged_pairs=%.2f redeem_calls=%s equity=%.2f elapsed=%.2fs",
                    self._cycle_counter,
                    merged_pairs,
                    redeem_calls,
                    self.risk.current_equity(),
                    time.time() - cycle_started,
                )
            else:
                if (
                    self._no_snapshot_streak == 1
                    or (self._no_snapshot_streak % 15) == 0
                ):
                    LOGGER.info(
                        "cycle=%s no_snapshots streak=%s",
                        self._cycle_counter,
                        self._no_snapshot_streak,
                    )
            return

        self._no_snapshot_streak = 0
        self._update_book_motion(books, now_ts=now.timestamp())
        self._record_snapshots(snapshots)
        marks = {token_id: book.mid for token_id, book in books.items() if book.mid > 0}
        self.risk.mark_to_market(marks)
        max_total = self.config.bankroll_usdc * self.config.max_total_exposure_pct
        max_market = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        market_exposure_text = ", ".join(
            f"{m.market_id}:{self.risk.market_exposure.get(m.market_id, 0.0):.2f}/{max_market:.2f}"
            for m in markets
        )
        LOGGER.info(
            "cycle=%s risk equity=%.2f dd=%.3f exp_total=%.2f/%.2f exp_market=[%s]",
            self._cycle_counter,
            self.risk.current_equity(),
            self.risk.daily_drawdown_pct(),
            self.risk.total_exposure(),
            max_total,
            market_exposure_text,
        )
        self.risk.check_kill_switches(now)
        if self.risk.halted:
            self._cancel_quotes()
            self.storage.record_risk_event(
                "risk_halt", {"reason": self.risk.halted_reason}
            )
            self.storage.record_risk_state(self.risk.state())
            LOGGER.error("Trading halted: %s", self.risk.halted_reason)
            if self.config.live_mode and self._is_fatal_halt_reason(
                self.risk.halted_reason
            ):
                LOGGER.error("Stopping live loop after risk halt")
                self.stop()
                return

        fair_by_market = self._compute_fair_map(snapshots, spot_price)
        intents = self._generate_intents(snapshots, fair_by_market)
        executed_count, fill_count, error_count = self._execute_intents(intents, books)
        if self.risk.halted:
            self._cancel_quotes()
            self.storage.record_risk_event(
                "risk_halt", {"reason": self.risk.halted_reason}
            )
            self.storage.record_risk_state(self.risk.state())
            LOGGER.error("Trading halted: %s", self.risk.halted_reason)
            if self.config.live_mode and self._is_fatal_halt_reason(
                self.risk.halted_reason
            ):
                LOGGER.error("Stopping live loop after risk halt")
                self.stop()
                return
        trade_fill_count = 0
        if self.config.live_mode:
            trade_fill_count = self._process_live_trade_reconciliation(books)
            sweep_fill_count = self._process_live_quote_reconciliation(books)
        else:
            sweep_fill_count = self._process_paper_sweeps(books)
        merged_pairs, redeem_calls = self._settle_pairs_and_redeem(snapshots, now=now)
        self.storage.record_risk_state(self.risk.state())
        LOGGER.info(
            "cycle=%s snapshots=%s intents=%s executed=%s fills=%s trade_fills=%s sweep_fills=%s merged_pairs=%.2f redeem_calls=%s exec_errors=%s equity=%.2f elapsed=%.2fs",
            self._cycle_counter,
            len(snapshots),
            len(intents),
            executed_count,
            fill_count,
            trade_fill_count,
            sweep_fill_count,
            merged_pairs,
            redeem_calls,
            error_count,
            self.risk.current_equity(),
            time.time() - cycle_started,
        )

    def _single_5m_pause_active(self, now_ts: float | None = None) -> bool:
        if not (
            self.config.single_5m_deep_mode and self.config.single_5m_pause_for_redeem
        ):
            return False
        pause_until = float(getattr(self, "_single_5m_pause_until_ts", 0.0) or 0.0)
        if pause_until <= 0.0:
            return False
        if pause_until == float("inf"):
            return True
        current_ts = float(now_ts if now_ts is not None else time.time())
        if current_ts >= pause_until:
            self._single_5m_pause_until_ts = 0.0
            LOGGER.info("single_5m_pause_end")
            return False
        return True

    def _select_markets(self, markets: list[MarketInfo]) -> list[MarketInfo]:
        now = datetime.now(tz=timezone.utc)
        open_market_ids = {
            position.market_id
            for position in self.risk.positions.values()
            if position.size > 0
        }
        by_timeframe: dict[Timeframe, list[MarketInfo]] = {
            Timeframe.FIVE_MIN: [],
            Timeframe.FIFTEEN_MIN: [],
            Timeframe.ONE_HOUR: [],
            Timeframe.UNKNOWN: [],
        }
        for market in markets:
            seconds_to_start = (market.start_time - now).total_seconds()
            if seconds_to_start > 0:
                continue
            seconds_to_end = (market.end_time - now).total_seconds()
            keep_for_unwind = market.market_id in open_market_ids
            if (not keep_for_unwind) and seconds_to_end <= self._rollover_guard_seconds(
                market.timeframe
            ):
                continue
            by_timeframe.setdefault(market.timeframe, []).append(market)

        for candidates in by_timeframe.values():
            candidates.sort(key=lambda m: m.end_time)

        if self.config.single_5m_deep_mode:
            selected_5m: list[MarketInfo] = []
            for market in by_timeframe[Timeframe.FIVE_MIN]:
                if market.market_id in open_market_ids:
                    selected_5m.append(market)
                    break
            if self._single_5m_pause_active(now.timestamp()):
                return selected_5m
            if selected_5m:
                return selected_5m
            if self.config.max_trade_markets_5m <= 0:
                return []
            if by_timeframe[Timeframe.FIVE_MIN]:
                selected_5m.append(by_timeframe[Timeframe.FIVE_MIN][0])
            return selected_5m

        cap_5m = self.config.max_trade_markets_5m
        cap_15m = self.config.max_trade_markets_15m
        cap_1h = self.config.max_trade_markets_1h
        if self.config.bankroll_usdc <= 100.0:
            # Keep small bankroll recycling on shorter tenors and
            # scan wider for dislocations.
            cap_1h = 0 if self.config.max_trade_markets_1h <= 0 else cap_1h
            if self.config.max_trade_markets_15m > 0:
                cap_15m = max(cap_15m, 6)
            if self.config.max_trade_markets_5m > 0:
                cap_5m = max(cap_5m, 8)

        selected: list[MarketInfo] = []
        selected_ids: set[str] = set()

        def include_open_markets(timeframe: Timeframe) -> None:
            for market in by_timeframe[timeframe]:
                if market.market_id not in open_market_ids:
                    continue
                if market.market_id in selected_ids:
                    continue
                selected.append(market)
                selected_ids.add(market.market_id)

        def include_fresh_markets(timeframe: Timeframe, cap: int) -> None:
            if cap <= 0:
                return
            count = 0
            for market in by_timeframe[timeframe]:
                if market.market_id in open_market_ids:
                    continue
                if market.market_id in selected_ids:
                    continue
                selected.append(market)
                selected_ids.add(market.market_id)
                count += 1
                if count >= cap:
                    break

        include_open_markets(Timeframe.FIVE_MIN)
        include_open_markets(Timeframe.FIFTEEN_MIN)
        include_open_markets(Timeframe.ONE_HOUR)
        include_fresh_markets(Timeframe.FIVE_MIN, cap_5m)
        include_fresh_markets(Timeframe.FIFTEEN_MIN, cap_15m)
        include_fresh_markets(Timeframe.ONE_HOUR, cap_1h)
        selected.sort(key=lambda m: m.end_time)
        return selected

    def _rollover_guard_seconds(self, timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            return float(self.config.directional_min_time_left_5m)
        if timeframe == Timeframe.FIFTEEN_MIN:
            return float(self.config.directional_min_time_left_15m)
        if timeframe == Timeframe.ONE_HOUR:
            return 180.0
        return 30.0

    def _log_market_switches(self, markets: list[MarketInfo]) -> None:
        seen_timeframes: set[Timeframe] = set()
        for market in markets:
            seen_timeframes.add(market.timeframe)
            previous = self._active_market_by_tf.get(market.timeframe)
            if previous == market.market_id:
                continue
            self._active_market_by_tf[market.timeframe] = market.market_id
            LOGGER.info(
                "market_switch tf=%s market=%s seconds_to_end=%.0f",
                market.timeframe.value,
                market.market_id,
                market.seconds_to_end,
            )
            if (
                self.config.single_5m_deep_mode
                and self.config.single_5m_pause_for_redeem
                and market.timeframe == Timeframe.FIVE_MIN
                and previous
            ):
                pause_seconds = max(0.0, float(self.config.single_5m_pause_seconds))
                if pause_seconds <= 0:
                    self._single_5m_pause_until_ts = float("inf")
                    pause_until_text = "manual_resume"
                else:
                    pause_until = time.time() + pause_seconds
                    self._single_5m_pause_until_ts = pause_until
                    pause_until_text = datetime.fromtimestamp(
                        pause_until, tz=timezone.utc
                    ).isoformat()
                self.storage.record_risk_event(
                    "single_5m_pause_start",
                    {
                        "completed_market": previous,
                        "next_market": market.market_id,
                        "pause_seconds": pause_seconds,
                        "pause_until": pause_until_text,
                    },
                )
                LOGGER.info(
                    "single_5m_pause_start completed_market=%s next_market=%s pause_seconds=%.0f pause_until=%s",
                    previous,
                    market.market_id,
                    pause_seconds,
                    pause_until_text,
                )
        for timeframe in list(self._active_market_by_tf.keys()):
            if timeframe not in seen_timeframes:
                self._active_market_by_tf.pop(timeframe, None)

    @staticmethod
    def _quote_key(intent: OrderIntent) -> QuoteKey:
        intent_type = str(intent.metadata.get("intent_type") or "quote")
        level_id = str(intent.metadata.get("quote_level_id") or "l1")
        return (
            intent.engine,
            intent.market_id,
            intent.token_id,
            intent.side.value,
            intent_type,
            level_id,
        )

    def _ensure_market_buy_exec_state(self, market_id: str) -> MarketBuyExecutionState:
        state = self._market_buy_exec_state.get(market_id)
        if state is None:
            state = MarketBuyExecutionState()
            self._market_buy_exec_state[market_id] = state
        return state

    @staticmethod
    def _quote_plan_age_seconds(
        plan: tuple[float, float, float] | None, *, now_ts: float
    ) -> float:
        if not plan:
            return float("inf")
        try:
            placed_ts = float(plan[2])
        except (TypeError, ValueError, IndexError):
            return float("inf")
        if placed_ts <= 0:
            return float("inf")
        return max(0.0, now_ts - placed_ts)

    def _should_preserve_quote_on_empty_intent_cycle(
        self,
        *,
        key: QuoteKey,
        order_id: str | None,
        now_ts: float,
    ) -> bool:
        if not order_id:
            return False
        state = self.quote_order_state.get(order_id)
        if state is None:
            return False
        tracked_intent = state.intent
        timeframe = str(tracked_intent.metadata.get("timeframe") or "")
        if timeframe != Timeframe.FIVE_MIN.value:
            return False
        hold_queue = bool(tracked_intent.metadata.get("hold_queue", False))
        try:
            quote_max_age_seconds = max(
                0.8,
                float(tracked_intent.metadata.get("quote_max_age_seconds", 6.0)),
            )
        except (TypeError, ValueError):
            quote_max_age_seconds = 6.0
        age = self._quote_plan_age_seconds(self.quote_order_plan.get(key), now_ts=now_ts)
        grace_multiplier = 1.85 if hold_queue else 1.35
        grace_seconds = quote_max_age_seconds * grace_multiplier
        if self.config.single_5m_deep_mode:
            grace_seconds = max(grace_seconds, 12.0)
        try:
            seconds_to_end = float(tracked_intent.metadata.get("seconds_to_end", 0.0))
        except (TypeError, ValueError):
            seconds_to_end = 0.0
        if 0.0 < seconds_to_end <= 12.0:
            return False
        return age <= grace_seconds

    def _clear_quote_slot(
        self,
        key: QuoteKey,
        *,
        order_id: str | None = None,
        keep_reconcile: bool = False,
    ) -> None:
        tracked_id = order_id or self.quote_order_ids.get(key)
        self.quote_order_ids.pop(key, None)
        self.quote_order_notional.pop(key, None)
        self.quote_order_plan.pop(key, None)
        if not tracked_id:
            return
        if keep_reconcile:
            existing = self.quote_order_state.get(tracked_id)
            if existing is not None:
                existing.key = None
                existing.retired_ts = time.time()
                self.quote_order_state[tracked_id] = existing
                return
        self.quote_order_state.pop(tracked_id, None)

    def _sync_quotes_with_selected_markets(self, markets: list[MarketInfo]) -> None:
        selected_ids = {market.market_id for market in markets}
        stale_keys: list[QuoteKey] = []
        for key in list(self.quote_order_ids.keys()):
            market_id = key[1]
            if market_id not in selected_ids:
                stale_keys.append(key)
        if not stale_keys:
            return
        cancelled = 0
        for key in stale_keys:
            order_id = self.quote_order_ids.get(key)
            if order_id and self.executor.cancel_order(order_id):
                cancelled += 1
            self._clear_quote_slot(key, order_id=order_id, keep_reconcile=True)
        LOGGER.info(
            "cycle=%s stale_quote_cleanup cancelled=%s remaining_open_quotes=%s",
            self._cycle_counter,
            cancelled,
            len(self.quote_order_ids),
        )

    def _tracked_inventory_token_market_map(self) -> dict[str, str]:
        tracked: dict[str, str] = {}
        for token_id, position in self.risk.positions.items():
            cleaned_token = str(token_id or "").strip()
            cleaned_market = str(position.market_id or "").strip()
            if cleaned_token and cleaned_market:
                tracked[cleaned_token] = cleaned_market
        for state in self.quote_order_state.values():
            intent = state.intent
            cleaned_token = str(intent.token_id or "").strip()
            cleaned_market = str(intent.market_id or "").strip()
            if cleaned_token and cleaned_market:
                tracked.setdefault(cleaned_token, cleaned_market)
        return tracked

    def _sync_live_inventory_from_exchange(
        self,
        *,
        now_ts: float,
        markets: list[MarketInfo] | None = None,
        force: bool = False,
    ) -> None:
        if not self.config.live_mode:
            return
        last_sync = float(getattr(self, "_last_inventory_sync_ts", 0.0) or 0.0)
        if (not force) and (now_ts - last_sync) < 1.5:
            return

        get_collateral_balance = getattr(self.executor, "get_collateral_balance", None)
        get_token_balance = getattr(self.executor, "get_token_balance", None)
        if not callable(get_collateral_balance) or not callable(get_token_balance):
            return

        token_market_map = self._tracked_inventory_token_market_map()
        if markets:
            for market in markets:
                market_id = str(market.market_id or "").strip()
                primary_token_id = str(market.primary_token_id or "").strip()
                secondary_token_id = str(market.secondary_token_id or "").strip()
                if market_id and primary_token_id:
                    token_market_map.setdefault(primary_token_id, market_id)
                if market_id and secondary_token_id:
                    token_market_map.setdefault(secondary_token_id, market_id)
        try:
            collateral_balance = float(get_collateral_balance())
            token_sizes: dict[str, float] = {}
            for token_id in sorted(token_market_map.keys()):
                token_sizes[token_id] = float(get_token_balance(token_id))
            sync_result = self.risk.sync_exchange_inventory(
                cash=collateral_balance,
                token_sizes=token_sizes,
                token_market_ids=token_market_map,
            )
        except Exception as exc:
            self.storage.record_risk_event(
                "inventory_sync_error",
                {"error": str(exc)},
            )
            LOGGER.warning("cycle=%s inventory_sync_error=%s", self._cycle_counter, exc)
            return

        self._last_inventory_sync_ts = now_ts
        self._last_exchange_cash = collateral_balance
        self._last_exchange_token_sizes = token_sizes
        cash_delta = sync_result.cash_after - sync_result.cash_before
        significant_cash_drift = abs(cash_delta) >= 0.01
        if (
            not significant_cash_drift
            and not sync_result.token_size_changes
            and not sync_result.unknown_market_tokens
        ):
            return

        drift_tokens = [
            {
                "token_id": token_id,
                "local_size": before,
                "exchange_size": after,
            }
            for token_id, (before, after) in sorted(
                sync_result.token_size_changes.items()
            )
        ]
        self.storage.record_risk_event(
            "inventory_sync_drift",
            {
                "cash_before": sync_result.cash_before,
                "cash_after": sync_result.cash_after,
                "cash_delta": cash_delta,
                "token_count": len(drift_tokens),
                "tokens": drift_tokens[:12],
                "unknown_market_tokens": list(sync_result.unknown_market_tokens),
            },
        )
        LOGGER.warning(
            "cycle=%s inventory_sync_drift cash=%.2f->%.2f token_drifts=%s unknown_market_tokens=%s",
            self._cycle_counter,
            sync_result.cash_before,
            sync_result.cash_after,
            len(drift_tokens),
            len(sync_result.unknown_market_tokens),
        )

    def _log_inventory_comparison(
        self,
        markets: list[MarketInfo],
        *,
        now_ts: float,
    ) -> None:
        if not self.config.live_mode:
            return
        if (now_ts - float(self._last_inventory_report_ts or 0.0)) < 60.0:
            return
        if not markets:
            return
        if self._last_exchange_cash is None:
            return
        lines: list[str] = []
        for market in markets[:2]:
            market_id = str(market.market_id or "").strip()
            primary_token = str(market.primary_token_id or "").strip()
            secondary_token = str(market.secondary_token_id or "").strip()
            if not market_id or not primary_token or not secondary_token:
                continue
            primary_local = max(
                0.0, float(self.risk.positions.get(primary_token, PositionState(token_id=primary_token, market_id=market_id)).size)
            )
            secondary_local = max(
                0.0, float(self.risk.positions.get(secondary_token, PositionState(token_id=secondary_token, market_id=market_id)).size)
            )
            primary_exchange = max(
                0.0, float(self._last_exchange_token_sizes.get(primary_token, 0.0))
            )
            secondary_exchange = max(
                0.0, float(self._last_exchange_token_sizes.get(secondary_token, 0.0))
            )
            lines.append(
                (
                    "market=%s up_local=%.2f up_exchange=%.2f down_local=%.2f down_exchange=%.2f"
                    % (
                        market_id,
                        primary_local,
                        primary_exchange,
                        secondary_local,
                        secondary_exchange,
                    )
                )
            )
        if not lines:
            return
        LOGGER.info(
            "cycle=%s inventory_compare cash_local=%.2f cash_exchange=%.2f %s",
            self._cycle_counter,
            float(self.risk.cash),
            float(self._last_exchange_cash),
            " | ".join(lines),
        )
        self._last_inventory_report_ts = now_ts

    def _build_snapshots(
        self, markets: list[MarketInfo]
    ) -> tuple[list[MarketSnapshot], dict[str, OrderBookSnapshot]]:
        snapshots: list[MarketSnapshot] = []
        books: dict[str, OrderBookSnapshot] = {}

        token_ids: list[str] = sorted(
            {
                token_id
                for market in markets
                for token_id in (market.primary_token_id, market.secondary_token_id)
                if token_id
            }
        )
        if token_ids:
            try:
                self.clob_stream.assert_healthy()
                self.clob_stream.set_assets(token_ids)
                books.update(self.clob_stream.get_books(token_ids))
            except RuntimeError as exc:
                LOGGER.warning(
                    "cycle=%s clob_ws_unavailable=%s; using rest fallback for missing books",
                    self._cycle_counter,
                    exc,
                )
            missing_token_ids = [
                token_id for token_id in token_ids if token_id not in books
            ]
            if missing_token_ids:
                restored = 0
                backfill_limit = (
                    len(missing_token_ids)
                    if not books
                    else min(18, len(missing_token_ids))
                )
                for token_id in missing_token_ids[:backfill_limit]:
                    if not self._keep_running:
                        break
                    try:
                        fallback_book = self.clob.get_book(token_id)
                    except Exception as exc:
                        LOGGER.debug(
                            "book_rest_backfill_failed token=%s error=%s", token_id, exc
                        )
                        continue
                    if fallback_book.best_bid <= 0 and fallback_book.best_ask <= 0:
                        continue
                    books[token_id] = fallback_book
                    restored += 1
                if restored > 0:
                    LOGGER.info(
                        "cycle=%s backfilled_books_rest=%s/%s",
                        self._cycle_counter,
                        restored,
                        len(missing_token_ids),
                    )

        fees: dict[str, FeeInfo] = {}
        if token_ids:
            for token_id in token_ids:
                if not self._keep_running:
                    break
                try:
                    fees[token_id] = self.fee_cache.get(token_id, self.clob)
                except Exception as exc:
                    LOGGER.warning("Fee fetch failed token=%s: %s", token_id, exc)

        missing_market_books = 0
        for market in markets:
            if not self._keep_running:
                break
            if not market.primary_token_id or not market.secondary_token_id:
                continue
            primary_book = books.get(market.primary_token_id)
            secondary_book = books.get(market.secondary_token_id)
            if primary_book is None or secondary_book is None:
                missing_market_books += 1
                continue
            primary_fee = fees.get(market.primary_token_id)
            secondary_fee = fees.get(market.secondary_token_id)
            if primary_fee is None or secondary_fee is None:
                LOGGER.warning(
                    "Skipping market=%s due to missing fee data", market.market_id
                )
                continue
            snapshots.append(
                MarketSnapshot(
                    market=market,
                    primary_book=primary_book,
                    secondary_book=secondary_book,
                    primary_fee=primary_fee,
                    secondary_fee=secondary_fee,
                )
            )
        if missing_market_books > 0:
            LOGGER.warning(
                "cycle=%s skipped_markets_missing_books=%s",
                self._cycle_counter,
                missing_market_books,
            )
        return snapshots, books

    def _record_snapshots(self, snapshots: list[MarketSnapshot]) -> None:
        for snapshot in snapshots:
            self.storage.record_snapshot(snapshot)

    def _update_book_motion(
        self, books: dict[str, OrderBookSnapshot], *, now_ts: float
    ) -> None:
        for token_id, book in books.items():
            self.book_motion.update_book(
                now_ts=now_ts,
                token_id=token_id,
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                mid=book.mid,
            )

    @staticmethod
    def _motion_window_seconds(timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            return 55.0
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 150.0
        if timeframe == Timeframe.ONE_HOUR:
            return 360.0
        return 90.0

    def _horizon_seconds(self, timeframe: Timeframe) -> int:
        if timeframe == Timeframe.FIVE_MIN:
            return 300
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 900
        if timeframe == Timeframe.ONE_HOUR:
            return 3600
        return 900

    def _compute_fair_probability(
        self, snapshot: MarketSnapshot, spot_price: float
    ) -> float:
        horizon = self._horizon_seconds(snapshot.market.timeframe)
        seconds_to_end = max(0.0, snapshot.market.seconds_to_end)
        ret_30 = self.spot_tracker.return_over_seconds(min(30, horizon))
        ret_120 = self.spot_tracker.return_over_seconds(min(120, horizon))
        ret_600 = self.spot_tracker.return_over_seconds(min(600, horizon))
        drift_raw = (0.45 * ret_30) + (0.35 * ret_120) + (0.20 * ret_600)
        ret_votes = [r for r in (ret_30, ret_120, ret_600) if abs(r) >= 0.0004]
        if ret_votes:
            vote_score = sum(1.0 if value > 0 else -1.0 for value in ret_votes)
            trend_consensus = abs(vote_score) / len(ret_votes)
        else:
            trend_consensus = 0.0
        drift = drift_raw * (0.55 + (0.45 * trend_consensus))

        vol_step = self.spot_tracker.realized_volatility(seconds=300)
        sigma = max(0.0012, vol_step * math.sqrt(max(1.0, horizon / 300.0)))
        z = drift / sigma
        p_base = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        pbid = snapshot.primary_book.best_bid
        pask = snapshot.primary_book.best_ask
        sbid = snapshot.secondary_book.best_bid
        sask = snapshot.secondary_book.best_ask
        synth_bid = max(pbid, 1.0 - sask if sask > 0 else 0.0)
        synth_ask = min(pask if pask > 0 else 1.0, 1.0 - sbid if sbid > 0 else 1.0)
        synth_mid = None
        synth_spread = None
        synth_quality = 0.0
        if synth_bid <= synth_ask:
            synth_mid = (synth_bid + synth_ask) / 2.0
            synth_spread = synth_ask - synth_bid
            # 0 when spread is wide (>= 20c), 1 when spread is tight.
            synth_quality = clamp((0.20 - synth_spread) / 0.20, 0.0, 1.0)

        primary_imb = snapshot.primary_book.top_size_imbalance(levels=2)
        secondary_imb = snapshot.secondary_book.top_size_imbalance(levels=2)
        pressure = clamp((primary_imb - secondary_imb) * 0.5, -1.0, 1.0)
        pressure_weight = 0.05 + (0.10 * synth_quality)
        p_orderbook = p_base + (pressure_weight * pressure)

        if synth_mid is not None and synth_spread is not None and synth_spread <= 0.30:
            # As expiry nears and the synthetic spread tightens, force the model
            # toward executable market-implied probability.
            expiry_phase = clamp(
                1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0
            )
            synth_weight = 0.18 + (0.52 * expiry_phase * synth_quality)
            p_orderbook = ((1.0 - synth_weight) * p_orderbook) + (synth_weight * synth_mid)

        horizon_remaining = max(1, int(min(float(horizon), seconds_to_end)))
        p_rnjd = self.spot_tracker.rnjd_probability(horizon_remaining)
        spot_points = self.spot_tracker.point_count()
        spot_age = self.spot_tracker.latest_age_seconds()
        if spot_points >= 120:
            rnjd_weight = 0.35
        elif spot_points >= 40:
            rnjd_weight = 0.22
        else:
            rnjd_weight = 0.10
        rnjd_weight *= 0.70 + (0.30 * trend_consensus)
        if spot_age > 2.0:
            rnjd_weight *= 0.5
        if synth_quality >= 0.75 and seconds_to_end <= (float(horizon) * 0.40):
            rnjd_weight *= 0.85
        p_blend = ((1.0 - rnjd_weight) * p_orderbook) + (rnjd_weight * p_rnjd)

        primary_label = snapshot.market.primary_label.lower()
        secondary_label = snapshot.market.secondary_label.lower()
        is_updown = ("up" in primary_label and "down" in secondary_label) or (
            "up" in secondary_label and "down" in primary_label
        )
        if is_updown:
            start_price = self.spot_tracker.price_at_or_before(
                snapshot.market.start_time.timestamp(),
                max_lookback_seconds=float(horizon) + 180.0,
            )
            if start_price is not None and start_price > 0 and spot_price > 0:
                target_log_return = math.log(start_price / spot_price)
                p_up_anchor = self.spot_tracker.rnjd_probability_above_log_return(
                    horizon_remaining,
                    target_log_return=target_log_return,
                )
                expiry_phase = clamp(
                    1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0
                )
                anchor_weight = 0.30 + (0.45 * expiry_phase) + (0.15 * (1.0 - synth_quality))
                anchor_weight = clamp(anchor_weight, 0.25, 0.88)
                p_blend = ((1.0 - anchor_weight) * p_blend) + (
                    anchor_weight * p_up_anchor
                )

        if synth_mid is not None and synth_spread is not None and synth_spread <= 0.30:
            # Final safety clamp: near expiry, model fair cannot diverge too far
            # from executable market-implied probability.
            expiry_phase = clamp(
                1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0
            )
            max_divergence = 0.35 - (0.27 * expiry_phase)  # 35% far out -> 8% near expiry
            max_divergence *= 1.0 - (0.50 * synth_quality)
            max_divergence = clamp(max_divergence, 0.05, 0.35)
            p_blend = clamp(
                p_blend, synth_mid - max_divergence, synth_mid + max_divergence
            )

        return clamp(p_blend, 0.03, 0.97)

    def _compute_fair_map(
        self, snapshots: list[MarketSnapshot], spot_price: float
    ) -> dict[str, float]:
        fair_map: dict[str, float] = {}
        for snapshot in snapshots:
            fair_map[snapshot.market.market_id] = self._compute_fair_probability(
                snapshot, spot_price
            )
        return fair_map

    def _settle_pairs_and_redeem(
        self, snapshots: list[MarketSnapshot], *, now: datetime
    ) -> tuple[float, int]:
        merged_total = 0.0
        redeem_calls = 0
        now_ts = now.timestamp()

        settlement_interval = 2.0
        if now_ts - self._last_settlement_ts >= settlement_interval:
            for snapshot in snapshots:
                self.executor.register_condition(
                    condition_id=snapshot.market.condition_id,
                    is_neg_risk=snapshot.market.is_neg_risk,
                    market_end_ts=snapshot.market.end_time.timestamp(),
                )
                primary = self.risk.positions.get(snapshot.market.primary_token_id)
                secondary = self.risk.positions.get(snapshot.market.secondary_token_id)
                primary_size = primary.size if primary and primary.size > 0 else 0.0
                secondary_size = (
                    secondary.size if secondary and secondary.size > 0 else 0.0
                )
                pair_size = min(primary_size, secondary_size)
                min_size = max(snapshot.market.order_min_size, 0.0)
                if pair_size < min_size:
                    continue

                primary_avg = primary.average_price if primary else 0.0
                secondary_avg = secondary.average_price if secondary else 0.0
                pair_avg_cost = primary_avg + secondary_avg
                near_end = snapshot.market.seconds_to_end <= max(
                    12.0,
                    self._rollover_guard_seconds(snapshot.market.timeframe) + 15.0,
                )
                favorable = pair_avg_cost <= self.engine_pair.target_rebalance_pair_cost
                if not favorable and not near_end:
                    continue

                merge_size = pair_size if favorable else max(min_size, pair_size * 0.50)
                result = self.executor.merge_pairs(
                    market_id=snapshot.market.market_id,
                    primary_token_id=snapshot.market.primary_token_id,
                    secondary_token_id=snapshot.market.secondary_token_id,
                    size=merge_size,
                    condition_id=snapshot.market.condition_id,
                    is_neg_risk=snapshot.market.is_neg_risk,
                )
                if result.unsupported:
                    if not self._settlement_method_missing:
                        self.storage.record_risk_event(
                            "settlement_merge_unsupported",
                            {
                                "reason": "executor merge method unavailable",
                                "raw": result.raw,
                            },
                        )
                        LOGGER.warning(
                            "Settlement merge unsupported by executor; merge automation disabled"
                        )
                    self._settlement_method_missing = True
                    break
                if result.status in {"pending", "submitted"}:
                    self.storage.record_risk_event(
                        "settlement_merge_pending",
                        {
                            "market_id": snapshot.market.market_id,
                            "requested_size": merge_size,
                            "raw": result.raw,
                        },
                    )
                    continue
                if not result.success:
                    self.storage.record_risk_event(
                        "settlement_merge_error",
                        {
                            "market_id": snapshot.market.market_id,
                            "requested_size": merge_size,
                            "pair_avg_cost": pair_avg_cost,
                            "raw": result.raw,
                        },
                    )
                    continue

                applied = self.risk.apply_merge(
                    market_id=snapshot.market.market_id,
                    primary_token_id=snapshot.market.primary_token_id,
                    secondary_token_id=snapshot.market.secondary_token_id,
                    pair_size=max(0.0, result.merged_size or merge_size),
                    payout_per_pair=1.0,
                )
                if applied <= 0:
                    continue
                merged_total += applied
                self.storage.record_risk_event(
                    "settlement_merge",
                    {
                        "market_id": snapshot.market.market_id,
                        "pair_size": applied,
                        "pair_avg_cost": pair_avg_cost,
                        "seconds_to_end": snapshot.market.seconds_to_end,
                        "raw": result.raw,
                    },
                )
            self._last_settlement_ts = now_ts

        redeem_interval = 30.0
        if now_ts - self._last_redeem_ts >= redeem_interval:
            redeem = self.executor.redeem_all()
            self._last_redeem_ts = now_ts
            if redeem.unsupported:
                if not self._redeem_method_missing:
                    self.storage.record_risk_event(
                        "settlement_redeem_unsupported",
                        {
                            "reason": "executor redeem method unavailable",
                            "raw": redeem.raw,
                        },
                    )
                    LOGGER.warning(
                        "Settlement redeem unsupported by executor; redeem automation disabled"
                    )
                self._redeem_method_missing = True
            elif redeem.status in {"pending", "submitted"}:
                self.storage.record_risk_event(
                    "settlement_redeem_pending",
                    {"raw": redeem.raw},
                )
            elif redeem.status == "redeemed":
                redeem_calls += 1
                self.storage.record_risk_event(
                    "settlement_redeem",
                    {"redeemed_usdc": redeem.redeemed_usdc, "raw": redeem.raw},
                )
            elif redeem.status == "ok":
                pass
            else:
                self.storage.record_risk_event(
                    "settlement_redeem_error",
                    {"raw": redeem.raw},
                )

        return merged_total, redeem_calls

    def _preferred_execution_side(
        self,
        *,
        snapshot: MarketSnapshot,
        primary_inventory: float,
        secondary_inventory: float,
    ) -> str | None:
        gross = max(0.0, primary_inventory) + max(0.0, secondary_inventory)
        net = max(0.0, primary_inventory) - max(0.0, secondary_inventory)
        min_size = max(snapshot.market.order_min_size, 1.0)
        naked_ratio = abs(net) / max(min_size, gross)

        # Rebalance bias when one-sided risk gets too high.
        if gross > 0 and naked_ratio >= 0.34:
            if net > 0:
                return "secondary"
            if net < 0:
                return "primary"
            return None

        state = self._market_buy_exec_state.get(snapshot.market.market_id)
        if state is None:
            return None
        last_side = state.last_side
        if last_side not in {"primary", "secondary"}:
            return None
        run_len = state.run_len
        if run_len >= 8:
            return None

        if last_side == "primary":
            ask = snapshot.primary_book.best_ask
            last_price = state.last_primary_price
        else:
            ask = snapshot.secondary_book.best_ask
            last_price = state.last_secondary_price
        if ask <= 0 or last_price <= 0:
            return None

        # Continue same-side laddering while execution stays near the
        # previous fill price; avoid forced alternation every fill.
        if ask <= (last_price + 0.015) and naked_ratio <= 0.34:
            return last_side
        return None

    @staticmethod
    def _is_paired_ladder_intent(intent: OrderIntent) -> bool:
        intent_type = str(intent.metadata.get("intent_type") or "")
        if intent_type not in {"pair_entry_primary", "pair_completion"}:
            return False
        pair_group_id = str(intent.metadata.get("pair_group_id") or "").strip()
        return bool(pair_group_id)

    def _truncate_intents_preserving_pair_groups(
        self,
        intents: list[OrderIntent],
        max_per_cycle: int,
    ) -> list[OrderIntent]:
        if len(intents) <= max_per_cycle:
            return intents

        grouped_pair_legs: dict[tuple[str, str], list[OrderIntent]] = {}
        for intent in intents:
            if not self._is_paired_ladder_intent(intent):
                continue
            pair_group_id = str(intent.metadata.get("pair_group_id") or "").strip()
            group_key = (intent.market_id, pair_group_id)
            grouped_pair_legs.setdefault(group_key, []).append(intent)

        selected: list[OrderIntent] = []
        selected_groups: set[tuple[str, str]] = set()
        for intent in intents:
            if len(selected) >= max_per_cycle:
                break

            if not self._is_paired_ladder_intent(intent):
                selected.append(intent)
                continue

            pair_group_id = str(intent.metadata.get("pair_group_id") or "").strip()
            group_key = (intent.market_id, pair_group_id)
            if group_key in selected_groups:
                continue

            group_legs = grouped_pair_legs.get(group_key, [])
            # Only admit complete groups so truncation cannot orphan one side.
            if len(group_legs) != 2:
                continue
            if (len(selected) + len(group_legs)) > max_per_cycle:
                continue

            selected.extend(group_legs)
            selected_groups.add(group_key)

        return selected

    def _generate_intents(
        self,
        snapshots: list[MarketSnapshot],
        fair_by_market: dict[str, float],
    ) -> list[OrderIntent]:
        if self.risk.halted:
            return []
        if self._single_5m_pause_active():
            return []

        intents: list[OrderIntent] = []
        now_ts = time.time()
        rolling_pair_price_avg, rolling_pair_all_in_avg, rolling_pair_cost_samples = (
            self._pair_cost_governor_state()
        )
        for snapshot in snapshots:
            fair = fair_by_market.get(snapshot.market.market_id, 0.5)
            self._log_signal(snapshot, fair)

            primary_pos = self.risk.positions.get(snapshot.market.primary_token_id)
            secondary_pos = self.risk.positions.get(snapshot.market.secondary_token_id)
            primary_inventory = (
                primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
            )
            secondary_inventory = (
                secondary_pos.size if secondary_pos and secondary_pos.size > 0 else 0.0
            )
            primary_avg_entry = (
                primary_pos.average_price
                if primary_pos and primary_pos.size > 0
                else 0.0
            )
            secondary_avg_entry = (
                secondary_pos.average_price
                if secondary_pos and secondary_pos.size > 0
                else 0.0
            )
            preferred_entry_side = self._preferred_execution_side(
                snapshot=snapshot,
                primary_inventory=primary_inventory,
                secondary_inventory=secondary_inventory,
            )
            motion_window = self._motion_window_seconds(snapshot.market.timeframe)
            primary_motion = self.book_motion.summarize(
                token_id=snapshot.market.primary_token_id,
                now_ts=now_ts,
                window_seconds=motion_window,
            )
            secondary_motion = self.book_motion.summarize(
                token_id=snapshot.market.secondary_token_id,
                now_ts=now_ts,
                window_seconds=motion_window,
            )

            (
                learned_primary_pair_price,
                learned_primary_success_rate,
                learned_primary_samples,
            ) = self.pair_learner.estimate(
                timeframe=snapshot.market.timeframe,
                side="primary",
                seconds_to_end=snapshot.market.seconds_to_end,
            )
            (
                learned_secondary_pair_price,
                learned_secondary_success_rate,
                learned_secondary_samples,
            ) = self.pair_learner.estimate(
                timeframe=snapshot.market.timeframe,
                side="secondary",
                seconds_to_end=snapshot.market.seconds_to_end,
            )
            market_intents = self.engine_pair.generate(
                snapshot=snapshot,
                fair_probability=fair,
                primary_inventory=primary_inventory,
                secondary_inventory=secondary_inventory,
                primary_avg_entry=primary_avg_entry,
                secondary_avg_entry=secondary_avg_entry,
                primary_motion=primary_motion,
                secondary_motion=secondary_motion,
                preferred_entry_side=preferred_entry_side,
                learned_primary_pair_price=learned_primary_pair_price,
                learned_primary_success_rate=learned_primary_success_rate,
                learned_primary_samples=learned_primary_samples,
                learned_secondary_pair_price=learned_secondary_pair_price,
                learned_secondary_success_rate=learned_secondary_success_rate,
                learned_secondary_samples=learned_secondary_samples,
                rolling_pair_cost_avg=rolling_pair_price_avg,
                rolling_pair_cost_samples=rolling_pair_cost_samples,
            )
            if market_intents:
                self._last_pair_idle_diag.pop(snapshot.market.market_id, None)
            else:
                idle_diag = self._pair_idle_diagnostics(
                    snapshot=snapshot,
                    fair_probability=fair,
                    primary_inventory=primary_inventory,
                    secondary_inventory=secondary_inventory,
                    rolling_pair_cost_avg=rolling_pair_price_avg,
                    rolling_pair_cost_samples=rolling_pair_cost_samples,
                )
                idle_reason = str(idle_diag.get("reason", "unknown"))
                secs_left = max(0.0, float(idle_diag.get("seconds_to_end", 0.0)))
                pair_all_in = max(0.0, float(idle_diag.get("pair_cost_all_in", 0.0)))
                idle_signature = (
                    idle_reason,
                    int(secs_left // 5.0),
                    int(round(pair_all_in * 1000.0)),
                )
                market_id = snapshot.market.market_id
                previous_idle = self._last_pair_idle_diag.get(market_id)
                if idle_signature != previous_idle or (self._cycle_counter % 15 == 0):
                    LOGGER.info(
                        "pair_arb idle market=%s reason=%s secs_left=%.0f top_all_in=%.4f entry_cap=%.4f alpha_guard=%.0f time_guard=%.0f gross_inv=%.2f naked=%.3f",
                        market_id,
                        idle_reason,
                        secs_left,
                        pair_all_in,
                        max(0.0, float(idle_diag.get("pair_cap", 0.0))),
                        max(0.0, float(idle_diag.get("alpha_guard_seconds", 0.0))),
                        max(0.0, float(idle_diag.get("time_guard_seconds", 0.0))),
                        max(0.0, float(idle_diag.get("gross_inventory", 0.0))),
                        max(0.0, float(idle_diag.get("naked_ratio", 0.0))),
                    )
                    self._last_pair_idle_diag[market_id] = idle_signature
            intents.extend(market_intents)

        # Sells are intentionally disabled in current run profile.
        # Keep sell wrappers in execution/risk code, but do not emit sell intents.
        intents = [intent for intent in intents if intent.side == Side.BUY]

        intents.sort(key=lambda i: i.expected_edge, reverse=True)
        max_per_cycle = max(4, int(self.config.pair_max_intents_per_cycle))
        if self.config.bankroll_usdc >= 150.0:
            max_per_cycle = max(max_per_cycle, 30)
        elif self.config.bankroll_usdc >= 100.0:
            max_per_cycle = max(max_per_cycle, 24)
        if self.config.single_5m_deep_mode:
            if self.config.bankroll_usdc >= 150.0:
                max_per_cycle = max(max_per_cycle, 64)
            elif self.config.bankroll_usdc >= 100.0:
                max_per_cycle = max(max_per_cycle, 48)
            else:
                max_per_cycle = max(max_per_cycle, 32)
        cycle_volatility = 0.0
        for intent in intents:
            try:
                score = float(intent.metadata.get("fluctuation_volatility_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            cycle_volatility = max(cycle_volatility, score)
        if cycle_volatility >= 0.85:
            max_per_cycle = max(max_per_cycle, 96 if self.config.single_5m_deep_mode else 56)
        elif cycle_volatility >= 0.70:
            max_per_cycle = max(max_per_cycle, 80 if self.config.single_5m_deep_mode else 48)
        elif cycle_volatility >= 0.55:
            max_per_cycle = max(max_per_cycle, 68 if self.config.single_5m_deep_mode else 42)
        intents = self._truncate_intents_preserving_pair_groups(intents, max_per_cycle)
        by_type = Counter(
            str(intent.metadata.get("intent_type") or "unknown") for intent in intents
        )
        pair_price_avg_text = (
            f"{rolling_pair_price_avg:.4f}"
            if rolling_pair_price_avg is not None
            else "na"
        )
        pair_all_in_avg_text = (
            f"{rolling_pair_all_in_avg:.4f}"
            if rolling_pair_all_in_avg is not None
            else "na"
        )
        summary = (
            len(intents),
            tuple(sorted((str(k), int(v)) for k, v in by_type.items())),
            pair_price_avg_text,
            pair_all_in_avg_text,
            rolling_pair_cost_samples,
        )
        should_log = (
            summary != self._last_pair_arb_summary
            or len(intents) > 0
            or (self._cycle_counter % 15 == 0)
        )
        if should_log:
            LOGGER.info(
                "pair_arb intents=%s breakdown=%s pair_price_avg=%s pair_all_in_avg=%s pair_cost_samples=%s",
                len(intents),
                dict(by_type),
                pair_price_avg_text,
                pair_all_in_avg_text,
                rolling_pair_cost_samples,
            )
            self._last_pair_arb_summary = summary
        return intents

    def _pair_idle_diagnostics(
        self,
        *,
        snapshot: MarketSnapshot,
        fair_probability: float,
        primary_inventory: float,
        secondary_inventory: float,
        rolling_pair_cost_avg: float | None,
        rolling_pair_cost_samples: int,
    ) -> dict[str, float | str]:
        market = snapshot.market
        timeframe = market.timeframe
        seconds_to_end = max(0.0, float(market.seconds_to_end))
        phase = self.engine_pair._execution_phase(
            timeframe=timeframe,
            now=datetime.now(tz=timezone.utc),
            start_time=market.start_time,
            seconds_to_end=seconds_to_end,
        )

        deep_mode_5m = self.config.single_5m_deep_mode and timeframe == Timeframe.FIVE_MIN
        ask_up = float(snapshot.primary_book.best_ask)
        ask_down = float(snapshot.secondary_book.best_ask)
        gross_inventory = max(0.0, primary_inventory) + max(0.0, secondary_inventory)
        net_delta = max(0.0, primary_inventory) - max(0.0, secondary_inventory)
        naked_ratio = abs(net_delta) / max(snapshot.market.order_min_size, gross_inventory)

        effective_target_ratio = clamp(self.engine_pair.target_naked_ratio, 0.08, 0.30)
        if gross_inventory <= (snapshot.market.order_min_size * 1.5):
            effective_target_ratio = min(effective_target_ratio, 0.10)
        effective_hard_ratio = max(
            self.engine_pair.hard_naked_ratio,
            min(0.60, effective_target_ratio + 0.14),
        )
        if deep_mode_5m:
            effective_target_ratio = 0.06
            effective_hard_ratio = 0.14

        time_guard_seconds = float(self.engine_pair._time_guard_seconds(timeframe))
        alpha_guard_seconds = float(self.engine_pair._alpha_entry_guard_seconds(timeframe))
        inside_time_guard = seconds_to_end <= time_guard_seconds
        inside_alpha_guard = seconds_to_end <= alpha_guard_seconds

        if ask_up <= 0 or ask_down <= 0:
            return {
                "reason": "missing_ask",
                "seconds_to_end": seconds_to_end,
                "pair_cost_all_in": 0.0,
                "pair_cap": 0.0,
                "alpha_guard_seconds": alpha_guard_seconds,
                "time_guard_seconds": time_guard_seconds,
                "gross_inventory": gross_inventory,
                "naked_ratio": naked_ratio,
            }

        fee_up = per_share_fee(ask_up, snapshot.primary_fee.base_fee)
        fee_down = per_share_fee(ask_down, snapshot.secondary_fee.base_fee)
        pair_cost = ask_up + ask_down
        pair_cost_all_in = pair_cost + fee_up + fee_down + (
            2.0 * self.config.directional_slippage_buffer
        )

        pair_entry_cap = self.engine_pair._pair_entry_cost_cap(timeframe, phase)
        pair_governor_cap = self.engine_pair._pair_cost_governor_cap(
            timeframe=timeframe,
            phase=phase,
            rolling_pair_cost_avg=rolling_pair_cost_avg,
            rolling_pair_cost_samples=rolling_pair_cost_samples,
        )
        if deep_mode_5m:
            pair_governor_cap = min(
                pair_governor_cap,
                self.engine_pair._deep_mode_governor_ceiling(rolling_pair_cost_avg),
            )
            pair_entry_cap = self.engine_pair._deep_mode_pair_entry_cap(
                pair_entry_cap=pair_entry_cap,
                governor_cap=pair_governor_cap,
            )
        pair_cap = min(
            pair_entry_cap,
            pair_governor_cap,
            self.engine_pair.absurd_pair_cost_guard,
        )

        reason = "no_candidate_or_alpha_filter"
        if inside_time_guard and gross_inventory <= 0:
            reason = "time_guard_block"
        elif inside_time_guard:
            reason = "pair_guard_block"
        elif inside_alpha_guard and gross_inventory <= 0:
            reason = "alpha_guard_block"
        elif pair_cost_all_in >= self.engine_pair.absurd_pair_cost_guard:
            reason = "absurd_pair_cost_guard"
        elif deep_mode_5m and gross_inventory > 0 and naked_ratio >= effective_hard_ratio:
            reason = "deep_mode_skew_halt"

        fair_up = clamp(fair_probability, 0.01, 0.99)
        fair_down = clamp(1.0 - fair_up, 0.01, 0.99)
        edge_up = fair_up - ask_up - fee_up - self.config.directional_slippage_buffer
        edge_down = fair_down - ask_down - fee_down - self.config.directional_slippage_buffer

        return {
            "reason": reason,
            "seconds_to_end": seconds_to_end,
            "pair_cost_all_in": pair_cost_all_in,
            "pair_cap": pair_cap,
            "alpha_guard_seconds": alpha_guard_seconds,
            "time_guard_seconds": time_guard_seconds,
            "gross_inventory": gross_inventory,
            "naked_ratio": naked_ratio,
            "edge_up": edge_up,
            "edge_down": edge_down,
        }

    def _generate_recycle_sell_intents(
        self, snapshots: list[MarketSnapshot]
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for snapshot in snapshots:
            primary_pos = self.risk.positions.get(snapshot.market.primary_token_id)
            secondary_pos = self.risk.positions.get(snapshot.market.secondary_token_id)
            primary_size = (
                primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
            )
            secondary_size = (
                secondary_pos.size if secondary_pos and secondary_pos.size > 0 else 0.0
            )
            pair_size = min(primary_size, secondary_size)
            min_size = snapshot.market.order_min_size
            if pair_size < min_size:
                continue
            if (
                snapshot.primary_book.best_bid <= 0
                or snapshot.secondary_book.best_bid <= 0
            ):
                continue

            primary_avg = primary_pos.average_price if primary_pos else 0.0
            secondary_avg = secondary_pos.average_price if secondary_pos else 0.0
            pair_avg_cost = primary_avg + secondary_avg
            pair_bid_value = (
                snapshot.primary_book.best_bid + snapshot.secondary_book.best_bid
            )
            # Recycle completed pairs only when exit bids clear our average cost.
            if pair_bid_value < (pair_avg_cost + 0.004):
                continue

            recycle_size = min(pair_size, max(min_size, pair_size * 0.35))
            if recycle_size < min_size:
                continue

            primary_price = clamp(
                round(snapshot.primary_book.best_bid - 0.01, 2),
                0.01,
                snapshot.primary_book.best_bid,
            )
            secondary_price = clamp(
                round(snapshot.secondary_book.best_bid - 0.01, 2),
                0.01,
                snapshot.secondary_book.best_bid,
            )
            intents.append(
                OrderIntent(
                    market_id=snapshot.market.market_id,
                    token_id=snapshot.market.primary_token_id,
                    side=Side.SELL,
                    price=primary_price,
                    size=recycle_size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_pair_arb",
                    expected_edge=120.0,
                    metadata={
                        "strategy": "pair-arb",
                        "intent_type": "pair_recycle_sell",
                        "picked_side": "primary",
                        "order_min_size": min_size,
                        "pair_avg_cost": pair_avg_cost,
                        "pair_bid_value": pair_bid_value,
                        "recycle_size": recycle_size,
                    },
                )
            )
            intents.append(
                OrderIntent(
                    market_id=snapshot.market.market_id,
                    token_id=snapshot.market.secondary_token_id,
                    side=Side.SELL,
                    price=secondary_price,
                    size=recycle_size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_pair_arb",
                    expected_edge=120.0,
                    metadata={
                        "strategy": "pair-arb",
                        "intent_type": "pair_recycle_sell",
                        "picked_side": "secondary",
                        "order_min_size": min_size,
                        "pair_avg_cost": pair_avg_cost,
                        "pair_bid_value": pair_bid_value,
                        "recycle_size": recycle_size,
                    },
                )
            )
        return intents

    def _generate_unwind_intents(
        self, snapshots: list[MarketSnapshot]
    ) -> list[OrderIntent]:
        by_token: dict[str, tuple[MarketSnapshot, OrderBookSnapshot, str]] = {}
        for snapshot in snapshots:
            by_token[snapshot.market.primary_token_id] = (
                snapshot,
                snapshot.primary_book,
                "primary",
            )
            by_token[snapshot.market.secondary_token_id] = (
                snapshot,
                snapshot.secondary_book,
                "secondary",
            )

        intents: list[OrderIntent] = []
        halt_reason = self.risk.halted_reason.lower()
        daily_dd_halt = self.risk.halted and (
            "daily drawdown threshold exceeded" in halt_reason
        )
        market_cap = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        for token_id, position in self.risk.positions.items():
            if position.size <= 0:
                continue
            token_info = by_token.get(token_id)
            if token_info is None:
                continue
            snapshot, book, side_label = token_info
            if book.best_bid <= 0:
                continue

            near_end = False
            if snapshot.market.timeframe == Timeframe.FIVE_MIN:
                near_end = snapshot.market.seconds_to_end <= 40
            elif snapshot.market.timeframe == Timeframe.FIFTEEN_MIN:
                near_end = snapshot.market.seconds_to_end <= 100
            elif snapshot.market.timeframe == Timeframe.ONE_HOUR:
                near_end = snapshot.market.seconds_to_end <= 240

            if not self.risk.halted and not near_end:
                continue

            market_exposure = self.risk.market_exposure.get(
                snapshot.market.market_id, 0.0
            )
            if (
                daily_dd_halt
                and not near_end
                and market_exposure <= (market_cap * 0.35)
            ):
                continue

            if self.risk.halted:
                if daily_dd_halt:
                    size = min(
                        position.size,
                        max(snapshot.market.order_min_size, position.size * 0.25),
                    )
                    intent_type = "drawdown_rebalance"
                    expected_edge = 5_000.0
                else:
                    size = position.size
                    intent_type = "risk_unwind"
                    expected_edge = 10_000.0
            else:
                size = min(
                    position.size,
                    max(snapshot.market.order_min_size, position.size * 0.60),
                )
                intent_type = "expiry_unwind"
                expected_edge = 1_000.0

            if size < snapshot.market.order_min_size:
                continue

            intents.append(
                OrderIntent(
                    market_id=snapshot.market.market_id,
                    token_id=token_id,
                    side=Side.SELL,
                    price=clamp(round(book.best_bid - 0.01, 2), 0.01, 0.99),
                    size=size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_pair_arb",
                    expected_edge=expected_edge,
                    metadata={
                        "strategy": "pair-arb",
                        "intent_type": intent_type,
                        "picked_side": side_label,
                        "order_min_size": snapshot.market.order_min_size,
                        "seconds_to_end": snapshot.market.seconds_to_end,
                    },
                )
            )
        return intents

    def _log_signal(self, snapshot: MarketSnapshot, fair_probability: float) -> None:
        up_ask = snapshot.primary_book.best_ask
        down_ask = snapshot.secondary_book.best_ask
        up_edge = (
            fair_probability
            - up_ask
            - per_share_fee(up_ask, snapshot.primary_fee.base_fee)
            - self.config.directional_slippage_buffer
            if up_ask > 0
            else 0.0
        )
        down_fair = 1.0 - fair_probability
        down_edge = (
            down_fair
            - down_ask
            - per_share_fee(down_ask, snapshot.secondary_fee.base_fee)
            - self.config.directional_slippage_buffer
            if down_ask > 0
            else 0.0
        )
        LOGGER.debug(
            "signal market=%s tf=%s sec_to_end=%.0f fair_up=%.3f up_ask=%.3f up_edge=%.4f down_ask=%.3f down_edge=%.4f",
            snapshot.market.market_id,
            snapshot.market.timeframe.value,
            snapshot.market.seconds_to_end,
            fair_probability,
            up_ask,
            up_edge,
            down_ask,
            down_edge,
        )

    @staticmethod
    def _parse_timeframe(raw: object) -> Timeframe:
        if isinstance(raw, Timeframe):
            return raw
        value = str(raw or "").strip().lower()
        for timeframe in (
            Timeframe.FIVE_MIN,
            Timeframe.FIFTEEN_MIN,
            Timeframe.ONE_HOUR,
        ):
            if value == timeframe.value:
                return timeframe
        return Timeframe.UNKNOWN

    @staticmethod
    def _hedge_timeout_seconds(timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            return 14.0
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 24.0
        if timeframe == Timeframe.ONE_HOUR:
            return 50.0
        return 20.0

    def _record_alpha_fill_for_learning(
        self, intent: OrderIntent, result: OrderResult, now_ts: float
    ) -> None:
        if result.side != Side.BUY or result.filled_size <= 0:
            return
        intent_type = str(intent.metadata.get("intent_type") or "")
        if intent_type != "alpha_entry":
            return
        opposite_token_id = str(intent.metadata.get("opposite_token_id") or "").strip()
        entry_side = str(intent.metadata.get("picked_side") or "").strip().lower()
        timeframe = self._parse_timeframe(intent.metadata.get("timeframe"))
        seconds_to_end_raw = intent.metadata.get("seconds_to_end")
        try:
            seconds_to_end = float(seconds_to_end_raw)
        except (TypeError, ValueError):
            seconds_to_end = 0.0
        if (
            not opposite_token_id
            or timeframe == Timeframe.UNKNOWN
            or entry_side not in {"primary", "secondary"}
        ):
            return
        leg = AlphaOpenLeg(
            market_id=result.market_id,
            entry_token_id=result.token_id,
            opposite_token_id=opposite_token_id,
            entry_side=entry_side,
            timeframe=timeframe.value,
            seconds_to_end_at_entry=max(0.0, seconds_to_end),
            entry_price=max(0.0, float(result.filled_price)),
            remaining_size=max(0.0, float(result.filled_size)),
            opened_ts=max(0.0, float(now_ts)),
        )
        self._open_alpha_legs.setdefault(result.market_id, []).append(leg)

        state = self._ensure_market_buy_exec_state(result.market_id)
        last_side = str(state.last_side or "")
        run_len = int(state.run_len or 0)
        if last_side == entry_side:
            run_len += 1
        else:
            run_len = 1
        state.last_side = entry_side
        state.run_len = run_len
        if entry_side == "primary":
            state.last_primary_price = max(0.0, float(result.filled_price))
        else:
            state.last_secondary_price = max(0.0, float(result.filled_price))
        state.updated_ts = max(0.0, float(now_ts))

    def _match_hedge_fill_for_learning(
        self, intent: OrderIntent, result: OrderResult, now_ts: float
    ) -> None:
        if result.side != Side.BUY or result.filled_size <= 0:
            return
        intent_type = str(intent.metadata.get("intent_type") or "")
        if intent_type not in {"equalize", "equalize_forced", "equalize_immediate"}:
            return

        legs = self._open_alpha_legs.get(result.market_id)
        if not legs:
            return
        remaining = float(result.filled_size)
        updated: list[AlphaOpenLeg] = []
        for leg in sorted(legs, key=lambda item: item.opened_ts):
            leg_remaining = float(leg.remaining_size)
            if leg_remaining <= 0:
                continue
            if remaining <= 0:
                updated.append(leg)
                continue
            opposite_token_id = str(leg.opposite_token_id or "")
            if opposite_token_id != result.token_id:
                updated.append(leg)
                continue

            matched = min(remaining, leg_remaining)
            pair_price_cost = float(leg.entry_price) + float(result.filled_price)
            delay = max(0.0, now_ts - float(leg.opened_ts))
            timeframe = self._parse_timeframe(leg.timeframe)
            entry_side = str(leg.entry_side or "")
            seconds_to_end = float(leg.seconds_to_end_at_entry)
            success = pair_price_cost <= 1.005
            self.pair_learner.observe(
                market_id=result.market_id,
                timeframe=timeframe,
                side=entry_side,
                seconds_to_end=seconds_to_end,
                pair_price_cost=pair_price_cost,
                hedge_delay_seconds=delay,
                success=success,
                source="bot_fill",
            )

            leg_remaining -= matched
            remaining -= matched
            if leg_remaining > 1e-9:
                leg.remaining_size = leg_remaining
                updated.append(leg)
        if updated:
            self._open_alpha_legs[result.market_id] = updated
        else:
            self._open_alpha_legs.pop(result.market_id, None)

    def _expire_unhedged_alpha_legs(
        self, books: dict[str, OrderBookSnapshot], now_ts: float
    ) -> None:
        for market_id, legs in list(self._open_alpha_legs.items()):
            updated: list[AlphaOpenLeg] = []
            for leg in legs:
                leg_remaining = float(leg.remaining_size)
                if leg_remaining <= 1e-9:
                    continue
                timeframe = self._parse_timeframe(leg.timeframe)
                opened_ts = float(leg.opened_ts if leg.opened_ts > 0 else now_ts)
                age = max(0.0, now_ts - opened_ts)
                timeout = self._hedge_timeout_seconds(timeframe)
                if age < timeout:
                    updated.append(leg)
                    continue

                opposite_token_id = str(leg.opposite_token_id or "")
                opposite_book = books.get(opposite_token_id)
                opposite_ask = (
                    opposite_book.best_ask
                    if opposite_book and opposite_book.best_ask > 0
                    else 0.99
                )
                pair_price_cost = float(leg.entry_price) + float(opposite_ask)
                entry_side = str(leg.entry_side or "")
                seconds_to_end = float(leg.seconds_to_end_at_entry)
                self.pair_learner.observe(
                    market_id=market_id,
                    timeframe=timeframe,
                    side=entry_side,
                    seconds_to_end=seconds_to_end,
                    pair_price_cost=pair_price_cost,
                    hedge_delay_seconds=age,
                    success=False,
                    source="stale_timeout",
                )
            if updated:
                self._open_alpha_legs[market_id] = updated
            else:
                self._open_alpha_legs.pop(market_id, None)

    def _ensure_pair_cost_trackers(self) -> None:
        if not hasattr(self, "_pair_cost_history"):
            self._pair_cost_history = deque(maxlen=320)
        if not hasattr(self, "_pair_cost_open_legs"):
            self._pair_cost_open_legs = {}

    def _record_pair_cost_fill(self, intent: OrderIntent, result: OrderResult) -> None:
        if intent.engine != "engine_pair_arb":
            return
        if result.side != Side.BUY or result.filled_size <= 0:
            return
        picked_side = str(intent.metadata.get("picked_side") or "").strip().lower()
        if picked_side not in {"primary", "secondary"}:
            return
        try:
            fee_bps = int(intent.metadata.get("fee_bps", 0))
        except (TypeError, ValueError):
            fee_bps = 0
        fill_size = max(0.0, float(result.filled_size))
        fill_price = max(0.0, float(result.filled_price))
        if fill_price <= 0.0:
            fill_price = max(0.0, float(intent.price))
        if fill_price <= 0:
            return
        # Track realized pair price separately from all-in (price + realized fee).
        leg_price_cost = fill_price
        realized_fee_per_share = 0.0
        if fill_size > 1e-9:
            realized_fee_per_share = max(0.0, float(result.fee_paid)) / fill_size
        if realized_fee_per_share <= 0.0:
            realized_fee_per_share = per_share_fee(fill_price, fee_bps)
        leg_all_in_cost = leg_price_cost + realized_fee_per_share

        self._ensure_pair_cost_trackers()
        by_market = self._pair_cost_open_legs.setdefault(
            result.market_id,
            {"primary": deque(), "secondary": deque()},
        )
        if picked_side not in by_market:
            by_market[picked_side] = deque()
        opposite_side = "secondary" if picked_side == "primary" else "primary"
        if opposite_side not in by_market:
            by_market[opposite_side] = deque()
        own_queue = by_market[picked_side]
        opposite_queue = by_market[opposite_side]

        remaining = fill_size
        while remaining > 1e-9 and opposite_queue:
            opp_raw = opposite_queue[0]
            if len(opp_raw) >= 3:
                opp_size, opp_price_cost, opp_all_in_cost = opp_raw
            else:
                opp_size, opp_price_cost = (
                    opp_raw  # pragma: no cover - backward compatibility
                )
                opp_all_in_cost = opp_price_cost
            matched = min(remaining, float(opp_size))
            if matched <= 1e-9:
                opposite_queue.popleft()
                continue
            self._pair_cost_history.append(
                (
                    leg_price_cost + float(opp_price_cost),
                    leg_all_in_cost + float(opp_all_in_cost),
                    matched,
                )
            )
            remaining -= matched
            opp_left = float(opp_size) - matched
            if opp_left <= 1e-9:
                opposite_queue.popleft()
            else:
                opposite_queue[0] = (
                    opp_left,
                    float(opp_price_cost),
                    float(opp_all_in_cost),
                )

        if remaining > 1e-9:
            own_queue.append((remaining, leg_price_cost, leg_all_in_cost))

        if not by_market.get("primary") and not by_market.get("secondary"):
            self._pair_cost_open_legs.pop(result.market_id, None)

    def _pair_cost_governor_state(self) -> tuple[float | None, float | None, int]:
        self._ensure_pair_cost_trackers()
        total_weight = 0.0
        weighted_price_cost = 0.0
        weighted_all_in_cost = 0.0
        sample_count = 0
        for history_entry in self._pair_cost_history:
            if len(history_entry) >= 3:
                pair_price_cost, pair_all_in_cost, weight = history_entry
            else:
                pair_price_cost, weight = (
                    history_entry  # pragma: no cover - backward compatibility
                )
                pair_all_in_cost = pair_price_cost
            safe_weight = max(0.0, float(weight))
            if safe_weight <= 0:
                continue
            sample_count += 1
            total_weight += safe_weight
            weighted_price_cost += float(pair_price_cost) * safe_weight
            weighted_all_in_cost += float(pair_all_in_cost) * safe_weight
        if total_weight <= 0:
            return None, None, 0
        return (
            weighted_price_cost / total_weight,
            weighted_all_in_cost / total_weight,
            sample_count,
        )

    def _build_immediate_equalizer(
        self,
        *,
        parent_intent: OrderIntent,
        fill: OrderResult,
        books: dict[str, OrderBookSnapshot],
    ) -> OrderIntent | None:
        if parent_intent.engine != "engine_pair_arb":
            return None
        parent_intent_type = str(parent_intent.metadata.get("intent_type") or "")
        if parent_intent_type not in {
            "alpha_entry",
            "pair_entry_primary",
            "pair_completion",
        }:
            return None
        if fill.side != Side.BUY or fill.filled_size <= 0:
            return None

        opposite_token_id = str(
            parent_intent.metadata.get("opposite_token_id") or ""
        ).strip()
        if not opposite_token_id or opposite_token_id == fill.token_id:
            return None

        opposite_book = books.get(opposite_token_id)
        if opposite_book is None or opposite_book.best_ask <= 0:
            return None

        try:
            min_size = float(parent_intent.metadata.get("order_min_size", 5.0))
        except (TypeError, ValueError):
            min_size = 5.0
        hedge_size = max(0.0, fill.filled_size)
        if hedge_size < min_size:
            return None

        opposite_ask = float(opposite_book.best_ask)
        taker_hedge_entry = round_tick(min(0.99, opposite_ask))
        maker_hedge_entry = self.engine_pair._maker_entry_from_book(opposite_book)
        if taker_hedge_entry <= 0:
            return None

        fee_bps_raw = parent_intent.metadata.get("opposite_fee_bps", 0)
        try:
            fee_bps = int(fee_bps_raw)
        except (TypeError, ValueError):
            fee_bps = 0

        primary_pos = self.risk.positions.get(fill.token_id)
        opposite_pos = self.risk.positions.get(opposite_token_id)
        primary_size = primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
        opposite_size = (
            opposite_pos.size if opposite_pos and opposite_pos.size > 0 else 0.0
        )
        gross_inventory = primary_size + opposite_size
        naked_ratio = abs(primary_size - opposite_size) / max(min_size, gross_inventory)
        must_reduce_naked = (
            gross_inventory >= (min_size * 3.0)
            and naked_ratio >= self.engine_pair.hard_naked_ratio
        )
        timeframe = self._parse_timeframe(parent_intent.metadata.get("timeframe"))
        try:
            seconds_to_end = float(parent_intent.metadata.get("seconds_to_end", 0.0))
        except (TypeError, ValueError):
            seconds_to_end = 0.0
        force_window = self.engine_pair._force_equalizer_seconds(timeframe)
        near_force_window = seconds_to_end > 0 and seconds_to_end <= force_window
        is_pair_leg_fill = parent_intent_type in {
            "pair_entry_primary",
            "pair_completion",
        }
        use_resting_hedge = (
            (not must_reduce_naked)
            and (not near_force_window)
            and maker_hedge_entry > 0
        )
        if is_pair_leg_fill:
            # Pair-leg fills should be balanced immediately; do not leave a new
            # one-sided leg resting for later.
            use_resting_hedge = False
        hedge_entry = maker_hedge_entry if use_resting_hedge else taker_hedge_entry
        hedge_tif = TimeInForce.GTC if use_resting_hedge else TimeInForce.IOC
        hedge_post_only = use_resting_hedge
        hedge_execution_style = (
            "resting_maker_equalize" if use_resting_hedge else "taker_ioc_equalize"
        )

        parent_fair = parent_intent.metadata.get("p_fair")
        if isinstance(parent_fair, (int, float)):
            opposite_fair = clamp(1.0 - float(parent_fair), 0.01, 0.99)
        else:
            opposite_fair = 0.5
        hedge_edge = (
            opposite_fair
            - hedge_entry
            - per_share_fee(hedge_entry, fee_bps)
            - self.config.directional_slippage_buffer
        )
        try:
            parent_fee_bps = int(parent_intent.metadata.get("fee_bps", 0))
        except (TypeError, ValueError):
            parent_fee_bps = 0
        projected_pair_cost = (
            fill.filled_price
            + hedge_entry
            + per_share_fee(fill.filled_price, parent_fee_bps)
            + per_share_fee(hedge_entry, fee_bps)
            + (2.0 * self.config.directional_slippage_buffer)
        )
        projected_pair_cost_price = (
            fill.filled_price
            + hedge_entry
            + (2.0 * self.config.directional_slippage_buffer)
        )
        completion_target = self.engine_pair.target_rebalance_pair_cost
        phase = str(parent_intent.metadata.get("phase") or "middle")
        rolling_pair_price_avg, _, rolling_pair_samples = (
            self._pair_cost_governor_state()
        )
        immediate_pair_cost_cap = self.engine_pair._pair_cost_governor_cap(
            timeframe=timeframe,
            phase=phase,
            rolling_pair_cost_avg=rolling_pair_price_avg,
            rolling_pair_cost_samples=rolling_pair_samples,
        )
        if self.config.single_5m_deep_mode and timeframe == Timeframe.FIVE_MIN:
            immediate_pair_cost_cap = min(
                immediate_pair_cost_cap,
                self.engine_pair._deep_mode_governor_ceiling(rolling_pair_price_avg),
            )

        # Micro-bankroll mode should not spend scarce cash on immediate hedges
        # above target pair cost unless we are close to forced close-out time.
        if (
            self.config.bankroll_usdc <= 50.0
            and (not near_force_window)
            and projected_pair_cost_price > completion_target
        ):
            return None

        # Match the trader-style clustered entries: avoid forced immediate
        # alternation unless pair completion is cheap or naked risk is high.
        immediate_price_gate = completion_target + 0.008
        if (
            (not is_pair_leg_fill)
            and (not must_reduce_naked)
            and projected_pair_cost_price > immediate_price_gate
        ):
            return None

        if is_pair_leg_fill:
            # Pair-leg fills should be hedged at full size to keep inventory
            # balanced; under-hedging here can leave large one-sided risk.
            ladder_clip = 1.00
        elif projected_pair_cost_price <= completion_target:
            ladder_clip = 0.75
        elif projected_pair_cost_price <= completion_target + 0.004:
            ladder_clip = 0.55
        elif projected_pair_cost_price <= completion_target + 0.008:
            ladder_clip = 0.40
        else:
            ladder_clip = 0.55 if must_reduce_naked else 0.30
        if must_reduce_naked:
            ladder_clip = max(0.65, ladder_clip)
        if self.config.bankroll_usdc <= 100.0:
            ladder_clip = max(0.35, ladder_clip)
        hedge_size = min(
            max(min_size, fill.filled_size * ladder_clip), fill.filled_size
        )
        # Keep immediate hedge clips human-readable (whole-share style when
        # possible) instead of odd fractional tails from partial fills.
        hedge_lot_step = max(min_size, 1.0)
        hedge_steps = int(max(0.0, hedge_size) / hedge_lot_step)
        if hedge_steps > 0:
            hedge_size = hedge_steps * hedge_lot_step
        else:
            hedge_size = min_size
        hedge_size = min(fill.filled_size, hedge_size)
        if hedge_size < min_size:
            return None

        picked_side = str(parent_intent.metadata.get("picked_side") or "")
        opposite_label = "secondary" if picked_side == "primary" else "primary"
        return OrderIntent(
            market_id=parent_intent.market_id,
            token_id=opposite_token_id,
            side=Side.BUY,
            price=hedge_entry,
            size=hedge_size,
            tif=hedge_tif,
            post_only=hedge_post_only,
            engine="engine_pair_arb",
            expected_edge=60.0,
            metadata={
                "strategy": "pair-arb",
                "intent_type": "equalize_immediate",
                "picked_side": opposite_label,
                "p_fair": opposite_fair,
                "edge_net": hedge_edge,
                "best_ask": opposite_ask,
                "maker_entry": maker_hedge_entry,
                "taker_entry": taker_hedge_entry,
                "execution_style": hedge_execution_style,
                "fee_bps": fee_bps,
                "order_min_size": min_size,
                "parent_order_id": fill.order_id,
                "parent_token_id": fill.token_id,
                "parent_intent_type": parent_intent_type,
                "parent_fill_size": fill.filled_size,
                "ladder_clip": ladder_clip,
                "must_reduce_naked": must_reduce_naked,
                "naked_ratio": naked_ratio,
                "marginal_pair_cost": projected_pair_cost,
                "marginal_pair_cost_price": projected_pair_cost_price,
                "target_pair_cost": completion_target,
                "immediate_pair_cost_cap": immediate_pair_cost_cap,
                "timeframe": timeframe.value,
                "seconds_to_end": seconds_to_end,
                "phase": phase,
                "quote_refresh_seconds": self.engine_pair._quote_refresh_seconds(
                    timeframe
                ),
                "quote_max_age_seconds": self.engine_pair._quote_max_age_seconds(
                    timeframe
                ),
                "quote_level_id": f"equalize_immediate_{opposite_label}_l1",
            },
        )

    def _execute_intents(
        self,
        intents: list[OrderIntent],
        books: dict[str, OrderBookSnapshot],
        *,
        prune_unplanned_quotes: bool = True,
    ) -> tuple[int, int, int]:
        executed = 0
        fills = 0
        errors = 0
        planned_ioc_total = 0.0
        planned_ioc_market: dict[str, float] = {}
        pending_intents = list(intents)
        if prune_unplanned_quotes:
            desired_quote_keys = {
                self._quote_key(intent)
                for intent in pending_intents
                if intent.post_only
            }
            no_fresh_quote_plan = not desired_quote_keys
            now_ts = time.time()
            preserve_existing_quotes = (
                self.config.single_5m_deep_mode and no_fresh_quote_plan
            )
            for key, order_id in list(self.quote_order_ids.items()):
                if key in desired_quote_keys:
                    continue
                if preserve_existing_quotes:
                    # In single-5m deep mode, avoid nuking the resting queue
                    # just because one cycle produced zero fresh intents.
                    continue
                if no_fresh_quote_plan and self._should_preserve_quote_on_empty_intent_cycle(
                    key=key,
                    order_id=order_id,
                    now_ts=now_ts,
                ):
                    # Briefly preserve 5m quotes when one cycle emits zero
                    # intents, so we do not churn queue priority.
                    continue
                self.executor.cancel_order(order_id)
                self._clear_quote_slot(key, order_id=order_id, keep_reconcile=True)
        intents_by_engine: Counter[str] = Counter(i.engine for i in pending_intents)
        acked_by_engine: Counter[str] = Counter()
        rejected_by_reason: Counter[str] = Counter()
        blocked_pair_groups: set[str] = set()
        pair_group_open_keys: dict[str, set[QuoteKey]] = {}
        for order_id, state in self.quote_order_state.items():
            key = state.key
            if key is None:
                continue
            if self.quote_order_ids.get(key) != order_id:
                continue
            tracked_intent = state.intent
            pair_group_id = str(
                tracked_intent.metadata.get("pair_group_id") or ""
            ).strip()
            if not pair_group_id:
                continue
            pair_group_open_keys.setdefault(pair_group_id, set()).add(key)

        def _cancel_open_pair_group_quotes(pair_group_id: str) -> None:
            open_keys = pair_group_open_keys.pop(pair_group_id, set())
            for open_key in list(open_keys):
                order_id = self.quote_order_ids.get(open_key)
                if order_id:
                    self.executor.cancel_order(order_id)
                self._clear_quote_slot(open_key, order_id=order_id, keep_reconcile=True)

        intent_idx = 0
        while intent_idx < len(pending_intents):
            intent = pending_intents[intent_idx]
            intent_idx += 1
            if not self._keep_running:
                break
            intent_type = str(intent.metadata.get("intent_type") or "alpha")
            pair_group_id = str(intent.metadata.get("pair_group_id") or "").strip()
            if pair_group_id and pair_group_id in blocked_pair_groups:
                rejected_by_reason["pair_group_blocked"] += 1
                continue
            is_rebalance_buy = intent.side == Side.BUY and intent_type in {
                "equalize",
                "equalize_forced",
                "equalize_immediate",
                "pair_completion",
            }
            effective_bankroll = max(
                self.config.bankroll_usdc, self.risk.current_equity()
            )
            key: QuoteKey | None = None
            if intent.post_only:
                key = self._quote_key(intent)
                if key in self.quote_order_ids:
                    plan = self.quote_order_plan.get(key)
                    prev_price = float(plan[0]) if plan else float(intent.price)
                    prev_size = float(plan[1]) if plan else float(intent.size)
                    prev_ts = float(plan[2]) if plan else 0.0
                    try:
                        quote_refresh_seconds = max(
                            0.25,
                            float(intent.metadata.get("quote_refresh_seconds", 3.0)),
                        )
                    except (TypeError, ValueError):
                        quote_refresh_seconds = 3.0
                    try:
                        quote_max_age_seconds = max(
                            quote_refresh_seconds,
                            float(intent.metadata.get("quote_max_age_seconds", 12.0)),
                        )
                    except (TypeError, ValueError):
                        quote_max_age_seconds = 12.0
                    hold_queue = bool(intent.metadata.get("hold_queue", False))
                    try:
                        min_quote_dwell_seconds = max(
                            0.0,
                            float(intent.metadata.get("min_quote_dwell_seconds", 0.0)),
                        )
                    except (TypeError, ValueError):
                        min_quote_dwell_seconds = 0.0
                    if hold_queue and min_quote_dwell_seconds <= 0:
                        min_quote_dwell_seconds = quote_refresh_seconds * 1.5
                    now_ts = time.time()
                    age = (
                        max(0.0, now_ts - prev_ts)
                        if prev_ts > 0
                        else quote_max_age_seconds + 1.0
                    )
                    price_threshold = 0.0095
                    size_threshold = max(0.5, prev_size * 0.15)
                    refresh_trigger_age = quote_refresh_seconds
                    refresh_due = age >= quote_max_age_seconds
                    if hold_queue:
                        # Hold queue priority during high-frequency oscillation:
                        # only reprice after minimum dwell and larger deltas.
                        price_threshold = max(price_threshold * 2.6, 0.03)
                        size_threshold = max(size_threshold * 2.2, 1.25)
                        refresh_trigger_age = max(
                            quote_refresh_seconds * 2.3,
                            min_quote_dwell_seconds + 0.75,
                        )
                        refresh_due = age >= max(
                            quote_max_age_seconds * 1.25,
                            min_quote_dwell_seconds + (quote_refresh_seconds * 1.8),
                        )
                    price_changed = abs(prev_price - intent.price) >= price_threshold
                    size_changed = abs(prev_size - intent.size) >= size_threshold
                    should_refresh = refresh_due or (
                        age >= refresh_trigger_age and (price_changed or size_changed)
                    )
                    if hold_queue and age < min_quote_dwell_seconds and not refresh_due:
                        should_refresh = False
                    if not should_refresh:
                        if hold_queue:
                            rejected_by_reason["hold_queue_wait"] += 1
                        else:
                            rejected_by_reason["already_quoted"] += 1
                        continue
                    existing_order_id = self.quote_order_ids.get(key)
                    cancelled = False
                    if existing_order_id:
                        cancelled = self.executor.cancel_order(existing_order_id)
                    # If cancel fails very quickly after placement, keep the old quote
                    # to avoid accidental duplicate open orders.
                    if (
                        existing_order_id
                        and (not cancelled)
                        and age < quote_max_age_seconds
                    ):
                        rejected_by_reason["quote_refresh_cancel_failed"] += 1
                        continue
                    self._clear_quote_slot(
                        key, order_id=existing_order_id, keep_reconcile=True
                    )

                # Guardrail: keep post-only quotes maker-safe against local book.
                # Instead of dropping intent outright, shift one tick away from cross.
                book = books.get(intent.token_id)
                if book:
                    if (
                        intent.side.value == "buy"
                        and book.best_ask > 0
                        and intent.price >= book.best_ask
                    ):
                        safety_ticks = 1
                        if bool(intent.metadata.get("hold_queue", False)):
                            safety_ticks = 2
                        try:
                            flip_rate = float(
                                intent.metadata.get("fluctuation_flip_rate_max", 0.0)
                            )
                        except (TypeError, ValueError):
                            flip_rate = 0.0
                        try:
                            swing_short = float(
                                intent.metadata.get("fluctuation_swing_short", 0.0)
                            )
                        except (TypeError, ValueError):
                            swing_short = 0.0
                        if flip_rate >= 0.14 or swing_short >= 0.020:
                            safety_ticks = max(safety_ticks, 2)
                        if flip_rate >= 0.22 or swing_short >= 0.030:
                            safety_ticks = max(safety_ticks, 3)
                        safe_price = round_tick(
                            clamp(book.best_ask - (0.01 * safety_ticks), 0.01, 0.99)
                        )
                        if safe_price < 0.01 or safe_price >= book.best_ask:
                            self.storage.record_risk_event(
                                "order_skip_cross_local",
                                {
                                    "market_id": intent.market_id,
                                    "engine": intent.engine,
                                    "side": intent.side.value,
                                },
                            )
                            rejected_by_reason["skip_cross_local"] += 1
                            continue
                        updated_metadata = dict(intent.metadata)
                        updated_metadata["post_only_sanitized"] = True
                        updated_metadata["post_only_sanitized_from"] = intent.price
                        updated_metadata["post_only_sanitized_ticks"] = safety_ticks
                        intent = replace(
                            intent, price=safe_price, metadata=updated_metadata
                        )
                    if (
                        intent.side.value == "sell"
                        and book.best_bid > 0
                        and intent.price <= book.best_bid
                    ):
                        safety_ticks = 1
                        if bool(intent.metadata.get("hold_queue", False)):
                            safety_ticks = 2
                        safe_price = round_tick(
                            clamp(book.best_bid + (0.01 * safety_ticks), 0.01, 0.99)
                        )
                        if safe_price > 0.99 or safe_price <= book.best_bid:
                            self.storage.record_risk_event(
                                "order_skip_cross_local",
                                {
                                    "market_id": intent.market_id,
                                    "engine": intent.engine,
                                    "side": intent.side.value,
                                },
                            )
                            rejected_by_reason["skip_cross_local"] += 1
                            continue
                        updated_metadata = dict(intent.metadata)
                        updated_metadata["post_only_sanitized"] = True
                        updated_metadata["post_only_sanitized_from"] = intent.price
                        updated_metadata["post_only_sanitized_ticks"] = safety_ticks
                        intent = replace(
                            intent, price=safe_price, metadata=updated_metadata
                        )

                # Reserve quote budget so live placement respects available balance, not just filled positions.
                max_total = effective_bankroll * self.config.max_total_exposure_pct
                projected_total = (
                    self.risk.total_exposure()
                    + self._reserved_notional_total()
                    + intent.notional
                )
                if projected_total > max_total:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "total exposure cap exceeded (including open quotes)",
                        },
                    )
                    rejected_by_reason["risk_total_cap"] += 1
                    continue

                max_market = effective_bankroll * self.config.max_market_exposure_pct
                projected_market = (
                    self.risk.market_exposure.get(intent.market_id, 0.0)
                    + self._reserved_notional_market(intent.market_id)
                    + intent.notional
                )
                if projected_market > max_market:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "market exposure cap exceeded (including open quotes)",
                        },
                    )
                    rejected_by_reason["risk_market_cap"] += 1
                    continue

            if (
                not intent.post_only
                and intent.side == Side.BUY
                and not is_rebalance_buy
            ):
                reserve_cash = (
                    0.0
                    if self.config.bankroll_usdc <= 50.0
                    else max(0.0, self.config.bankroll_usdc * 0.05)
                )
                available_alpha_cash = max(0.0, self.risk.cash - reserve_cash)
                min_size = float(intent.metadata.get("order_min_size", 5.0))
                min_notional = min_size * max(intent.price, 0.01)
                if available_alpha_cash < min_notional:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "alpha cash reserve",
                        },
                    )
                    rejected_by_reason["alpha_cash_reserve"] += 1
                    continue
                if intent.notional > available_alpha_cash:
                    sized_size = max(
                        min_size,
                        (available_alpha_cash / max(intent.price, 0.01)) * 0.995,
                    )
                    intent = replace(intent, size=sized_size)

                if intent_type == "alpha_entry":
                    opposite_entry_raw = intent.metadata.get("opposite_entry")
                    try:
                        opposite_entry = float(opposite_entry_raw)
                    except (TypeError, ValueError):
                        opposite_entry = 0.0
                    if opposite_entry > 0:
                        pair_unit_cost_raw = intent.metadata.get("pair_cost_hint")
                        try:
                            pair_unit_cost = float(pair_unit_cost_raw)
                        except (TypeError, ValueError):
                            pair_unit_cost = max(0.01, intent.price) + max(
                                0.01, opposite_entry
                            )
                        micro_bankroll = self.config.bankroll_usdc <= 50.0
                        if self.config.bankroll_usdc <= 50.0:
                            hedge_fraction = 0.45
                            pair_unit_cost = (
                                max(0.01, intent.price)
                                + (max(0.01, opposite_entry) * hedge_fraction)
                                + (2.0 * self.config.directional_slippage_buffer)
                            )
                        pair_unit_cost = max(0.01, pair_unit_cost)
                        max_alpha_size = (available_alpha_cash / pair_unit_cost) * 0.99
                        if max_alpha_size < min_size and not micro_bankroll:
                            self.storage.record_risk_event(
                                "risk_reject",
                                {
                                    "market_id": intent.market_id,
                                    "engine": intent.engine,
                                    "reason": "alpha completion cash",
                                },
                            )
                            rejected_by_reason["alpha_completion_cash"] += 1
                            continue
                        if intent.size > max_alpha_size and max_alpha_size >= min_size:
                            intent = replace(intent, size=max(min_size, max_alpha_size))

                # Reserve IOC budget within this cycle so a burst of aggressive
                # orders cannot exceed configured bankroll caps before fills post.
                max_total = effective_bankroll * self.config.max_total_exposure_pct
                projected_total = (
                    self.risk.total_exposure()
                    + self._reserved_notional_total()
                    + planned_ioc_total
                    + intent.notional
                )
                if projected_total > max_total:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "total exposure cap exceeded (including planned IOC buys)",
                        },
                    )
                    rejected_by_reason["risk_total_cap"] += 1
                    continue

                max_market = effective_bankroll * self.config.max_market_exposure_pct
                projected_market = (
                    self.risk.market_exposure.get(intent.market_id, 0.0)
                    + self._reserved_notional_market(intent.market_id)
                    + planned_ioc_market.get(intent.market_id, 0.0)
                    + intent.notional
                )
                if projected_market > max_market:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "market exposure cap exceeded (including planned IOC buys)",
                        },
                    )
                    rejected_by_reason["risk_market_cap"] += 1
                    continue

            if (
                self.config.live_mode
                and intent.side == Side.BUY
                and intent.notional < 1.0
            ):
                min_size = float(intent.metadata.get("order_min_size", 5.0))
                min_notional_target = 1.0
                sized_size = max(
                    min_size, (min_notional_target / max(intent.price, 0.01)) * 1.01
                )
                if sized_size > intent.size:
                    intent = replace(intent, size=sized_size)

            if intent.side == Side.BUY:
                pacing_headroom = self._buy_cash_pacing_headroom(
                    intent, intent_type=intent_type
                )
                if pacing_headroom is not None:
                    min_size = float(intent.metadata.get("order_min_size", 5.0))
                    min_notional = min_size * max(intent.price, 0.01)
                    if pacing_headroom < min_notional:
                        self.storage.record_risk_event(
                            "risk_reject",
                            {
                                "market_id": intent.market_id,
                                "engine": intent.engine,
                                "reason": "cash pacing cap",
                            },
                        )
                        rejected_by_reason["cash_pacing"] += 1
                        continue
                    if intent.notional > pacing_headroom:
                        sized_size = max(
                            min_size,
                            (pacing_headroom / max(intent.price, 0.01)) * 0.995,
                        )
                        intent = replace(intent, size=sized_size)

            if (
                self.config.live_mode
                and intent.side == Side.BUY
                and intent.notional < 1.0
            ):
                self.storage.record_risk_event(
                    "risk_reject",
                    {
                        "market_id": intent.market_id,
                        "engine": intent.engine,
                        "reason": "exchange min notional",
                    },
                )
                rejected_by_reason["exchange_min_notional"] += 1
                continue

            if intent.side == Side.BUY:
                fee_bps_raw = intent.metadata.get("fee_bps", 0)
                try:
                    fee_bps = int(fee_bps_raw)
                except (TypeError, ValueError):
                    fee_bps = 0
                per_share_cash = max(0.0, intent.price) + per_share_fee(
                    intent.price, fee_bps
                )
                if per_share_cash <= 0:
                    rejected_by_reason["risk_reject"] += 1
                    continue
                min_size = float(intent.metadata.get("order_min_size", 5.0))
                available_cash = max(
                    0.0,
                    self.risk.cash
                    - self._reserved_notional_total()
                    - planned_ioc_total,
                )
                min_required_cash = min_size * per_share_cash
                if available_cash + 1e-9 < min_required_cash:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "cash reserved by open quotes",
                        },
                    )
                    rejected_by_reason["cash_reserved"] += 1
                    continue
                max_affordable_size = max(
                    0.0, (available_cash / per_share_cash) * 0.995
                )
                max_affordable_size = PairArbEngine._size_to_lot(
                    max_affordable_size, min_size
                )
                if max_affordable_size < min_size:
                    self.storage.record_risk_event(
                        "risk_reject",
                        {
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "reason": "cash reserved by open quotes",
                        },
                    )
                    rejected_by_reason["cash_reserved"] += 1
                    continue
                if intent.size > max_affordable_size:
                    intent = replace(intent, size=max_affordable_size)

            decision = self.risk.can_place(intent)
            if (
                not decision.allowed
                and intent.side == Side.BUY
                and not intent.post_only
            ):
                # Fast path for IOC alpha: scale down to remaining cap headroom instead of hard reject spam.
                reason = (decision.reason or "").lower()
                if "cash budget exceeded" in reason or is_rebalance_buy:
                    remaining_notional = max(0.0, self.risk.cash)
                else:
                    max_total = effective_bankroll * self.config.max_total_exposure_pct
                    max_market = (
                        effective_bankroll * self.config.max_market_exposure_pct
                    )
                    remaining_total = (
                        max_total - self.risk.total_exposure() - planned_ioc_total
                    )
                    remaining_market = (
                        max_market
                        - self.risk.market_exposure.get(intent.market_id, 0.0)
                        - planned_ioc_market.get(intent.market_id, 0.0)
                    )
                    remaining_notional = min(remaining_total, remaining_market)
                if remaining_notional > 0:
                    min_size = float(intent.metadata.get("order_min_size", 5.0))
                    min_notional = min_size * max(intent.price, 0.01)
                    if remaining_notional >= min_notional:
                        sized_size = max(
                            min_size, remaining_notional / max(intent.price, 0.01)
                        )
                        # Keep below cap boundary to avoid float edge rejects.
                        sized_size = max(min_size, sized_size * 0.995)
                        intent = replace(intent, size=sized_size)
                        decision = self.risk.can_place(intent)

            if not decision.allowed:
                reason = decision.reason or "risk rejection"
                self.storage.record_risk_event(
                    "risk_reject",
                    {
                        "market_id": intent.market_id,
                        "engine": intent.engine,
                        "reason": reason,
                    },
                )
                if "market exposure cap exceeded" in reason:
                    rejected_by_reason["risk_market_cap"] += 1
                elif "total exposure cap exceeded" in reason:
                    rejected_by_reason["risk_total_cap"] += 1
                else:
                    rejected_by_reason["risk_reject"] += 1
                continue

            fair = intent.metadata.get("p_fair")
            edge = intent.metadata.get("edge_net")
            best_ask = intent.metadata.get("best_ask")
            if not intent.post_only:
                ioc_key = (
                    intent.engine,
                    intent.market_id,
                    intent.token_id,
                    intent.side.value,
                    intent_type,
                )
                cooldown_seconds = 0.20 if intent.engine == "engine_pair_arb" else 1.0
                now_ts = time.time()
                prev = self._last_ioc_submission.get(ioc_key)
                if (
                    cooldown_seconds > 0
                    and prev
                    and (now_ts - prev[0]) < cooldown_seconds
                    and abs(prev[1] - intent.price) < 1e-9
                ):
                    rejected_by_reason["cooldown_duplicate"] += 1
                    continue
                self._last_ioc_submission[ioc_key] = (now_ts, intent.price)

            if isinstance(fair, (int, float)) and isinstance(edge, (int, float)):
                LOGGER.info(
                    (
                        "order_try market=%s engine=%s intent_type=%s side=%s tif=%s post_only=%s "
                        "price=%.4f size=%.4f notional=%.2f fair=%.4f edge=%.4f best_ask=%.4f"
                    ),
                    intent.market_id,
                    intent.engine,
                    intent_type,
                    intent.side.value,
                    intent.tif.value,
                    intent.post_only,
                    intent.price,
                    intent.size,
                    intent.notional,
                    float(fair),
                    float(edge),
                    float(best_ask or 0.0),
                )
            else:
                LOGGER.info(
                    "order_try market=%s engine=%s intent_type=%s side=%s tif=%s post_only=%s price=%.4f size=%.4f notional=%.2f",
                    intent.market_id,
                    intent.engine,
                    intent_type,
                    intent.side.value,
                    intent.tif.value,
                    intent.post_only,
                    intent.price,
                    intent.size,
                    intent.notional,
                )
            executed += 1
            if not intent.post_only and intent.side == Side.BUY:
                planned_ioc_total += intent.notional
                planned_ioc_market[intent.market_id] = (
                    planned_ioc_market.get(intent.market_id, 0.0) + intent.notional
                )
            result = self.executor.place_order(intent, books)
            self.storage.record_order(intent, result, self.config.mode)
            LOGGER.info(
                "order_result market=%s engine=%s status=%s order_id=%s filled=%.4f",
                intent.market_id,
                intent.engine,
                result.status,
                result.order_id,
                result.filled_size,
            )

            if result.is_error:
                error_text = self._extract_error_text(result.raw)
                lowered = error_text.lower()
                LOGGER.warning(
                    "order_error market=%s engine=%s status=%s reason=%s",
                    intent.market_id,
                    intent.engine,
                    result.status,
                    error_text or "unknown",
                )

                if self._is_post_only_cross_error(error_text):
                    self.storage.record_risk_event(
                        "order_reject_cross_exchange",
                        {
                            "order_id": result.order_id,
                            "market_id": intent.market_id,
                            "engine": intent.engine,
                            "raw": result.raw,
                        },
                    )
                    # This is expected for stale post-only quotes; don't accumulate execution-failure state.
                    self.risk.on_execution_success()
                    rejected_by_reason["reject_cross_exchange"] += 1
                    if pair_group_id and intent_type in {
                        "pair_entry_primary",
                        "pair_completion",
                    }:
                        # Pair legs should not be left orphaned when one side is
                        # rejected as crossing; cancel siblings in the same pair
                        # group so inventory stays balanced.
                        blocked_pair_groups.add(pair_group_id)
                        _cancel_open_pair_group_quotes(pair_group_id)
                        rejected_by_reason["pair_group_cross_cancel"] += 1
                    continue

                errors += 1
                self.storage.record_risk_event(
                    "execution_error", {"order_id": result.order_id, "raw": result.raw}
                )
                if self._is_balance_allowance_error(error_text):
                    if intent.side == Side.SELL:
                        rejected_by_reason["insufficient_inventory"] += 1
                        self.storage.record_risk_event(
                            "order_reject_insufficient_inventory",
                            {
                                "order_id": result.order_id,
                                "market_id": intent.market_id,
                                "engine": intent.engine,
                                "raw": result.raw,
                            },
                        )
                        # Selling without inventory should not halt the bot.
                        self.risk.on_execution_success()
                        continue
                    rejected_by_reason["balance_allowance"] += 1
                    self.storage.record_risk_event(
                        "funding_error",
                        {
                            "order_id": result.order_id,
                            "message": (
                                "insufficient available balance/allowance for submitted live order; "
                                "order skipped without halting"
                            ),
                            "raw": result.raw,
                        },
                    )
                    if is_rebalance_buy:
                        self.risk.on_execution_success()
                        continue
                    self.risk.on_execution_error()
                    if pair_group_id:
                        blocked_pair_groups.add(pair_group_id)
                        _cancel_open_pair_group_quotes(pair_group_id)
                    continue
                if "invalid signature" in lowered:
                    rejected_by_reason["invalid_signature"] += 1
                    self.risk.halt("invalid signature")
                    self.storage.record_risk_event(
                        "auth_error",
                        {
                            "order_id": result.order_id,
                            "message": "invalid signature from CLOB, check key/proxy/signature type",
                            "raw": result.raw,
                        },
                    )
                    break
                rejected_by_reason["execution_error"] += 1
                self.risk.on_execution_error()
                if pair_group_id:
                    blocked_pair_groups.add(pair_group_id)
                    _cancel_open_pair_group_quotes(pair_group_id)
            else:
                self.risk.on_execution_success()
                acked_by_engine[intent.engine] += 1
            if result.is_filled:
                fills += 1
                self.risk.apply_fill(result)
                self._record_pair_cost_fill(intent, result)
                now_ts = time.time()
                self._record_alpha_fill_for_learning(intent, result, now_ts)
                self._match_hedge_fill_for_learning(intent, result, now_ts)
                immediate_hedge = self._build_immediate_equalizer(
                    parent_intent=intent,
                    fill=result,
                    books=books,
                )
                if immediate_hedge is not None:
                    pending_intents.insert(intent_idx, immediate_hedge)
                    intents_by_engine[immediate_hedge.engine] += 1

            if intent.post_only and key:
                terminal_status = {
                    "rejected",
                    "error",
                    "filled",
                    "cancelled",
                    "canceled",
                }
                if result.status in terminal_status or max(0.0, result.filled_size) >= (
                    intent.size - 1e-9
                ):
                    self._clear_quote_slot(key, order_id=result.order_id)
                else:
                    remaining_size = max(
                        0.0, intent.size - max(0.0, result.filled_size)
                    )
                    if remaining_size <= 1e-9:
                        self._clear_quote_slot(key, order_id=result.order_id)
                    elif not result.order_id:
                        self._clear_quote_slot(key)
                    else:
                        self.quote_order_ids[key] = result.order_id
                        self.quote_order_notional[key] = remaining_size * max(
                            0.01, intent.price
                        )
                        self.quote_order_plan[key] = (
                            intent.price,
                            remaining_size,
                            time.time(),
                        )
                        self.quote_order_state[result.order_id] = QuoteOrderState(
                            key=key,
                            intent=intent,
                            cum_filled_size=max(0.0, result.filled_size),
                            cum_fee_paid=max(0.0, result.fee_paid),
                            cum_notional=max(0.0, result.filled_size)
                            * max(0.0, result.filled_price),
                        )

                if pair_group_id and key:
                    if key in self.quote_order_ids:
                        pair_group_open_keys.setdefault(pair_group_id, set()).add(key)
                    elif pair_group_id in pair_group_open_keys:
                        pair_group_open_keys[pair_group_id].discard(key)
                        if not pair_group_open_keys[pair_group_id]:
                            pair_group_open_keys.pop(pair_group_id, None)
        self._expire_unhedged_alpha_legs(books, time.time())
        LOGGER.info(
            "cycle=%s actions intents_by_engine=%s acked_by_engine=%s rejects=%s open_quotes=%s reserved_notional=%.2f",
            self._cycle_counter,
            dict(intents_by_engine),
            dict(acked_by_engine),
            dict(rejected_by_reason),
            len(self.quote_order_ids),
            self._reserved_notional_total(),
        )
        return executed, fills, errors

    def _process_paper_sweeps(self, books: dict[str, OrderBookSnapshot]) -> int:
        fills = 0
        sweep_results = self.executor.sweep(books)
        for intent, result in sweep_results:
            self.storage.record_order(intent, result, self.config.mode)
            if result.is_filled:
                fills += 1
                self.risk.apply_fill(result)
                self._record_pair_cost_fill(intent, result)
                for key, order_id in list(self.quote_order_ids.items()):
                    if order_id != result.order_id:
                        continue
                    self._clear_quote_slot(key, order_id=order_id)
        return fills

    def _apply_trade_fill_to_quote_state(
        self,
        *,
        order_id: str,
        fill_size: float,
        fill_price: float,
    ) -> None:
        state = self.quote_order_state.get(order_id)
        if state is None:
            return
        intent = state.intent
        prev_cum_size = max(0.0, float(state.cum_filled_size))
        prev_cum_notional = max(0.0, float(state.cum_notional))
        prev_cum_fee = max(0.0, float(state.cum_fee_paid))
        new_cum_size = min(intent.size, prev_cum_size + max(0.0, float(fill_size)))
        new_cum_notional = prev_cum_notional + (
            max(0.0, float(fill_size)) * max(0.0, float(fill_price))
        )
        state.cum_filled_size = new_cum_size
        state.cum_notional = max(prev_cum_notional, new_cum_notional)
        state.cum_fee_paid = prev_cum_fee
        key = state.key
        if key is None:
            if new_cum_size >= (intent.size - 1e-9):
                self.quote_order_state.pop(order_id, None)
            return
        remaining = max(0.0, intent.size - new_cum_size)
        if remaining <= 1e-9:
            self._clear_quote_slot(key, order_id=order_id)
            return
        self.quote_order_notional[key] = remaining * max(0.01, intent.price)
        plan = self.quote_order_plan.get(key)
        prev_ts = float(plan[2]) if plan else time.time()
        self.quote_order_plan[key] = (intent.price, remaining, prev_ts)

    def _process_live_trade_reconciliation(
        self, books: dict[str, OrderBookSnapshot]
    ) -> int:
        get_recent_trades = getattr(self.executor, "get_recent_maker_trade_fills", None)
        if not callable(get_recent_trades):
            return 0
        after_ts = max(0, int(self._last_trade_sync_ts) - 2)
        try:
            maker_fills = get_recent_trades(after_ts=after_ts)
        except Exception as exc:
            self.storage.record_risk_event(
                "trade_reconcile_error",
                {"error": str(exc)},
            )
            LOGGER.warning(
                "cycle=%s trade_reconcile_error=%s",
                self._cycle_counter,
                exc,
            )
            self._last_trade_sync_ts = int(time.time())
            return 0

        token_market_map = self._tracked_inventory_token_market_map()
        fill_count = 0
        follow_up_intents: list[OrderIntent] = []
        max_match_time = after_ts
        for fill in maker_fills:
            trade_id = str(getattr(fill, "trade_id", "")).strip()
            order_id = str(getattr(fill, "order_id", "")).strip()
            token_id = str(getattr(fill, "token_id", "")).strip()
            if not trade_id or not order_id or not token_id:
                continue
            dedupe_key = (trade_id, order_id)
            if dedupe_key in self._seen_trade_fills:
                continue
            self._seen_trade_fills.add(dedupe_key)
            match_time = int(getattr(fill, "match_time", 0) or 0)
            max_match_time = max(max_match_time, match_time)
            tracked_intent: OrderIntent | None = None
            tracked_state = self.quote_order_state.get(order_id)
            if tracked_state is not None:
                tracked_intent = tracked_state.intent
            market_id = ""
            if tracked_intent is not None:
                market_id = str(tracked_intent.market_id or "").strip()
            if not market_id:
                market_id = token_market_map.get(token_id, "")
            if not market_id:
                self.storage.record_risk_event(
                    "trade_reconcile_unknown_token",
                    {
                        "trade_id": trade_id,
                        "order_id": order_id,
                        "token_id": token_id,
                    },
                )
                continue
            side = getattr(fill, "side", Side.BUY)
            if side not in {Side.BUY, Side.SELL}:
                continue
            fill_size = max(0.0, float(getattr(fill, "size", 0.0) or 0.0))
            fill_price = max(0.0, float(getattr(fill, "price", 0.0) or 0.0))
            if fill_size <= 0 or fill_price <= 0:
                continue
            intent: OrderIntent
            if (
                tracked_intent is not None
                and tracked_intent.token_id == token_id
                and tracked_intent.side == side
            ):
                intent = tracked_intent
            else:
                intent = OrderIntent(
                    market_id=market_id,
                    token_id=token_id,
                    side=side,
                    price=fill_price,
                    size=fill_size,
                    tif=TimeInForce.GTC,
                    post_only=True,
                    engine="engine_pair_arb",
                    expected_edge=0.0,
                    metadata={
                        "intent_type": "trade_reconcile",
                        "source": "clob_trades",
                        "trade_id": trade_id,
                        "match_time": match_time,
                    },
                )
            fill_result = OrderResult(
                order_id=order_id,
                market_id=market_id,
                token_id=token_id,
                side=side,
                price=fill_price,
                size=fill_size,
                status="filled",
                filled_size=fill_size,
                filled_price=fill_price,
                fee_paid=0.0,
                engine="engine_pair_arb",
                created_at=datetime.now(tz=timezone.utc),
                raw={
                    "reconciled": True,
                    "source": "clob_trades",
                    "trade_id": trade_id,
                    "match_time": match_time,
                },
            )
            self.storage.record_order(intent, fill_result, self.config.mode)
            self._record_pair_cost_fill(intent, fill_result)
            self._apply_trade_fill_to_quote_state(
                order_id=order_id,
                fill_size=fill_size,
                fill_price=fill_price,
            )
            fill_count += 1
            immediate_hedge = self._build_immediate_equalizer(
                parent_intent=intent,
                fill=fill_result,
                books=books,
            )
            if immediate_hedge is not None:
                follow_up_intents.append(immediate_hedge)

        if follow_up_intents:
            _, follow_fill_count, _ = self._execute_intents(
                follow_up_intents,
                books,
                prune_unplanned_quotes=False,
            )
            fill_count += follow_fill_count

        self._last_trade_sync_ts = max(
            int(time.time()),
            int(getattr(self, "_last_trade_sync_ts", 0) or 0),
            int(max_match_time),
        )
        if fill_count > 0:
            LOGGER.info(
                "cycle=%s trade_reconcile_fills=%s",
                self._cycle_counter,
                fill_count,
            )
        return fill_count

    @staticmethod
    def _is_terminal_order_status(status: str) -> bool:
        lowered = (status or "").strip().lower()
        return lowered in {
            "rejected",
            "error",
            "filled",
            "cancelled",
            "canceled",
            "closed",
            "executed",
            "matched",
        }

    def _process_live_quote_reconciliation(
        self, books: dict[str, OrderBookSnapshot]
    ) -> int:
        fills = 0
        follow_up_intents: list[OrderIntent] = []
        for order_id, state in list(self.quote_order_state.items()):
            intent = state.intent
            key = state.key

            latest = self.executor.get_order_result(order_id, intent)
            if latest is None:
                continue

            prev_cum_size = max(0.0, float(state.cum_filled_size))
            prev_cum_fee = max(0.0, float(state.cum_fee_paid))
            prev_cum_notional = max(0.0, float(state.cum_notional))

            cum_size = max(0.0, latest.filled_size)
            cum_fee = max(0.0, latest.fee_paid)
            exchange_cum_notional = cum_size * max(0.0, latest.filled_price)
            cum_notional = max(prev_cum_notional, exchange_cum_notional)

            delta_size = max(0.0, cum_size - prev_cum_size)
            if delta_size > 1e-9:
                delta_notional = max(0.0, cum_notional - prev_cum_notional)
                delta_fee = max(0.0, cum_fee - prev_cum_fee)
                if delta_notional > 0:
                    delta_fill_price = delta_notional / delta_size
                elif latest.filled_price > 0:
                    delta_fill_price = latest.filled_price
                else:
                    delta_fill_price = max(0.01, intent.price)
                fill_result = OrderResult(
                    order_id=order_id,
                    market_id=intent.market_id,
                    token_id=intent.token_id,
                    side=intent.side,
                    price=intent.price,
                    size=intent.size,
                    status="partial_fill",
                    filled_size=delta_size,
                    filled_price=delta_fill_price,
                    fee_paid=delta_fee,
                    engine=intent.engine,
                    created_at=datetime.now(tz=timezone.utc),
                    raw={
                        "reconciled": True,
                        "order_status": latest.status,
                        "order_state": latest.raw.get("order_state", latest.raw),
                    },
                )
                self.storage.record_order(intent, fill_result, self.config.mode)
                fills += 1
                self.risk.apply_fill(fill_result)
                self._record_pair_cost_fill(intent, fill_result)
                now_ts = time.time()
                self._record_alpha_fill_for_learning(intent, fill_result, now_ts)
                self._match_hedge_fill_for_learning(intent, fill_result, now_ts)
                immediate_hedge = self._build_immediate_equalizer(
                    parent_intent=intent,
                    fill=fill_result,
                    books=books,
                )
                if immediate_hedge is not None:
                    follow_up_intents.append(immediate_hedge)

            state.cum_filled_size = max(prev_cum_size, cum_size)
            state.cum_fee_paid = max(prev_cum_fee, cum_fee)
            state.cum_notional = max(prev_cum_notional, cum_notional)

            terminal = self._is_terminal_order_status(latest.status) or (
                state.cum_filled_size >= (intent.size - 1e-9)
            )
            if key is not None:
                remaining_size = max(0.0, intent.size - float(state.cum_filled_size))
                if terminal or remaining_size <= 1e-9:
                    self._clear_quote_slot(key, order_id=order_id)
                else:
                    self.quote_order_notional[key] = remaining_size * max(
                        0.01, intent.price
                    )
                    plan = self.quote_order_plan.get(key)
                    prev_ts = float(plan[2]) if plan else time.time()
                    self.quote_order_plan[key] = (intent.price, remaining_size, prev_ts)
            elif terminal:
                self.quote_order_state.pop(order_id, None)

        if follow_up_intents:
            _, follow_fill_count, _ = self._execute_intents(
                follow_up_intents,
                books,
                prune_unplanned_quotes=False,
            )
            fills += follow_fill_count
        return fills

    def _cancel_quotes(self) -> None:
        self.executor.cancel_all()
        self.quote_order_ids.clear()
        self.quote_order_notional.clear()
        self.quote_order_plan.clear()
        self.quote_order_state.clear()

    def _finalize_live_quote_reconciliation(self, timeout_seconds: float = 4.0) -> None:
        if not self.config.live_mode:
            return
        if not self.quote_order_ids and not self.quote_order_state:
            return
        try:
            self.executor.cancel_all()
        except Exception:
            pass

        deadline = time.time() + max(0.5, float(timeout_seconds))
        while time.time() < deadline and self.quote_order_state:
            fill_count = self._process_live_quote_reconciliation({})
            self.storage.record_risk_state(self.risk.state())
            if fill_count <= 0:
                time.sleep(0.25)

        if self.quote_order_state:
            LOGGER.warning(
                "final_reconciliation_incomplete pending_orders=%s; clearing local quote state",
                len(self.quote_order_state),
            )
        self.quote_order_ids.clear()
        self.quote_order_notional.clear()
        self.quote_order_plan.clear()
        self.quote_order_state.clear()

    @staticmethod
    def _is_fatal_halt_reason(reason: str) -> bool:
        lowered = (reason or "").lower()
        fatal_tokens = (
            "invalid signature",
            "not enough balance",
            "allowance",
            "consecutive execution errors",
            "auth",
            "funding",
        )
        return any(token in lowered for token in fatal_tokens)

    @staticmethod
    def _extract_error_text(raw: dict) -> str:
        if not raw:
            return ""
        if isinstance(raw.get("error_msg"), dict):
            value = raw.get("error_msg", {}).get("error", "")
            return str(value or "")
        if "error_msg" in raw:
            return str(raw.get("error_msg") or "")
        if "error" in raw:
            return str(raw.get("error") or "")
        return json.dumps(raw, separators=(",", ":"), default=str)

    @staticmethod
    def _is_post_only_cross_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return "post-only" in lowered and "cross" in lowered

    @staticmethod
    def _is_balance_allowance_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return "not enough balance" in lowered or "allowance" in lowered

    def _reserved_notional_total(self) -> float:
        return sum(self.quote_order_notional.values())

    def _reserved_notional_market(self, market_id: str) -> float:
        total = 0.0
        for key, value in self.quote_order_notional.items():
            if key[1] == market_id:
                total += value
        return total

    def _buy_cash_pacing_headroom(
        self, intent: OrderIntent, *, intent_type: str
    ) -> float | None:
        if intent.side != Side.BUY:
            return None
        if intent_type in {"equalize", "equalize_forced", "equalize_immediate"}:
            return None
        timeframe = self._parse_timeframe(intent.metadata.get("timeframe"))
        if timeframe != Timeframe.FIVE_MIN:
            return None
        try:
            seconds_to_end = float(intent.metadata.get("seconds_to_end", 0.0))
        except (TypeError, ValueError):
            return None
        if seconds_to_end <= 0:
            return None

        # Pace cash usage across the full 5m market window so inventory can be
        # recycled through the interval instead of consuming all buying power
        # in the first burst.
        horizon_seconds = 300.0
        progress = clamp(1.0 - (seconds_to_end / horizon_seconds), 0.0, 1.0)
        # Anchor pacing to live equity so a static config bankroll does not
        # throttle buys after inventory sync updates real wallet cash.
        bankroll = max(
            1.0,
            float(self.risk.current_equity()),
            max(0.0, float(self.risk.cash)),
        )
        if bankroll >= 150.0:
            spend_fraction = 0.55 + (0.40 * progress)  # 55% at open -> 95% by expiry
        else:
            spend_fraction = 0.35 + (0.55 * progress)  # 35% at open -> 90% by expiry
        if seconds_to_end <= 45.0:
            spend_fraction = 1.00
        used_cash = max(0.0, bankroll - max(0.0, self.risk.cash))
        soft_cap = bankroll * clamp(spend_fraction, 0.15, 1.00)
        return max(0.0, soft_cap - used_cash)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "web3", "websocket"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _duration_cap(current: int) -> int:
    return current if current > 0 else 1


def _apply_duration_profile(config: BotConfig, duration: str | None) -> BotConfig:
    normalized = str(duration or "").strip().lower()
    if not normalized or normalized == "all":
        return config
    if normalized == "5m":
        return replace(
            config,
            enabled_tags=(_DURATION_5M_TAG,),
            max_trade_markets_5m=_duration_cap(config.max_trade_markets_5m),
            max_trade_markets_15m=0,
            max_trade_markets_1h=0,
        )
    if normalized == "15m":
        return replace(
            config,
            enabled_tags=(_DURATION_15M_TAG,),
            max_trade_markets_5m=0,
            max_trade_markets_15m=_duration_cap(config.max_trade_markets_15m),
            max_trade_markets_1h=0,
        )
    if normalized == "1h":
        return replace(
            config,
            enabled_tags=(_DURATION_1H_TAG,),
            max_trade_markets_5m=0,
            max_trade_markets_15m=0,
            max_trade_markets_1h=_duration_cap(config.max_trade_markets_1h),
        )
    raise ValueError(f"unsupported duration={duration!r}")


def _run_command(args: argparse.Namespace) -> int:
    config = load_config()
    if args.mode:
        config = replace(config, mode=args.mode.lower())
    _setup_logging(config.log_level)
    if args.bankroll is not None:
        if args.bankroll <= 0:
            LOGGER.error("--bankroll must be > 0")
            return 2
        config = replace(config, bankroll_usdc=float(args.bankroll))
    try:
        config = _apply_duration_profile(config, getattr(args, "duration", None))
    except ValueError as exc:
        LOGGER.error(str(exc))
        return 2

    if config.live_mode and not config.live_permitted:
        LOGGER.error(
            "Live mode blocked by compliance gate. BOT_REGION=%s BOT_PROVINCE=%s",
            config.bot_region,
            config.bot_province,
        )
        return 2

    runtime = BotRuntime(config)
    try:
        runtime.preflight()
    except Exception as exc:
        LOGGER.error("Live preflight failed: %s", exc)
        runtime.close()
        return 2
    LOGGER.info(
        "Starting bot mode=%s tags=%s",
        config.mode,
        ",".join(str(x) for x in config.enabled_tags),
    )
    signal_count = {"count": 0}

    def _handle_signal(signum: int, _frame: object) -> None:
        signal_count["count"] += 1
        if signal_count["count"] >= 2:
            LOGGER.error("Received signal %s again, forcing exit now", signum)
            raise SystemExit(130)
        LOGGER.warning(
            "Received signal %s, stopping loop (press Ctrl+C again to force-exit)",
            signum,
        )
        runtime.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        runtime.run()
        return 0
    except Exception as exc:
        LOGGER.error("Fatal runtime error: %s", exc)
        return 2
    finally:
        runtime.close()


def _report_command(args: argparse.Namespace) -> int:
    config = load_config()
    _setup_logging(config.log_level)
    storage = Storage(config.database_path)
    try:
        report = storage.report(args.window)
        print(json.dumps(report, indent=2, default=str))
    finally:
        storage.close()
    return 0


def _parse_seed_timeframe(raw: str) -> Timeframe | None:
    value = str(raw or "").strip().lower()
    if value in {"", "auto"}:
        return None
    if value == "5m":
        return Timeframe.FIVE_MIN
    if value == "15m":
        return Timeframe.FIFTEEN_MIN
    if value == "1h":
        return Timeframe.ONE_HOUR
    raise ValueError(f"unsupported timeframe={raw!r}")


def _seed_learning_command(args: argparse.Namespace) -> int:
    config = load_config()
    _setup_logging(config.log_level)
    storage = Storage(config.database_path)
    try:
        if args.reset_existing:
            storage.clear_pair_learning_data()
        learner = PairTimingLearner(storage)
        forced_timeframe = _parse_seed_timeframe(args.timeframe)
        result: LearningSeedResult = seed_pair_learning_from_trader_history(
            learner=learner,
            data_api_url=config.data_api_url,
            user=args.user,
            max_pages=int(args.max_pages),
            forced_timeframe=forced_timeframe,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    except Exception as exc:
        LOGGER.error("seed-learning failed: %s", exc)
        return 2
    finally:
        storage.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket_bot", description="BTC tenor trading bot"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run trading loop")
    run.add_argument("--mode", choices=("paper", "live"), default=None)
    run.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Starting bankroll in USDC (e.g. 100)",
    )
    run.add_argument(
        "--duration",
        choices=("5m", "15m", "1h", "all"),
        default=None,
        help="Restrict trading to a single timeframe (e.g. --duration 5m)",
    )
    run.set_defaults(func=_run_command)

    report = sub.add_parser("report", help="Print PnL/report summary from SQLite")
    report.add_argument("--window", type=int, default=24, help="Window in hours")
    report.set_defaults(func=_report_command)

    seed = sub.add_parser(
        "seed-learning",
        help="One-time seed of learner stats from a trader's historical activity",
    )
    seed.add_argument("--user", required=True, help="Trader wallet address")
    seed.add_argument(
        "--timeframe",
        choices=("auto", "5m", "15m", "1h"),
        default="auto",
        help="Force timeframe for all imported trades, or auto-detect",
    )
    seed.add_argument(
        "--max-pages",
        type=int,
        default=120,
        help="Maximum number of activity pages to scan",
    )
    seed.add_argument(
        "--reset-existing",
        action="store_true",
        help="Clear existing pair learning events/stats before seeding",
    )
    seed.set_defaults(func=_seed_learning_command)
    return parser


def cli(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()

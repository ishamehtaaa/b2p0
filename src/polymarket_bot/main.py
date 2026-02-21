from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import math
import signal
import threading
import sys
import time
from typing import Iterable

from polymarket_bot.clients_clob import ClobClient, ClobMarketStream
from polymarket_bot.clients_gamma import GammaClient
from polymarket_bot.clients_spot import BtcSpotClient, BtcSpotStream
from polymarket_bot.config import BotConfig, load_config
from polymarket_bot.engines.engine_pair_arb import PairArbEngine
from polymarket_bot.execution import BaseExecutor, LiveExecutor, PaperExecutor
from polymarket_bot.http_utils import get_json
from polymarket_bot.learning import PairTimingLearner
from polymarket_bot.models import (
    FeeInfo,
    MarketInfo,
    MarketSnapshot,
    OrderBookSnapshot,
    OrderIntent,
    OrderResult,
    Side,
    TimeInForce,
    Timeframe,
)
from polymarket_bot.pricing import clamp, per_share_fee, round_tick
from polymarket_bot.risk import RiskManager
from polymarket_bot.storage import Storage


LOGGER = logging.getLogger("polymarket_bot")


class SpotTracker:
    def __init__(self) -> None:
        self.points: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def update(self, now_ts: float, price: float) -> None:
        with self._lock:
            self.points.append((now_ts, price))
            while self.points and now_ts - self.points[0][0] > 900:
                self.points.popleft()

    def _snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self.points)

    def point_count(self) -> int:
        with self._lock:
            return len(self.points)

    def latest_age_seconds(self, now_ts: float | None = None) -> float:
        points = self._snapshot()
        if not points:
            return float("inf")
        current_ts = now_ts if now_ts is not None else time.time()
        return max(0.0, current_ts - points[-1][0])

    def price_at_or_before(self, target_ts: float, max_lookback_seconds: float = 600.0) -> float | None:
        points = self._snapshot()
        if not points:
            return None
        if target_ts < points[0][0]:
            return None

        candidate: float | None = None
        candidate_ts = 0.0
        for ts, price in points:
            if ts <= target_ts:
                candidate = price
                candidate_ts = ts
            else:
                break
        if candidate is None:
            return None
        if target_ts - candidate_ts > max_lookback_seconds:
            return None
        return candidate

    def return_over_seconds(self, seconds: int) -> float:
        points = self._snapshot()
        if len(points) < 2:
            return 0.0
        latest_ts, latest_price = points[-1]
        for ts, price in reversed(points):
            if latest_ts - ts >= seconds:
                if price <= 0:
                    return 0.0
                return (latest_price - price) / price
        first_price = points[0][1]
        if first_price <= 0:
            return 0.0
        return (latest_price - first_price) / first_price

    def realized_volatility(self, seconds: int = 300) -> float:
        points = self._snapshot()
        if len(points) < 3:
            return 0.0
        latest_ts = points[-1][0]
        window = [(ts, price) for ts, price in points if latest_ts - ts <= seconds and price > 0]
        if len(window) < 3:
            return 0.0
        returns: list[float] = []
        for i in range(1, len(window)):
            prev = window[i - 1][1]
            curr = window[i][1]
            if prev <= 0 or curr <= 0:
                continue
            returns.append(math.log(curr / prev))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(max(0.0, variance))

    def _rnjd_params(self, horizon_seconds: int) -> tuple[float, float] | None:
        points = self._snapshot()
        if len(points) < 20:
            return None

        latest_ts = points[-1][0]
        window = [(ts, px) for ts, px in points if latest_ts - ts <= 300 and px > 0]
        if len(window) < 20:
            return None

        returns: list[float] = []
        dts: list[float] = []
        for i in range(1, len(window)):
            dt = max(1e-3, window[i][0] - window[i - 1][0])
            prev = window[i - 1][1]
            curr = window[i][1]
            if prev <= 0 or curr <= 0:
                continue
            returns.append(math.log(curr / prev))
            dts.append(dt)
        if len(returns) < 10:
            return None

        scaled = [r / math.sqrt(dt) for r, dt in zip(returns, dts)]
        if not scaled:
            return None
        median_scaled = sorted(scaled)[len(scaled) // 2]
        abs_dev = [abs(x - median_scaled) for x in scaled]
        mad = sorted(abs_dev)[len(abs_dev) // 2]
        sigma_ps = max(1e-6, 1.4826 * mad)

        jump_mask: list[bool] = []
        for r, dt in zip(returns, dts):
            jump_mask.append(abs(r) > 3.0 * sigma_ps * math.sqrt(dt))

        total_time = max(1.0, sum(dts))
        jump_returns = [r for r, is_jump in zip(returns, jump_mask) if is_jump]
        cont_returns = [r for r, is_jump in zip(returns, jump_mask) if not is_jump]
        cont_dts = [dt for dt, is_jump in zip(dts, jump_mask) if not is_jump]

        lam = len(jump_returns) / total_time
        if cont_returns and cont_dts:
            mu_c = sum(cont_returns) / max(1e-6, sum(cont_dts))
            scaled_cont = [r / math.sqrt(dt) for r, dt in zip(cont_returns, cont_dts)]
            if len(scaled_cont) >= 2:
                mean_cont = sum(scaled_cont) / len(scaled_cont)
                var_c = sum((x - mean_cont) ** 2 for x in scaled_cont) / (len(scaled_cont) - 1)
            else:
                var_c = sigma_ps**2
        else:
            mu_c = 0.0
            var_c = sigma_ps**2

        if jump_returns:
            mu_j = sum(jump_returns) / len(jump_returns)
            if len(jump_returns) >= 2:
                mean_j = mu_j
                var_j = sum((x - mean_j) ** 2 for x in jump_returns) / (len(jump_returns) - 1)
            else:
                var_j = (abs(mu_j) * 0.5) ** 2
        else:
            mu_j = 0.0
            var_j = 0.0

        h = float(max(1, horizon_seconds))
        mu_h = (mu_c * h) + (lam * h * mu_j)
        var_h = max(1e-10, (var_c * h) + (lam * h * (var_j + mu_j * mu_j)))

        # Short-horizon momentum component to react faster in 5m / 15m windows.
        ret_8 = self.return_over_seconds(min(8, horizon_seconds))
        ret_20 = self.return_over_seconds(min(20, horizon_seconds))
        mu_h += 0.35 * ret_8 + 0.15 * ret_20
        return mu_h, var_h

    def rnjd_probability_above_log_return(self, horizon_seconds: int, target_log_return: float = 0.0) -> float:
        params = self._rnjd_params(horizon_seconds)
        if params is None:
            return 0.5
        mu_h, var_h = params
        z = (mu_h - target_log_return) / math.sqrt(var_h)
        return clamp(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.01, 0.99)

    def rnjd_probability(self, horizon_seconds: int) -> float:
        return self.rnjd_probability_above_log_return(horizon_seconds, 0.0)


class FeeCache:
    def __init__(self, refresh_seconds: float) -> None:
        self.refresh_seconds = refresh_seconds
        self.cache: dict[str, FeeInfo] = {}
        self._lock = threading.Lock()

    def get(self, token_id: str, clob: ClobClient) -> FeeInfo:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            current = self.cache.get(token_id)
        if current is not None:
            age = (now - current.fetched_at).total_seconds()
            if age <= self.refresh_seconds:
                return current
        fresh = clob.get_fee_rate(token_id, side="buy")
        with self._lock:
            self.cache[token_id] = fresh
        return fresh


class BotRuntime:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.storage = Storage(config.database_path)
        self.gamma = GammaClient(config.gamma_url, timeout_seconds=config.api_timeout_seconds)
        self.clob = ClobClient(config.clob_url, timeout_seconds=config.api_timeout_seconds)
        self.clob_stream = ClobMarketStream(config.clob_ws_url)
        self.spot_rest = BtcSpotClient(config.btc_spot_url, timeout_seconds=config.api_timeout_seconds)
        self.risk = RiskManager(config)
        self.fee_cache = FeeCache(config.fee_poll_interval_seconds)
        self.spot_tracker = SpotTracker()
        self.pair_learner = PairTimingLearner(self.storage)
        self.spot_stream = BtcSpotStream(
            ws_url=config.btc_spot_ws_url,
            rest_client=self.spot_rest,
            on_price=self._on_spot_tick,
        )
        self.engine_pair = PairArbEngine(config)
        self.quote_order_ids: dict[tuple[str, str, str, str], str] = {}
        self.quote_order_notional: dict[tuple[str, str, str, str], float] = {}
        self._last_ioc_submission: dict[tuple[str, ...], tuple[float, float]] = {}
        self._last_settlement_ts = 0.0
        self._last_redeem_ts = 0.0
        self._settlement_method_missing = False
        self._redeem_method_missing = False
        self._active_market_by_tf: dict[Timeframe, str] = {}
        self._cycle_counter = 0
        self._open_alpha_legs: dict[str, list[dict[str, object]]] = {}
        self._market_buy_exec_state: dict[str, dict[str, float | str | int]] = {}
        self._reference_trader = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
        self._last_reference_poll_ts = 0.0
        self._seen_reference_activity: set[str] = set()
        self._reference_offsets = (0, 30, 60, 90, 120, 150, 180, 210)
        self._reference_offset_idx = 0

        self.executor: BaseExecutor
        if config.live_mode:
            self.executor = LiveExecutor(config)
        else:
            self.executor = PaperExecutor(config)

        self._keep_running = True

    def _on_spot_tick(self, ts: float, price: float) -> None:
        self.spot_tracker.update(ts, price)

    def preflight(self) -> None:
        if self.config.live_mode:
            self.executor.preflight()

    def stop(self) -> None:
        self._keep_running = False

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

    def close(self) -> None:
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
                LOGGER.warning("cycle=%s spot_unavailable=%s; skipping cycle", self._cycle_counter, message)
                return
            raise
        self.spot_tracker.update(now.timestamp(), spot_price)
        spot_return_60s = self.spot_tracker.return_over_seconds(60)

        all_markets = self.gamma.fetch_btc_markets(self.config.enabled_tags, self.config.max_markets_per_tag)
        markets = self._select_markets(all_markets)
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
            LOGGER.warning("No snapshots to trade in current cycle")
            return

        self._record_snapshots(snapshots)
        if self.config.use_reference_trader_learning:
            self._learn_from_reference_trader(snapshots, now=now)
        marks = {
            token_id: book.mid
            for token_id, book in books.items()
            if book.mid > 0
        }
        self.risk.mark_to_market(marks)
        max_total = self.config.bankroll_usdc * self.config.max_total_exposure_pct
        max_market = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        market_exposure_text = ", ".join(
            f"{m.market_id}:{self.risk.market_exposure.get(m.market_id, 0.0):.2f}/{max_market:.2f}" for m in markets
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
            self.storage.record_risk_event("risk_halt", {"reason": self.risk.halted_reason})
            self.storage.record_risk_state(self.risk.state())
            LOGGER.error("Trading halted: %s", self.risk.halted_reason)
            if self.config.live_mode and self._is_fatal_halt_reason(self.risk.halted_reason):
                LOGGER.error("Stopping live loop after risk halt")
                self.stop()
                return

        fair_by_market = self._compute_fair_map(snapshots, spot_price)
        intents = self._generate_intents(snapshots, fair_by_market)
        executed_count, fill_count, error_count = self._execute_intents(intents, books)
        if self.risk.halted:
            self._cancel_quotes()
            self.storage.record_risk_event("risk_halt", {"reason": self.risk.halted_reason})
            self.storage.record_risk_state(self.risk.state())
            LOGGER.error("Trading halted: %s", self.risk.halted_reason)
            if self.config.live_mode and self._is_fatal_halt_reason(self.risk.halted_reason):
                LOGGER.error("Stopping live loop after risk halt")
                self.stop()
                return
        sweep_fill_count = self._process_paper_sweeps(books)
        merged_pairs, redeem_calls = self._settle_pairs_and_redeem(snapshots, now=now)
        self.storage.record_risk_state(self.risk.state())
        LOGGER.info(
            "cycle=%s snapshots=%s intents=%s executed=%s fills=%s sweep_fills=%s merged_pairs=%.2f redeem_calls=%s exec_errors=%s equity=%.2f elapsed=%.2fs",
            self._cycle_counter,
            len(snapshots),
            len(intents),
            executed_count,
            fill_count,
            sweep_fill_count,
            merged_pairs,
            redeem_calls,
            error_count,
            self.risk.current_equity(),
            time.time() - cycle_started,
        )

    def _select_markets(self, markets: list[MarketInfo]) -> list[MarketInfo]:
        now = datetime.now(tz=timezone.utc)
        open_market_ids = {
            position.market_id for position in self.risk.positions.values() if position.size > 0
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
            if (not keep_for_unwind) and seconds_to_end <= self._rollover_guard_seconds(market.timeframe):
                continue
            by_timeframe.setdefault(market.timeframe, []).append(market)

        for candidates in by_timeframe.values():
            candidates.sort(key=lambda m: m.end_time)

        cap_5m = self.config.max_trade_markets_5m
        cap_15m = self.config.max_trade_markets_15m
        cap_1h = self.config.max_trade_markets_1h
        if self.config.bankroll_usdc <= 100.0:
            # Keep small bankroll recycling on shorter tenors and
            # scan wider for dislocations.
            cap_1h = 0
            cap_15m = max(cap_15m, 6)
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
        for timeframe in list(self._active_market_by_tf.keys()):
            if timeframe not in seen_timeframes:
                self._active_market_by_tf.pop(timeframe, None)

    def _sync_quotes_with_selected_markets(self, markets: list[MarketInfo]) -> None:
        selected_ids = {market.market_id for market in markets}
        stale_keys: list[tuple[str, str, str, str]] = []
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
            self.quote_order_ids.pop(key, None)
            self.quote_order_notional.pop(key, None)
        LOGGER.info(
            "cycle=%s stale_quote_cleanup cancelled=%s remaining_open_quotes=%s",
            self._cycle_counter,
            cancelled,
            len(self.quote_order_ids),
        )

    def _build_snapshots(self, markets: list[MarketInfo]) -> tuple[list[MarketSnapshot], dict[str, OrderBookSnapshot]]:
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
            self.clob_stream.assert_healthy()
            self.clob_stream.set_assets(token_ids)
            books.update(self.clob_stream.get_books(token_ids))
            missing_token_ids = [token_id for token_id in token_ids if token_id not in books]
            if missing_token_ids:
                restored = 0
                backfill_limit = min(18, len(missing_token_ids))
                for token_id in missing_token_ids[:backfill_limit]:
                    if not self._keep_running:
                        break
                    try:
                        fallback_book = self.clob.get_book(token_id)
                    except Exception as exc:
                        LOGGER.debug("book_rest_backfill_failed token=%s error=%s", token_id, exc)
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
                LOGGER.warning("Skipping market=%s due to missing fee data", market.market_id)
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
            LOGGER.warning("cycle=%s skipped_markets_missing_books=%s", self._cycle_counter, missing_market_books)
        return snapshots, books

    def _record_snapshots(self, snapshots: list[MarketSnapshot]) -> None:
        for snapshot in snapshots:
            self.storage.record_snapshot(snapshot)

    def _horizon_seconds(self, timeframe: Timeframe) -> int:
        if timeframe == Timeframe.FIVE_MIN:
            return 300
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 900
        if timeframe == Timeframe.ONE_HOUR:
            return 3600
        return 900

    def _compute_fair_probability(self, snapshot: MarketSnapshot, spot_price: float) -> float:
        horizon = self._horizon_seconds(snapshot.market.timeframe)
        seconds_to_end = max(0.0, snapshot.market.seconds_to_end)
        ret_30 = self.spot_tracker.return_over_seconds(min(30, horizon))
        ret_120 = self.spot_tracker.return_over_seconds(min(120, horizon))
        ret_600 = self.spot_tracker.return_over_seconds(min(600, horizon))
        drift = (0.45 * ret_30) + (0.35 * ret_120) + (0.20 * ret_600)

        vol_step = self.spot_tracker.realized_volatility(seconds=300)
        sigma = max(0.0012, vol_step * math.sqrt(max(1.0, horizon / 300.0)))
        z = drift / sigma
        p_base = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        primary_imb = snapshot.primary_book.top_size_imbalance(levels=2)
        secondary_imb = snapshot.secondary_book.top_size_imbalance(levels=2)
        pressure = clamp((primary_imb - secondary_imb) * 0.5, -1.0, 1.0)
        p_orderbook = p_base + (0.10 * pressure)

        pbid = snapshot.primary_book.best_bid
        pask = snapshot.primary_book.best_ask
        sbid = snapshot.secondary_book.best_bid
        sask = snapshot.secondary_book.best_ask
        synth_bid = max(pbid, 1.0 - sask if sask > 0 else 0.0)
        synth_ask = min(pask if pask > 0 else 1.0, 1.0 - sbid if sbid > 0 else 1.0)
        synth_mid = None
        synth_spread = None
        if synth_bid <= synth_ask:
            synth_mid = (synth_bid + synth_ask) / 2.0
            synth_spread = synth_ask - synth_bid
            # Blend with market-implied value only when the synthetic spread is informative.
            if synth_spread <= 0.25:
                # Increase market-implied anchoring as expiry approaches.
                expiry_phase = clamp(1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0)
                synth_weight = 0.25 + (0.45 * expiry_phase)
                p_orderbook = (1.0 - synth_weight) * p_orderbook + synth_weight * synth_mid

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
        if spot_age > 2.0:
            rnjd_weight *= 0.5
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
                expiry_phase = clamp(1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0)
                anchor_weight = 0.45 + (0.40 * expiry_phase)
                p_blend = ((1.0 - anchor_weight) * p_blend) + (anchor_weight * p_up_anchor)

        if synth_mid is not None and synth_spread is not None and synth_spread <= 0.25:
            # Final safety clamp: near expiry, model fair cannot diverge too far
            # from executable market-implied probability.
            expiry_phase = clamp(1.0 - (seconds_to_end / max(1.0, float(horizon))), 0.0, 1.0)
            max_divergence = 0.35 - (0.27 * expiry_phase)  # 35% far out -> 8% near expiry
            p_blend = clamp(p_blend, synth_mid - max_divergence, synth_mid + max_divergence)

        return clamp(p_blend, 0.03, 0.97)

    def _compute_fair_map(self, snapshots: list[MarketSnapshot], spot_price: float) -> dict[str, float]:
        fair_map: dict[str, float] = {}
        for snapshot in snapshots:
            fair_map[snapshot.market.market_id] = self._compute_fair_probability(snapshot, spot_price)
        return fair_map

    def _learn_from_reference_trader(self, snapshots: list[MarketSnapshot], *, now: datetime) -> None:
        now_ts = now.timestamp()
        if now_ts - self._last_reference_poll_ts < 12.0:
            return
        self._last_reference_poll_ts = now_ts
        if not snapshots:
            return

        by_condition: dict[str, MarketSnapshot] = {
            str(snapshot.market.condition_id).strip().lower(): snapshot for snapshot in snapshots
        }
        if not by_condition:
            return

        offset = self._reference_offsets[self._reference_offset_idx % len(self._reference_offsets)]
        self._reference_offset_idx += 1
        try:
            payload = get_json(
                f"{self.config.data_api_url}/activity",
                params={
                    "user": self._reference_trader,
                    "limit": "30",
                    "offset": str(offset),
                },
                timeout=max(1.0, min(2.5, self.config.api_timeout_seconds)),
            )
        except Exception as exc:
            LOGGER.debug("reference_trader_poll_failed: %s", exc)
            return

        if not isinstance(payload, list):
            return

        learned = 0
        for event in payload:
            if not isinstance(event, dict):
                continue
            if str(event.get("type") or "").upper() != "TRADE":
                continue
            if str(event.get("side") or "").upper() != "BUY":
                continue
            tx_hash = str(event.get("transactionHash") or "").strip().lower()
            asset = str(event.get("asset") or "").strip().lower()
            if not tx_hash:
                continue
            seen_key = f"{tx_hash}:{asset}"
            if seen_key in self._seen_reference_activity:
                continue

            condition_id = str(event.get("conditionId") or "").strip().lower()
            snapshot = by_condition.get(condition_id)
            if snapshot is None:
                continue

            outcome = str(event.get("outcome") or "").strip().lower()
            outcome_index = event.get("outcomeIndex")
            primary_label = snapshot.market.primary_label.lower()
            secondary_label = snapshot.market.secondary_label.lower()
            entry_side = ""
            if outcome and outcome == primary_label:
                entry_side = "primary"
            elif outcome and outcome == secondary_label:
                entry_side = "secondary"
            elif outcome_index == 0:
                entry_side = "primary"
            elif outcome_index == 1:
                entry_side = "secondary"
            if entry_side not in {"primary", "secondary"}:
                continue

            try:
                entry_price = float(event.get("price") or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            if entry_price <= 0:
                continue

            opposite_book = snapshot.secondary_book if entry_side == "primary" else snapshot.primary_book
            if opposite_book.best_bid <= 0 and opposite_book.best_ask <= 0:
                continue

            # Trader data gives only one leg, so we proxy completion using a
            # conservative opposite-leg estimate near current bid + 1 tick.
            opposite_entry_proxy = max(0.01, opposite_book.best_bid + 0.01)
            pair_price_cost = entry_price + opposite_entry_proxy + self.config.directional_slippage_buffer
            if pair_price_cost > 1.025:
                # Ignore clearly expensive completions from copied flow.
                continue
            self.pair_learner.observe(
                market_id=snapshot.market.market_id,
                timeframe=snapshot.market.timeframe,
                side=entry_side,
                seconds_to_end=snapshot.market.seconds_to_end,
                pair_price_cost=pair_price_cost,
                hedge_delay_seconds=10.0,
                success=pair_price_cost <= 1.005,
                source="trader_copy",
            )
            self._seen_reference_activity.add(seen_key)
            learned += 1

        if len(self._seen_reference_activity) > 8000:
            self._seen_reference_activity = set(sorted(self._seen_reference_activity)[-4000:])
        if learned > 0:
            LOGGER.info(
                "cycle=%s trader_learn offset=%s events=%s tracked=%s",
                self._cycle_counter,
                offset,
                learned,
                len(self._seen_reference_activity),
            )

    def _settle_pairs_and_redeem(self, snapshots: list[MarketSnapshot], *, now: datetime) -> tuple[float, int]:
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
                secondary_size = secondary.size if secondary and secondary.size > 0 else 0.0
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
                            {"reason": "executor merge method unavailable", "raw": result.raw},
                        )
                        LOGGER.warning("Settlement merge unsupported by executor; merge automation disabled")
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
                        {"reason": "executor redeem method unavailable", "raw": redeem.raw},
                    )
                    LOGGER.warning("Settlement redeem unsupported by executor; redeem automation disabled")
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
        if not state:
            return None
        last_side = str(state.get("last_side") or "").strip().lower()
        if last_side not in {"primary", "secondary"}:
            return None
        run_len = int(state.get("run_len") or 0)
        if run_len >= 8:
            return None

        if last_side == "primary":
            ask = snapshot.primary_book.best_ask
            last_price = float(state.get("last_primary_price") or 0.0)
        else:
            ask = snapshot.secondary_book.best_ask
            last_price = float(state.get("last_secondary_price") or 0.0)
        if ask <= 0 or last_price <= 0:
            return None

        # Continue same-side laddering while execution stays near the
        # previous fill price; avoid forced alternation every fill.
        if ask <= (last_price + 0.015) and naked_ratio <= 0.34:
            return last_side
        return None

    def _generate_intents(
        self,
        snapshots: list[MarketSnapshot],
        fair_by_market: dict[str, float],
    ) -> list[OrderIntent]:
        if self.risk.halted:
            return []

        intents: list[OrderIntent] = []
        for snapshot in snapshots:
            fair = fair_by_market.get(snapshot.market.market_id, 0.5)
            self._log_signal(snapshot, fair)

            primary_pos = self.risk.positions.get(snapshot.market.primary_token_id)
            secondary_pos = self.risk.positions.get(snapshot.market.secondary_token_id)
            primary_inventory = primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
            secondary_inventory = secondary_pos.size if secondary_pos and secondary_pos.size > 0 else 0.0
            primary_avg_entry = primary_pos.average_price if primary_pos and primary_pos.size > 0 else 0.0
            secondary_avg_entry = secondary_pos.average_price if secondary_pos and secondary_pos.size > 0 else 0.0
            preferred_entry_side = self._preferred_execution_side(
                snapshot=snapshot,
                primary_inventory=primary_inventory,
                secondary_inventory=secondary_inventory,
            )

            learned_primary_pair_price, learned_primary_success_rate, learned_primary_samples = self.pair_learner.estimate(
                timeframe=snapshot.market.timeframe,
                side="primary",
                seconds_to_end=snapshot.market.seconds_to_end,
            )
            learned_secondary_pair_price, learned_secondary_success_rate, learned_secondary_samples = self.pair_learner.estimate(
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
                preferred_entry_side=preferred_entry_side,
                learned_primary_pair_price=learned_primary_pair_price,
                learned_primary_success_rate=learned_primary_success_rate,
                learned_primary_samples=learned_primary_samples,
                learned_secondary_pair_price=learned_secondary_pair_price,
                learned_secondary_success_rate=learned_secondary_success_rate,
                learned_secondary_samples=learned_secondary_samples,
            )
            intents.extend(market_intents)

        recycle_intents = self._generate_recycle_sell_intents(snapshots)
        intents.extend(recycle_intents)

        intents.sort(key=lambda i: i.expected_edge, reverse=True)
        max_per_cycle = max(4, int(self.config.pair_max_intents_per_cycle))
        intents = intents[:max_per_cycle]
        by_type = Counter(str(intent.metadata.get("intent_type") or "unknown") for intent in intents)
        LOGGER.info("pair_arb intents=%s breakdown=%s", len(intents), dict(by_type))
        return intents

    def _generate_recycle_sell_intents(self, snapshots: list[MarketSnapshot]) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for snapshot in snapshots:
            primary_pos = self.risk.positions.get(snapshot.market.primary_token_id)
            secondary_pos = self.risk.positions.get(snapshot.market.secondary_token_id)
            primary_size = primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
            secondary_size = secondary_pos.size if secondary_pos and secondary_pos.size > 0 else 0.0
            pair_size = min(primary_size, secondary_size)
            min_size = snapshot.market.order_min_size
            if pair_size < min_size:
                continue
            if snapshot.primary_book.best_bid <= 0 or snapshot.secondary_book.best_bid <= 0:
                continue

            primary_avg = primary_pos.average_price if primary_pos else 0.0
            secondary_avg = secondary_pos.average_price if secondary_pos else 0.0
            pair_avg_cost = primary_avg + secondary_avg
            pair_bid_value = snapshot.primary_book.best_bid + snapshot.secondary_book.best_bid
            # Recycle completed pairs only when exit bids clear our average cost.
            if pair_bid_value < (pair_avg_cost + 0.004):
                continue

            recycle_size = min(pair_size, max(min_size, pair_size * 0.35))
            if recycle_size < min_size:
                continue

            primary_price = clamp(round(snapshot.primary_book.best_bid - 0.01, 2), 0.01, snapshot.primary_book.best_bid)
            secondary_price = clamp(round(snapshot.secondary_book.best_bid - 0.01, 2), 0.01, snapshot.secondary_book.best_bid)
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

    def _generate_unwind_intents(self, snapshots: list[MarketSnapshot]) -> list[OrderIntent]:
        by_token: dict[str, tuple[MarketSnapshot, OrderBookSnapshot, str]] = {}
        for snapshot in snapshots:
            by_token[snapshot.market.primary_token_id] = (snapshot, snapshot.primary_book, "primary")
            by_token[snapshot.market.secondary_token_id] = (snapshot, snapshot.secondary_book, "secondary")

        intents: list[OrderIntent] = []
        halt_reason = self.risk.halted_reason.lower()
        daily_dd_halt = self.risk.halted and ("daily drawdown threshold exceeded" in halt_reason)
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

            market_exposure = self.risk.market_exposure.get(snapshot.market.market_id, 0.0)
            if daily_dd_halt and not near_end and market_exposure <= (market_cap * 0.35):
                continue

            if self.risk.halted:
                if daily_dd_halt:
                    size = min(position.size, max(snapshot.market.order_min_size, position.size * 0.25))
                    intent_type = "drawdown_rebalance"
                    expected_edge = 5_000.0
                else:
                    size = position.size
                    intent_type = "risk_unwind"
                    expected_edge = 10_000.0
            else:
                size = min(position.size, max(snapshot.market.order_min_size, position.size * 0.60))
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
            fair_probability - up_ask - per_share_fee(up_ask, snapshot.primary_fee.base_fee) - self.config.directional_slippage_buffer
            if up_ask > 0
            else 0.0
        )
        down_fair = 1.0 - fair_probability
        down_edge = (
            down_fair - down_ask - per_share_fee(down_ask, snapshot.secondary_fee.base_fee) - self.config.directional_slippage_buffer
            if down_ask > 0
            else 0.0
        )
        LOGGER.info(
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
        for timeframe in (Timeframe.FIVE_MIN, Timeframe.FIFTEEN_MIN, Timeframe.ONE_HOUR):
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

    def _record_alpha_fill_for_learning(self, intent: OrderIntent, result: OrderResult, now_ts: float) -> None:
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
        if not opposite_token_id or timeframe == Timeframe.UNKNOWN or entry_side not in {"primary", "secondary"}:
            return
        leg = {
            "market_id": result.market_id,
            "entry_token_id": result.token_id,
            "opposite_token_id": opposite_token_id,
            "entry_side": entry_side,
            "timeframe": timeframe.value,
            "seconds_to_end_at_entry": max(0.0, seconds_to_end),
            "entry_price": float(result.filled_price),
            "remaining_size": float(result.filled_size),
            "opened_ts": float(now_ts),
        }
        self._open_alpha_legs.setdefault(result.market_id, []).append(leg)

        state = self._market_buy_exec_state.get(result.market_id, {})
        last_side = str(state.get("last_side") or "")
        run_len = int(state.get("run_len") or 0)
        if last_side == entry_side:
            run_len += 1
        else:
            run_len = 1
        state["last_side"] = entry_side
        state["run_len"] = run_len
        if entry_side == "primary":
            state["last_primary_price"] = float(result.filled_price)
        else:
            state["last_secondary_price"] = float(result.filled_price)
        state["updated_ts"] = float(now_ts)
        self._market_buy_exec_state[result.market_id] = state

    def _match_hedge_fill_for_learning(self, intent: OrderIntent, result: OrderResult, now_ts: float) -> None:
        if result.side != Side.BUY or result.filled_size <= 0:
            return
        intent_type = str(intent.metadata.get("intent_type") or "")
        if intent_type not in {"equalize", "equalize_forced", "equalize_immediate"}:
            return

        legs = self._open_alpha_legs.get(result.market_id)
        if not legs:
            return
        remaining = float(result.filled_size)
        updated: list[dict[str, object]] = []
        for leg in sorted(legs, key=lambda item: float(item.get("opened_ts", 0.0))):
            leg_remaining = float(leg.get("remaining_size", 0.0))
            if leg_remaining <= 0:
                continue
            if remaining <= 0:
                updated.append(leg)
                continue
            opposite_token_id = str(leg.get("opposite_token_id") or "")
            if opposite_token_id != result.token_id:
                updated.append(leg)
                continue

            matched = min(remaining, leg_remaining)
            pair_price_cost = float(leg.get("entry_price", 0.0)) + float(result.filled_price)
            delay = max(0.0, now_ts - float(leg.get("opened_ts", now_ts)))
            timeframe = self._parse_timeframe(leg.get("timeframe"))
            entry_side = str(leg.get("entry_side") or "")
            seconds_to_end = float(leg.get("seconds_to_end_at_entry", 0.0))
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
                leg["remaining_size"] = leg_remaining
                updated.append(leg)
        if updated:
            self._open_alpha_legs[result.market_id] = updated
        else:
            self._open_alpha_legs.pop(result.market_id, None)

    def _expire_unhedged_alpha_legs(self, books: dict[str, OrderBookSnapshot], now_ts: float) -> None:
        for market_id, legs in list(self._open_alpha_legs.items()):
            updated: list[dict[str, object]] = []
            for leg in legs:
                leg_remaining = float(leg.get("remaining_size", 0.0))
                if leg_remaining <= 1e-9:
                    continue
                timeframe = self._parse_timeframe(leg.get("timeframe"))
                opened_ts = float(leg.get("opened_ts", now_ts))
                age = max(0.0, now_ts - opened_ts)
                timeout = self._hedge_timeout_seconds(timeframe)
                if age < timeout:
                    updated.append(leg)
                    continue

                opposite_token_id = str(leg.get("opposite_token_id") or "")
                opposite_book = books.get(opposite_token_id)
                opposite_ask = opposite_book.best_ask if opposite_book and opposite_book.best_ask > 0 else 0.99
                pair_price_cost = float(leg.get("entry_price", 0.0)) + float(opposite_ask)
                entry_side = str(leg.get("entry_side") or "")
                seconds_to_end = float(leg.get("seconds_to_end_at_entry", 0.0))
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

    def _build_immediate_equalizer(
        self,
        *,
        parent_intent: OrderIntent,
        fill: OrderResult,
        books: dict[str, OrderBookSnapshot],
    ) -> OrderIntent | None:
        if parent_intent.engine != "engine_pair_arb":
            return None
        if str(parent_intent.metadata.get("intent_type") or "") != "alpha_entry":
            return None
        if fill.side != Side.BUY or fill.filled_size <= 0:
            return None

        opposite_token_id = str(parent_intent.metadata.get("opposite_token_id") or "").strip()
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
        hedge_entry = round_tick(min(0.99, opposite_ask))
        if hedge_entry <= 0:
            return None

        fee_bps_raw = parent_intent.metadata.get("opposite_fee_bps", 0)
        try:
            fee_bps = int(fee_bps_raw)
        except (TypeError, ValueError):
            fee_bps = 0

        parent_fair = parent_intent.metadata.get("p_fair")
        if isinstance(parent_fair, (int, float)):
            opposite_fair = clamp(1.0 - float(parent_fair), 0.01, 0.99)
        else:
            opposite_fair = 0.5
        hedge_edge = (
            opposite_fair
            - opposite_ask
            - per_share_fee(opposite_ask, fee_bps)
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
        projected_pair_cost_price = fill.filled_price + hedge_entry + (2.0 * self.config.directional_slippage_buffer)
        completion_target = self.engine_pair.target_rebalance_pair_cost

        primary_pos = self.risk.positions.get(fill.token_id)
        opposite_pos = self.risk.positions.get(opposite_token_id)
        primary_size = primary_pos.size if primary_pos and primary_pos.size > 0 else 0.0
        opposite_size = opposite_pos.size if opposite_pos and opposite_pos.size > 0 else 0.0
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
        if (not must_reduce_naked) and projected_pair_cost_price > immediate_price_gate:
            return None

        if projected_pair_cost_price <= completion_target:
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
        hedge_size = min(max(min_size, fill.filled_size * ladder_clip), fill.filled_size)
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
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=60.0,
            metadata={
                "strategy": "pair-arb",
                "intent_type": "equalize_immediate",
                "picked_side": opposite_label,
                "p_fair": opposite_fair,
                "edge_net": hedge_edge,
                "best_ask": opposite_ask,
                "fee_bps": fee_bps,
                "order_min_size": min_size,
                "parent_order_id": fill.order_id,
                "parent_token_id": fill.token_id,
                "parent_fill_size": fill.filled_size,
                "ladder_clip": ladder_clip,
                "must_reduce_naked": must_reduce_naked,
                "naked_ratio": naked_ratio,
                "marginal_pair_cost": projected_pair_cost,
                "marginal_pair_cost_price": projected_pair_cost_price,
                "target_pair_cost": completion_target,
            },
        )

    def _execute_intents(
        self,
        intents: list[OrderIntent],
        books: dict[str, OrderBookSnapshot],
    ) -> tuple[int, int, int]:
        executed = 0
        fills = 0
        errors = 0
        planned_ioc_total = 0.0
        planned_ioc_market: dict[str, float] = {}
        pending_intents = list(intents)
        intents_by_engine: Counter[str] = Counter(i.engine for i in pending_intents)
        acked_by_engine: Counter[str] = Counter()
        rejected_by_reason: Counter[str] = Counter()
        intent_idx = 0
        while intent_idx < len(pending_intents):
            intent = pending_intents[intent_idx]
            intent_idx += 1
            if not self._keep_running:
                break
            intent_type = str(intent.metadata.get("intent_type") or "alpha")
            is_rebalance_buy = (
                intent.side == Side.BUY
                and intent_type in {"equalize", "equalize_forced", "equalize_immediate", "pair_completion"}
            )
            effective_bankroll = max(self.config.bankroll_usdc, self.risk.current_equity())
            key: tuple[str, str, str, str] | None = None
            if intent.post_only:
                key = (intent.engine, intent.market_id, intent.token_id, intent.side.value)
                if key in self.quote_order_ids:
                    rejected_by_reason["already_quoted"] += 1
                    continue

                # Guardrail: don't send post-only orders that already cross the local book.
                book = books.get(intent.token_id)
                if book:
                    if intent.side.value == "buy" and book.best_ask > 0 and intent.price >= book.best_ask:
                        self.storage.record_risk_event(
                            "order_skip_cross_local",
                            {"market_id": intent.market_id, "engine": intent.engine, "side": intent.side.value},
                        )
                        rejected_by_reason["skip_cross_local"] += 1
                        continue
                    if intent.side.value == "sell" and book.best_bid > 0 and intent.price <= book.best_bid:
                        self.storage.record_risk_event(
                            "order_skip_cross_local",
                            {"market_id": intent.market_id, "engine": intent.engine, "side": intent.side.value},
                        )
                        rejected_by_reason["skip_cross_local"] += 1
                        continue

                # Reserve quote budget so live placement respects available balance, not just filled positions.
                max_total = effective_bankroll * self.config.max_total_exposure_pct
                projected_total = self.risk.total_exposure() + self._reserved_notional_total() + intent.notional
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

            if not intent.post_only and intent.side == Side.BUY and not is_rebalance_buy:
                reserve_cash = 0.0 if self.config.bankroll_usdc <= 50.0 else max(0.0, self.config.bankroll_usdc * 0.05)
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
                    sized_size = max(min_size, (available_alpha_cash / max(intent.price, 0.01)) * 0.995)
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
                            pair_unit_cost = max(0.01, intent.price) + max(0.01, opposite_entry)
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

            if self.config.live_mode and intent.side == Side.BUY and intent.notional < 1.0:
                min_size = float(intent.metadata.get("order_min_size", 5.0))
                min_notional_target = 1.0
                sized_size = max(min_size, (min_notional_target / max(intent.price, 0.01)) * 1.01)
                if sized_size > intent.size:
                    intent = replace(intent, size=sized_size)

            if self.config.live_mode and intent.side == Side.BUY and intent.notional < 1.0:
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

            decision = self.risk.can_place(intent)
            if not decision.allowed and intent.side == Side.BUY and not intent.post_only:
                # Fast path for IOC alpha: scale down to remaining cap headroom instead of hard reject spam.
                reason = (decision.reason or "").lower()
                if "cash budget exceeded" in reason or is_rebalance_buy:
                    remaining_notional = max(0.0, self.risk.cash)
                else:
                    max_total = effective_bankroll * self.config.max_total_exposure_pct
                    max_market = effective_bankroll * self.config.max_market_exposure_pct
                    remaining_total = max_total - self.risk.total_exposure() - planned_ioc_total
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
                        sized_size = max(min_size, remaining_notional / max(intent.price, 0.01))
                        # Keep below cap boundary to avoid float edge rejects.
                        sized_size = max(min_size, sized_size * 0.995)
                        intent = replace(intent, size=sized_size)
                        decision = self.risk.can_place(intent)

            if not decision.allowed:
                reason = decision.reason or "risk rejection"
                self.storage.record_risk_event(
                    "risk_reject",
                    {"market_id": intent.market_id, "engine": intent.engine, "reason": reason},
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
                ioc_key = (intent.engine, intent.market_id, intent.token_id, intent.side.value, intent_type)
                cooldown_seconds = 0.20 if intent.engine == "engine_pair_arb" else 1.0
                now_ts = time.time()
                prev = self._last_ioc_submission.get(ioc_key)
                if cooldown_seconds > 0 and prev and (now_ts - prev[0]) < cooldown_seconds and abs(prev[1] - intent.price) < 1e-9:
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
                planned_ioc_market[intent.market_id] = planned_ioc_market.get(intent.market_id, 0.0) + intent.notional
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
                        {"order_id": result.order_id, "market_id": intent.market_id, "engine": intent.engine, "raw": result.raw},
                    )
                    # This is expected for stale post-only quotes; don't accumulate execution-failure state.
                    self.risk.on_execution_success()
                    rejected_by_reason["reject_cross_exchange"] += 1
                    continue

                errors += 1
                self.storage.record_risk_event("execution_error", {"order_id": result.order_id, "raw": result.raw})
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
                    self.risk.halt("not enough balance / allowance")
                    self.storage.record_risk_event(
                        "funding_error",
                        {
                            "order_id": result.order_id,
                            "message": "insufficient available balance/allowance for submitted live orders",
                            "raw": result.raw,
                        },
                    )
                    break
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
            else:
                self.risk.on_execution_success()
                acked_by_engine[intent.engine] += 1
            if result.is_filled:
                fills += 1
                self.risk.apply_fill(result)
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

            if intent.post_only and result.status not in {"rejected", "error", "filled", "cancelled", "canceled"} and key:
                self.quote_order_ids[key] = result.order_id
                self.quote_order_notional[key] = intent.notional
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
        return fills

    def _cancel_quotes(self) -> None:
        self.executor.cancel_all()
        self.quote_order_ids.clear()
        self.quote_order_notional.clear()

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


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "web3", "websocket"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


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
    LOGGER.info("Starting bot mode=%s tags=%s", config.mode, ",".join(str(x) for x in config.enabled_tags))
    signal_count = {"count": 0}

    def _handle_signal(signum: int, _frame: object) -> None:
        signal_count["count"] += 1
        if signal_count["count"] >= 2:
            LOGGER.error("Received signal %s again, forcing exit now", signum)
            raise SystemExit(130)
        LOGGER.warning("Received signal %s, stopping loop (press Ctrl+C again to force-exit)", signum)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polymarket_bot", description="BTC tenor trading bot")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run trading loop")
    run.add_argument("--mode", choices=("paper", "live"), default=None)
    run.add_argument("--bankroll", type=float, default=None, help="Starting bankroll in USDC (e.g. 100)")
    run.set_defaults(func=_run_command)

    report = sub.add_parser("report", help="Print PnL/report summary from SQLite")
    report.add_argument("--window", type=int, default=24, help="Window in hours")
    report.set_defaults(func=_report_command)
    return parser


def cli(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import MarketSnapshot, OrderBookSnapshot, OrderIntent, Side, TimeInForce, Timeframe
from polymarket_bot.pricing import clamp, per_share_fee, round_tick


@dataclass
class PairArbEngine:
    config: BotConfig

    # Complete-set style: prioritize buying both sides cheaply, then merge/redeem.
    absurd_pair_cost_guard: float = 1.20
    max_completion_cost: float = 1.0450
    max_entry_projected_completion_cost: float = 1.0350
    max_entry_projected_completion_cost_all_in: float = 1.0600
    max_entry_pair_cost: float = 1.0200
    target_rebalance_pair_cost: float = 1.0200
    projected_opposite_improvement_cap: float = 0.0300
    projected_opposite_improvement_frac: float = 0.50
    alpha_edge_min: float = 0.0030
    target_naked_ratio: float = 0.18
    hard_naked_ratio: float = 0.42
    rebalance_ladder_soft_step_1: float = 0.0100
    rebalance_ladder_soft_step_2: float = 0.0200
    rebalance_ladder_soft_clip_1: float = 0.45
    rebalance_ladder_soft_clip_2: float = 0.25
    maker_entry_tick: float = 0.0100
    maker_completion_slack: float = 0.0100
    rolling_pair_cost_target: float = 1.0000
    rolling_pair_cost_base_allowance: float = 0.0080
    rolling_pair_cost_warmup_allowance: float = 0.0120
    rolling_pair_cost_max_allowance: float = 0.0250
    rolling_pair_cost_credit_multiplier: float = 1.8
    rolling_pair_cost_debt_penalty: float = 3.2
    rolling_pair_cost_warmup_samples: int = 20
    fast_iteration_bankroll_threshold: float = 150.0
    fast_iteration_notional_scale: float = 0.58
    fast_iteration_pair_notional_cap: float = 4.50
    fast_iteration_alpha_notional_cap: float = 3.25
    fast_iteration_refresh_scale: float = 0.62
    fast_iteration_max_age_scale: float = 0.70
    fast_iteration_hold_refresh_multiplier: float = 1.12
    fast_iteration_hold_max_age_multiplier: float = 1.28
    fast_iteration_hold_dwell_scale: float = 0.65
    fair_bias_threshold: float = 0.0100
    fair_confidence_min: float = 0.18
    fair_confidence_max: float = 0.92

    @staticmethod
    def _opening_window_seconds(timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            return 90.0
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 180.0
        if timeframe == Timeframe.ONE_HOUR:
            return 600.0
        return 120.0

    def _execution_phase(self, *, timeframe: Timeframe, now: datetime, start_time: datetime, seconds_to_end: float) -> str:
        elapsed = max(0.0, (now - start_time).total_seconds())
        if seconds_to_end <= (self._force_equalizer_seconds(timeframe) + 20.0):
            return "late"
        if elapsed <= self._opening_window_seconds(timeframe):
            return "open"
        return "middle"

    def _fast_iteration_mode(self) -> bool:
        return self.config.bankroll_usdc >= self.fast_iteration_bankroll_threshold

    def _quote_refresh_seconds(self, timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            base = 1.9
        elif timeframe == Timeframe.FIFTEEN_MIN:
            base = 4.0
        elif timeframe == Timeframe.ONE_HOUR:
            base = 8.0
        else:
            base = 4.0
        if self._fast_iteration_mode():
            base *= self.fast_iteration_refresh_scale
        return max(0.25, base)

    def _quote_max_age_seconds(self, timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            base = 6.8
        elif timeframe == Timeframe.FIFTEEN_MIN:
            base = 14.0
        elif timeframe == Timeframe.ONE_HOUR:
            base = 26.0
        else:
            base = 12.0
        if self._fast_iteration_mode():
            base *= self.fast_iteration_max_age_scale
        return max(0.6, base)

    def _pair_entry_cost_cap(self, timeframe: Timeframe, phase: str) -> float:
        base = self.max_entry_pair_cost
        if timeframe == Timeframe.FIVE_MIN:
            if phase == "open":
                return min(base, 1.0120)
            if phase == "middle":
                return min(base, 1.0180)
            return min(base, 1.0220)
        if timeframe == Timeframe.FIFTEEN_MIN:
            if phase == "open":
                return min(base, 1.0150)
            if phase == "middle":
                return min(base, 1.0200)
            return min(base, 1.0240)
        if timeframe == Timeframe.ONE_HOUR:
            if phase == "open":
                return min(base, 1.0180)
            if phase == "middle":
                return min(base, 1.0240)
            return min(base, 1.0300)
        return base

    def _rebalance_pair_cost_cap(self, timeframe: Timeframe, phase: str) -> float:
        base = self.target_rebalance_pair_cost
        if timeframe == Timeframe.FIVE_MIN:
            if phase == "open":
                return min(base, 1.0160)
            if phase == "middle":
                return min(base, 1.0200)
            return min(base, 1.0250)
        if timeframe == Timeframe.FIFTEEN_MIN:
            if phase == "open":
                return min(base, 1.0180)
            if phase == "middle":
                return min(base, 1.0220)
            return min(base, 1.0280)
        if timeframe == Timeframe.ONE_HOUR:
            if phase == "open":
                return min(base, 1.0200)
            if phase == "middle":
                return min(base, 1.0260)
            return min(base, 1.0320)
        return base

    def _time_guard_seconds(self, timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            if self.config.single_5m_deep_mode:
                # In focused 5m mode we want to keep trading deeper into the
                # window instead of shutting down too early.
                return 25.0
            return float(self.config.directional_min_time_left_5m)
        if timeframe == Timeframe.FIFTEEN_MIN:
            return float(self.config.directional_min_time_left_15m)
        if timeframe == Timeframe.ONE_HOUR:
            return 240.0
        return 30.0

    def _alpha_entry_guard_seconds(self, timeframe: Timeframe) -> float:
        # Alpha entries need more runway than generic order activity so that
        # the follow-up equalization is not forced into a near-expiry scramble.
        base = self._time_guard_seconds(timeframe)
        small_bankroll = self.config.bankroll_usdc <= 50.0
        if timeframe == Timeframe.FIVE_MIN:
            if self.config.single_5m_deep_mode:
                # Keep pair-entry quoting active longer in deep mode.
                return max(base, 45.0)
            if small_bankroll:
                return max(55.0, float(self.config.directional_min_time_left_5m) * 0.60)
            # 100s was too conservative for 5m windows and caused idle cycles
            # in otherwise tradable intervals.
            return max(base, 70.0)
        if timeframe == Timeframe.FIFTEEN_MIN:
            if small_bankroll:
                return max(90.0, float(self.config.directional_min_time_left_15m) * 0.70)
            return max(base, 180.0)
        if timeframe == Timeframe.ONE_HOUR:
            if small_bankroll:
                return max(base, 300.0)
            return max(base, 420.0)
        return base

    @staticmethod
    def _force_equalizer_seconds(timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
            return 45.0
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 120.0
        if timeframe == Timeframe.ONE_HOUR:
            return 300.0
        return 60.0

    @staticmethod
    def _project_average_price(current_size: float, current_avg: float, add_size: float, add_price: float) -> float:
        if add_size <= 0:
            return max(0.0, current_avg)
        total_size = max(0.0, current_size) + add_size
        if total_size <= 0:
            return 0.0
        total_cost = (max(0.0, current_size) * max(0.0, current_avg)) + (add_size * max(0.0, add_price))
        return total_cost / total_size

    @staticmethod
    def _entry_from_ask(ask: float) -> float:
        if ask <= 0:
            return 0.0
        return round_tick(min(0.99, ask))

    @staticmethod
    def _size_to_lot(size: float, min_size: float) -> float:
        step = max(0.0, float(min_size))
        if step <= 0:
            return max(0.0, float(size))
        steps = int(max(0.0, float(size)) / step)
        if steps <= 0:
            return step
        return steps * step

    def _maker_entry_from_book(self, book: OrderBookSnapshot) -> float:
        ask = book.best_ask
        bid = book.best_bid
        if ask <= 0:
            return 0.0
        if ask <= self.maker_entry_tick:
            return 0.0
        target = ask - self.maker_entry_tick
        if bid > 0:
            target = min(target, bid + self.maker_entry_tick)
            target = max(target, bid)
        if target <= 0:
            return 0.0
        return round_tick(clamp(target, 0.01, 0.99))

    def _maker_touch_entry(self, book: OrderBookSnapshot) -> float:
        ask = book.best_ask
        bid = book.best_bid
        if ask <= 0 or ask <= self.maker_entry_tick:
            return 0.0
        target = ask - self.maker_entry_tick
        if bid > 0:
            target = max(target, bid)
        if target <= 0:
            return 0.0
        return round_tick(clamp(target, 0.01, 0.99))

    @staticmethod
    def _max_ladder_levels(timeframe: Timeframe, phase: str) -> int:
        if phase == "late":
            return 1
        if timeframe == Timeframe.FIVE_MIN:
            return 6 if phase == "open" else 5
        if timeframe == Timeframe.FIFTEEN_MIN:
            return 4 if phase == "open" else 3
        if timeframe == Timeframe.ONE_HOUR:
            return 3
        return 2

    @staticmethod
    def _motion_metric(motion: dict[str, float] | None, key: str) -> float:
        if not motion:
            return 0.0
        raw = motion.get(key, 0.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _synthetic_market_implied_primary(snapshot: MarketSnapshot) -> tuple[float | None, float | None]:
        pbid = float(snapshot.primary_book.best_bid)
        pask = float(snapshot.primary_book.best_ask)
        sbid = float(snapshot.secondary_book.best_bid)
        sask = float(snapshot.secondary_book.best_ask)
        synth_bid = max(pbid, 1.0 - sask if sask > 0 else 0.0)
        synth_ask = min(pask if pask > 0 else 1.0, 1.0 - sbid if sbid > 0 else 1.0)
        if synth_bid <= synth_ask:
            return (synth_bid + synth_ask) / 2.0, max(0.0, synth_ask - synth_bid)
        return None, None

    def _fair_value_signal(
        self,
        *,
        snapshot: MarketSnapshot,
        fair_up: float,
        primary_motion: dict[str, float] | None,
        secondary_motion: dict[str, float] | None,
        fluctuation_regime: dict[str, float],
    ) -> dict[str, float | str | None]:
        synth_mid, synth_spread = self._synthetic_market_implied_primary(snapshot)
        if synth_mid is None:
            implied_up = clamp(snapshot.primary_book.mid, 0.01, 0.99)
            spread_quality = 0.25
        else:
            implied_up = clamp(synth_mid, 0.01, 0.99)
            spread_quality = clamp((0.18 - (synth_spread or 0.18)) / 0.18, 0.0, 1.0)

        fair_dislocation = fair_up - implied_up
        flip_rate = max(
            self._motion_metric(primary_motion, "ask_flip_rate"),
            self._motion_metric(primary_motion, "mid_flip_rate"),
            self._motion_metric(secondary_motion, "ask_flip_rate"),
            self._motion_metric(secondary_motion, "mid_flip_rate"),
            float(fluctuation_regime.get("flip_rate_max", 0.0)),
        )
        swing_short = max(
            self._motion_metric(primary_motion, "ask_swing_short"),
            self._motion_metric(primary_motion, "mid_swing_short"),
            self._motion_metric(secondary_motion, "ask_swing_short"),
            self._motion_metric(secondary_motion, "mid_swing_short"),
            float(fluctuation_regime.get("swing_short", 0.0)),
        )
        turbulence_penalty = clamp(max(0.0, flip_rate - 0.08) * 2.8, 0.0, 0.40) + clamp(
            max(0.0, swing_short - 0.015) * 8.0, 0.0, 0.35
        )
        confidence = clamp(
            0.35 + (0.50 * spread_quality) - turbulence_penalty,
            self.fair_confidence_min,
            self.fair_confidence_max,
        )

        bias_side: str | None = None
        if fair_dislocation >= self.fair_bias_threshold:
            bias_side = "primary"
        elif fair_dislocation <= -self.fair_bias_threshold:
            bias_side = "secondary"

        return {
            "implied_up": implied_up,
            "dislocation": fair_dislocation,
            "confidence": confidence,
            "bias_side": bias_side,
            "spread_quality": spread_quality,
        }

    @staticmethod
    def _pair_ladder_index_plan(
        *,
        primary_levels: int,
        secondary_levels: int,
        anticipatory: bool,
        extreme: bool,
    ) -> list[tuple[int, int]]:
        max_levels = min(max(0, primary_levels), max(0, secondary_levels))
        if max_levels <= 0:
            return []

        plan: list[tuple[int, int]] = []
        depth_cap = min(max_levels, 6 if extreme else 5)
        max_offset = 2 if extreme else 1
        for level in range(max_levels):
            plan.append((level, level))
            if not anticipatory or max_levels < 2 or level >= depth_cap:
                continue
            for offset in range(1, max_offset + 1):
                paired = level + offset
                if paired >= depth_cap:
                    continue
                plan.append((level, paired))
                plan.append((paired, level))

        unique: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in plan:
            if pair in seen:
                continue
            seen.add(pair)
            unique.append(pair)
        return unique

    def _fluctuation_regime(
        self,
        *,
        timeframe: Timeframe,
        phase: str,
        primary_motion: dict[str, float] | None,
        secondary_motion: dict[str, float] | None,
    ) -> dict[str, float]:
        primary_flip = max(
            self._motion_metric(primary_motion, "ask_flip_rate"),
            self._motion_metric(primary_motion, "mid_flip_rate"),
        )
        secondary_flip = max(
            self._motion_metric(secondary_motion, "ask_flip_rate"),
            self._motion_metric(secondary_motion, "mid_flip_rate"),
        )
        flip_rate_max = max(primary_flip, secondary_flip)
        swing_short = max(
            self._motion_metric(primary_motion, "ask_swing_short"),
            self._motion_metric(secondary_motion, "ask_swing_short"),
            self._motion_metric(primary_motion, "mid_swing_short"),
            self._motion_metric(secondary_motion, "mid_swing_short"),
        )
        swing_long = max(
            self._motion_metric(primary_motion, "ask_swing"),
            self._motion_metric(secondary_motion, "ask_swing"),
        )

        if timeframe == Timeframe.FIVE_MIN:
            fast = flip_rate_max >= 0.08 or swing_short >= 0.012
            extreme = flip_rate_max >= 0.14 or swing_short >= 0.022
            frenzy = flip_rate_max >= 0.22 or swing_short >= 0.032
            min_dwell = 3.0
        elif timeframe == Timeframe.FIFTEEN_MIN:
            fast = flip_rate_max >= 0.06 and swing_short >= 0.020
            extreme = flip_rate_max >= 0.12 or swing_short >= 0.040
            frenzy = False
            min_dwell = 4.5
        else:
            fast = flip_rate_max >= 0.04 and swing_short >= 0.025
            extreme = flip_rate_max >= 0.08 or swing_short >= 0.050
            frenzy = False
            min_dwell = 6.0

        hold_queue = phase in {"open", "middle"} and (fast or swing_long >= 0.020)
        if not hold_queue:
            min_dwell = 0.0
        elif extreme:
            min_dwell += 1.4
            if frenzy:
                min_dwell += 1.0

        extra_levels = 0.0
        if fast:
            extra_levels = 1.0
        if extreme:
            extra_levels = 2.0
        if frenzy:
            extra_levels = 3.0

        if timeframe == Timeframe.FIVE_MIN:
            volatility_score = clamp(
                max(
                    flip_rate_max / 0.24,
                    swing_short / 0.035,
                    swing_long / 0.080,
                ),
                0.0,
                1.0,
            )
        elif timeframe == Timeframe.FIFTEEN_MIN:
            volatility_score = clamp(
                max(
                    flip_rate_max / 0.16,
                    swing_short / 0.050,
                    swing_long / 0.120,
                ),
                0.0,
                1.0,
            )
        else:
            volatility_score = clamp(
                max(
                    flip_rate_max / 0.10,
                    swing_short / 0.060,
                    swing_long / 0.150,
                ),
                0.0,
                1.0,
            )

        return {
            "flip_rate_max": flip_rate_max,
            "swing_short": swing_short,
            "swing_long": swing_long,
            "hold_queue": 1.0 if hold_queue else 0.0,
            "min_dwell_seconds": min_dwell,
            "extra_levels": extra_levels,
            "volatility_score": volatility_score,
            "frenzy": 1.0 if frenzy else 0.0,
        }

    def _pair_cost_governor_cap(
        self,
        *,
        timeframe: Timeframe,
        phase: str,
        rolling_pair_cost_avg: float | None,
        rolling_pair_cost_samples: int,
    ) -> float:
        if rolling_pair_cost_avg is None or rolling_pair_cost_samples <= 0:
            allowance = self.rolling_pair_cost_warmup_allowance
        elif rolling_pair_cost_samples < self.rolling_pair_cost_warmup_samples:
            allowance = self.rolling_pair_cost_warmup_allowance * 0.90
        else:
            credit = max(0.0, self.rolling_pair_cost_target - rolling_pair_cost_avg)
            debt = max(0.0, rolling_pair_cost_avg - self.rolling_pair_cost_target)
            allowance = (
                self.rolling_pair_cost_base_allowance
                + (credit * self.rolling_pair_cost_credit_multiplier)
                - (debt * self.rolling_pair_cost_debt_penalty)
            )
        if timeframe == Timeframe.FIVE_MIN and phase == "open":
            allowance += 0.002
        if phase == "late":
            allowance -= 0.002
        allowance = clamp(allowance, 0.001, self.rolling_pair_cost_max_allowance)
        return 1.0 + allowance

    def _volatility_thrive_cap_bonus(
        self,
        *,
        timeframe: Timeframe,
        phase: str,
        fluctuation_regime: dict[str, float],
        rolling_pair_cost_avg: float | None,
        rolling_pair_cost_samples: int,
        deep_mode_5m: bool,
    ) -> tuple[float, float]:
        if timeframe != Timeframe.FIVE_MIN or phase not in {"open", "middle"}:
            return 0.0, 0.0

        volatility_score = clamp(
            float(fluctuation_regime.get("volatility_score", 0.0)),
            0.0,
            1.0,
        )
        flip_rate = float(fluctuation_regime.get("flip_rate_max", 0.0))
        swing_short = float(fluctuation_regime.get("swing_short", 0.0))
        if volatility_score < 0.30 and flip_rate < 0.08 and swing_short < 0.012:
            return 0.0, 0.0

        # Keep the rolling-average governor in control. If we are already paying
        # up on average, do not widen caps further.
        if (
            rolling_pair_cost_avg is not None
            and rolling_pair_cost_samples >= self.rolling_pair_cost_warmup_samples
            and rolling_pair_cost_avg > 1.004
        ):
            return 0.0, 0.0

        entry_bonus = 0.003 + (0.009 * volatility_score)
        governor_bonus = 0.002 + (0.007 * volatility_score)
        if rolling_pair_cost_avg is None or rolling_pair_cost_samples < self.rolling_pair_cost_warmup_samples:
            entry_bonus *= 0.85
            governor_bonus *= 0.85
        elif rolling_pair_cost_avg <= 0.995:
            entry_bonus += 0.003
            governor_bonus += 0.002
        elif rolling_pair_cost_avg >= 1.002:
            entry_bonus *= 0.60
            governor_bonus *= 0.60

        if deep_mode_5m:
            # Deep mode is where we want volatility capture to be strongest.
            # Keep bonuses somewhat moderated, but do not suppress them.
            entry_bonus *= 0.90
            governor_bonus *= 0.92

        return (
            clamp(entry_bonus, 0.0, 0.018),
            clamp(governor_bonus, 0.0, 0.015),
        )

    def _deep_mode_governor_ceiling(self, rolling_pair_cost_avg: float | None) -> float:
        # Keep deep 5m trading active, but tighten quickly as rolling average drifts
        # above parity.
        if rolling_pair_cost_avg is None:
            return 1.015
        if rolling_pair_cost_avg > 1.006:
            return 1.006
        if rolling_pair_cost_avg > 1.004:
            return 1.008
        if rolling_pair_cost_avg > 1.002:
            return 1.010
        if rolling_pair_cost_avg > 1.000:
            return 1.012
        return 1.015

    def _deep_mode_pair_entry_cap(self, *, pair_entry_cap: float, governor_cap: float) -> float:
        # In deep mode, keep price-cost entry cap aligned with the all-in
        # governor so we do not over-filter viable maker levels before the
        # governor check runs.
        bounded = clamp(governor_cap - 0.0015, 1.006, 1.018)
        return clamp(max(pair_entry_cap, bounded), 1.006, 1.018)

    @staticmethod
    def _deep_mode_equalizer_all_in_cap(*, forced: bool, seconds_to_end: float) -> float:
        if not forced:
            return 1.012
        if seconds_to_end > 20.0:
            return 1.018
        return 1.024

    def _maker_ladder_from_book(
        self,
        *,
        book: OrderBookSnapshot,
        timeframe: Timeframe,
        phase: str,
        motion: dict[str, float] | None = None,
        extra_levels: int = 0,
    ) -> list[float]:
        tick = max(0.01, self.maker_entry_tick)
        top = self._maker_touch_entry(book)
        if top <= 0:
            return []

        ask = float(book.best_ask)
        bid = float(book.best_bid)
        max_levels_base = self._max_ladder_levels(timeframe, phase)
        extra_levels_safe = max(0, int(extra_levels))
        max_levels = max_levels_base + extra_levels_safe
        spread_ticks = 1
        if ask > 0 and bid > 0 and ask > bid:
            spread_ticks = max(1, int(round((ask - bid) / tick)))
        levels_target = 1
        if spread_ticks >= 3:
            levels_target = min(max_levels, spread_ticks - 1)
        if phase in {"open", "middle"}:
            # During fast/imbalanced books, keep extra resting depth one-to-two
            # ticks below touch so fills can "fall into" queued levels.
            imbalance = abs(book.top_size_imbalance(levels=2))
            if imbalance >= 0.35:
                levels_target = max(levels_target, 2)
            if imbalance >= 0.55:
                levels_target = min(max_levels, max(levels_target, 3))
            if timeframe == Timeframe.FIVE_MIN:
                ask_swing = max(0.0, self._motion_metric(motion, "ask_swing"))
                sample_count = int(max(0.0, self._motion_metric(motion, "samples")))
                if sample_count >= 6 and ask_swing >= (2.0 * tick):
                    swing_levels = int(round(ask_swing / tick)) + 1
                    # Allow two extra levels in 5m fast swings so prints can
                    # fall into our resting queue instead of missing us by one tick.
                    swing_max_levels = max_levels + 2
                    levels_target = max(levels_target, min(swing_max_levels, swing_levels))
                ask_flip_rate = max(
                    self._motion_metric(motion, "ask_flip_rate"),
                    self._motion_metric(motion, "mid_flip_rate"),
                )
                swing_short = max(
                    self._motion_metric(motion, "ask_swing_short"),
                    self._motion_metric(motion, "mid_swing_short"),
                )
                if ask_flip_rate >= 0.10 and swing_short >= (1.5 * tick):
                    levels_target = min(max_levels + 1, max(levels_target, 3))
                if ask_flip_rate >= 0.18 and swing_short >= (2.5 * tick):
                    levels_target = min(max_levels + 2, max(levels_target, 4))

        if bid > 0:
            lower_bound = max(0.01, round_tick(bid - tick))
        else:
            lower_bound = max(0.01, round_tick(top - ((levels_target - 1) * tick)))
        observed_ask_low = self._motion_metric(motion, "ask_low")
        if observed_ask_low > 0 and timeframe == Timeframe.FIVE_MIN and phase in {"open", "middle"}:
            lower_bound = min(lower_bound, round_tick(max(0.01, observed_ask_low - tick)))
        # Keep the ladder bounded so we do not rest far below active flow.
        max_depth_ticks = max_levels + (2 if timeframe == Timeframe.FIVE_MIN else 1)
        floor_guard = round_tick(max(0.01, top - (max_depth_ticks * tick)))
        lower_bound = max(lower_bound, floor_guard)

        ladder: list[float] = []
        for idx in range(levels_target):
            price = round_tick(clamp(top - (idx * tick), 0.01, 0.99))
            if ask > 0 and price >= ask:
                price = round_tick(max(0.01, ask - tick))
            if price < (lower_bound - 1e-9):
                break
            if price <= 0:
                continue
            if ladder and abs(ladder[-1] - price) < 0.0095:
                continue
            ladder.append(price)
        if not ladder:
            return [top]
        return ladder

    def _volatility_capture_fraction(
        self,
        *,
        timeframe: Timeframe,
        phase: str,
        primary_motion: dict[str, float] | None,
        secondary_motion: dict[str, float] | None,
    ) -> float:
        if timeframe != Timeframe.FIVE_MIN:
            if timeframe == Timeframe.FIFTEEN_MIN:
                return 0.30 if phase == "open" else 0.25
            if timeframe == Timeframe.ONE_HOUR:
                return 0.20
            return 0.25

        if phase == "open":
            base = 0.55
        elif phase == "middle":
            base = 0.45
        else:
            base = 0.25

        primary_swing = max(0.0, self._motion_metric(primary_motion, "ask_swing"))
        secondary_swing = max(0.0, self._motion_metric(secondary_motion, "ask_swing"))
        swing = max(primary_swing, secondary_swing)
        flip_rate = max(
            self._motion_metric(primary_motion, "ask_flip_rate"),
            self._motion_metric(primary_motion, "mid_flip_rate"),
            self._motion_metric(secondary_motion, "ask_flip_rate"),
            self._motion_metric(secondary_motion, "mid_flip_rate"),
        )
        swing_short = max(
            self._motion_metric(primary_motion, "ask_swing_short"),
            self._motion_metric(primary_motion, "mid_swing_short"),
            self._motion_metric(secondary_motion, "ask_swing_short"),
            self._motion_metric(secondary_motion, "mid_swing_short"),
        )
        if swing >= 0.03:
            base += 0.10
        if swing >= 0.05:
            base += 0.10
        if flip_rate >= 0.10:
            base += 0.08
        if flip_rate >= 0.18:
            base += 0.07
        if swing_short >= 0.02:
            base += 0.06
        return clamp(base, 0.20, 0.75)

    def _project_opposite_entry(self, opposite_ask: float, opposite_fair: float) -> float:
        if opposite_ask <= 0:
            return 0.0
        improvement_room = max(0.0, opposite_ask - clamp(opposite_fair, 0.01, 0.99))
        expected_improvement = min(
            self.projected_opposite_improvement_cap,
            improvement_room * self.projected_opposite_improvement_frac,
        )
        return round_tick(clamp(opposite_ask - expected_improvement, 0.01, 0.99))

    def _alpha_edge_floor(
        self,
        *,
        side: str,
        learned_primary_pair_price: float | None,
        learned_primary_success_rate: float | None,
        learned_primary_samples: int,
        learned_secondary_pair_price: float | None,
        learned_secondary_success_rate: float | None,
        learned_secondary_samples: int,
    ) -> float:
        edge_floor = self.alpha_edge_min
        if side == "primary":
            if (
                learned_primary_pair_price is not None
                and learned_primary_pair_price <= self.max_entry_projected_completion_cost
                and (learned_primary_success_rate or 0.0) >= 0.58
                and learned_primary_samples >= 12
            ):
                return edge_floor * 0.50
            return edge_floor
        if (
            learned_secondary_pair_price is not None
            and learned_secondary_pair_price <= self.max_entry_projected_completion_cost
            and (learned_secondary_success_rate or 0.0) >= 0.58
            and learned_secondary_samples >= 12
        ):
            return edge_floor * 0.50
        return edge_floor

    def _build_equalizer(
        self,
        *,
        snapshot: MarketSnapshot,
        fair_probability: float,
        primary_inventory: float,
        secondary_inventory: float,
        primary_avg_entry: float,
        secondary_avg_entry: float,
        gross_inventory: float,
        net_delta: float,
        forced: bool,
        target_naked_ratio: float,
        hard_naked_ratio: float,
        seconds_to_end: float,
        phase: str,
        max_pair_cost: float | None = None,
    ) -> OrderIntent | None:
        if abs(net_delta) <= 0:
            return None

        timeframe = snapshot.market.timeframe
        deep_mode_5m = self.config.single_5m_deep_mode and timeframe == Timeframe.FIVE_MIN
        if net_delta > 0:
            # Too many primary shares; buy secondary to neutralize.
            token_id = snapshot.market.secondary_token_id
            side_book = snapshot.secondary_book
            ask = side_book.best_ask
            fair_side = 1.0 - fair_probability
            fee_bps = snapshot.secondary_fee.base_fee
            picked_side = "secondary"
            dominant_avg = primary_avg_entry if primary_avg_entry > 0 else max(snapshot.primary_book.mid, 0.01)
        else:
            token_id = snapshot.market.primary_token_id
            side_book = snapshot.primary_book
            ask = side_book.best_ask
            fair_side = fair_probability
            fee_bps = snapshot.primary_fee.base_fee
            picked_side = "primary"
            dominant_avg = secondary_avg_entry if secondary_avg_entry > 0 else max(snapshot.secondary_book.mid, 0.01)

        if ask <= 0:
            return None

        taker_entry = self._entry_from_ask(ask)
        maker_entry = self._maker_entry_from_book(side_book)
        entry = taker_entry
        tif = TimeInForce.IOC
        post_only = False
        execution_style = "taker_ioc"
        if not forced and maker_entry > 0 and seconds_to_end > self._force_equalizer_seconds(timeframe):
            entry = maker_entry
            tif = TimeInForce.GTC
            post_only = True
            execution_style = "resting_maker"
        if entry <= 0:
            return None

        dominant_fee_bps = snapshot.primary_fee.base_fee if net_delta > 0 else snapshot.secondary_fee.base_fee
        marginal_pair_cost = dominant_avg + entry
        marginal_pair_cost_all_in = (
            dominant_avg
            + entry
            + per_share_fee(dominant_avg, dominant_fee_bps)
            + per_share_fee(entry, fee_bps)
            + (2.0 * self.config.directional_slippage_buffer)
        )
        if deep_mode_5m:
            deep_cap = self._deep_mode_equalizer_all_in_cap(forced=forced, seconds_to_end=seconds_to_end)
            if marginal_pair_cost_all_in > deep_cap:
                return None
        if not forced and self.config.bankroll_usdc <= 50.0 and marginal_pair_cost > self.max_completion_cost:
            return None

        target_ratio = target_naked_ratio * (0.5 if forced else 1.0)
        required = max(0.0, (abs(net_delta) - (target_ratio * gross_inventory)) / (1.0 + target_ratio))
        if forced:
            required = max(required, abs(net_delta) * 0.75)
        elif max_pair_cost is not None and marginal_pair_cost > max_pair_cost:
            over = marginal_pair_cost - max_pair_cost
            if over <= self.rebalance_ladder_soft_step_1:
                required *= self.rebalance_ladder_soft_clip_1
            elif over <= self.rebalance_ladder_soft_step_2:
                required *= self.rebalance_ladder_soft_clip_2
            else:
                return None

        min_size = snapshot.market.order_min_size
        if not forced and required < (min_size * 0.98):
            # For single-lot one-sided exposure in fast 5m windows, force one
            # equalizer lot instead of standing down, so we can complete sets.
            if deep_mode_5m and abs(net_delta) >= (min_size * 1.20):
                required = min_size
            elif abs(net_delta) >= (min_size * 0.90) and gross_inventory <= (min_size * 2.5):
                required = min_size
            else:
                return None
        size = max(min_size, required)

        # Equalizers are risk-reducing intents; do not suppress them using
        # market-cap heuristics. Cash/risk checks at execution time handle
        # affordability and hard limits.
        fee = per_share_fee(entry, fee_bps)
        edge_net = fair_side - entry - fee - self.config.directional_slippage_buffer
        return OrderIntent(
            market_id=snapshot.market.market_id,
            token_id=token_id,
            side=Side.BUY,
            price=entry,
            size=size,
            tif=tif,
            post_only=post_only,
            engine="engine_pair_arb",
            expected_edge=50.0 if forced else 15.0,
            metadata={
                "strategy": "pair-arb",
                "intent_type": "equalize_forced" if forced else "equalize",
                "picked_side": picked_side,
                "p_fair": fair_side,
                "edge_net": edge_net,
                "best_ask": ask,
                "maker_entry": maker_entry,
                "taker_entry": taker_entry,
                "execution_style": execution_style,
                "fee_bps": fee_bps,
                "net_delta_before": net_delta,
                "gross_inventory_before": gross_inventory,
                "target_naked_ratio": target_naked_ratio,
                "hard_naked_ratio": hard_naked_ratio,
                "order_min_size": snapshot.market.order_min_size,
                "marginal_pair_cost": marginal_pair_cost,
                "marginal_pair_cost_all_in": marginal_pair_cost_all_in,
                "target_pair_cost": max_pair_cost,
                "ladder_step_1": self.rebalance_ladder_soft_step_1,
                "ladder_step_2": self.rebalance_ladder_soft_step_2,
                "timeframe": timeframe.value,
                "seconds_to_end": seconds_to_end,
                "phase": phase,
                "quote_refresh_seconds": self._quote_refresh_seconds(timeframe),
                "quote_max_age_seconds": self._quote_max_age_seconds(timeframe),
                "quote_level_id": f"equalize_{picked_side}_l1",
            },
        )

    def generate(
        self,
        snapshot: MarketSnapshot,
        fair_probability: float,
        primary_inventory: float = 0.0,
        secondary_inventory: float = 0.0,
        primary_avg_entry: float = 0.0,
        secondary_avg_entry: float = 0.0,
        primary_motion: dict[str, float] | None = None,
        secondary_motion: dict[str, float] | None = None,
        now: datetime | None = None,
        preferred_entry_side: str | None = None,
        learned_primary_pair_price: float | None = None,
        learned_primary_success_rate: float | None = None,
        learned_primary_samples: int = 0,
        learned_secondary_pair_price: float | None = None,
        learned_secondary_success_rate: float | None = None,
        learned_secondary_samples: int = 0,
        rolling_pair_cost_avg: float | None = None,
        rolling_pair_cost_samples: int = 0,
    ) -> list[OrderIntent]:
        # Strategy is principle-first rather than copy-trader:
        # build complete sets around discounted legs, then merge/redeem.
        market = snapshot.market
        if market.timeframe not in {Timeframe.FIVE_MIN, Timeframe.FIFTEEN_MIN, Timeframe.ONE_HOUR}:
            return []

        now = now or datetime.now(tz=timezone.utc)
        if (market.start_time - now).total_seconds() > 0:
            return []
        seconds_to_end = (market.end_time - now).total_seconds()
        phase = self._execution_phase(
            timeframe=market.timeframe,
            now=now,
            start_time=market.start_time,
            seconds_to_end=seconds_to_end,
        )
        deep_mode_5m = self.config.single_5m_deep_mode and market.timeframe == Timeframe.FIVE_MIN
        inside_time_guard = seconds_to_end <= self._time_guard_seconds(market.timeframe)

        ask_up = snapshot.primary_book.best_ask
        ask_down = snapshot.secondary_book.best_ask
        if ask_up <= 0 or ask_down <= 0:
            return []

        fair_up = clamp(fair_probability, 0.01, 0.99)
        fair_down = clamp(1.0 - fair_up, 0.01, 0.99)
        fee_up = per_share_fee(ask_up, snapshot.primary_fee.base_fee)
        fee_down = per_share_fee(ask_down, snapshot.secondary_fee.base_fee)
        edge_up = fair_up - ask_up - fee_up - self.config.directional_slippage_buffer
        edge_down = fair_down - ask_down - fee_down - self.config.directional_slippage_buffer

        gross_inventory = max(0.0, primary_inventory) + max(0.0, secondary_inventory)
        net_delta = max(0.0, primary_inventory) - max(0.0, secondary_inventory)
        naked_ratio = abs(net_delta) / max(snapshot.market.order_min_size, gross_inventory)
        effective_target_ratio = clamp(self.target_naked_ratio, 0.08, 0.30)
        if gross_inventory <= (snapshot.market.order_min_size * 1.5):
            effective_target_ratio = min(effective_target_ratio, 0.10)
        effective_hard_ratio = max(self.hard_naked_ratio, min(0.60, effective_target_ratio + 0.14))
        if deep_mode_5m:
            # Deep mode can ladder aggressively, but only while inventory stays
            # relatively balanced.
            effective_target_ratio = 0.06
            effective_hard_ratio = 0.14

        # Keep the near-expiry time guard for fresh entries, but still allow
        # inventory-reducing equalization when we already hold one-sided risk.
        if inside_time_guard and gross_inventory <= 0:
            return []

        # First priority: eliminate large one-sided exposure quickly.
        if gross_inventory > 0 and naked_ratio >= effective_hard_ratio:
            must_force = deep_mode_5m or (seconds_to_end <= self._force_equalizer_seconds(market.timeframe))
            if must_force:
                forced = self._build_equalizer(
                    snapshot=snapshot,
                    fair_probability=fair_up,
                    primary_inventory=primary_inventory,
                    secondary_inventory=secondary_inventory,
                    primary_avg_entry=primary_avg_entry,
                    secondary_avg_entry=secondary_avg_entry,
                    gross_inventory=gross_inventory,
                    net_delta=net_delta,
                    forced=True,
                    target_naked_ratio=effective_target_ratio,
                    hard_naked_ratio=effective_hard_ratio,
                    seconds_to_end=seconds_to_end,
                    phase=phase,
                    max_pair_cost=None,
                )
                if forced is not None:
                    return [forced]
            if deep_mode_5m:
                # In deep mode, stop expansion whenever skew is too large.
                return []

        intents: list[OrderIntent] = []
        allow_alpha_entries = True
        if gross_inventory > 0 and naked_ratio >= effective_target_ratio:
            soft_eq = self._build_equalizer(
                snapshot=snapshot,
                fair_probability=fair_up,
                primary_inventory=primary_inventory,
                secondary_inventory=secondary_inventory,
                primary_avg_entry=primary_avg_entry,
                secondary_avg_entry=secondary_avg_entry,
                gross_inventory=gross_inventory,
                net_delta=net_delta,
                forced=False,
                target_naked_ratio=effective_target_ratio,
                hard_naked_ratio=effective_hard_ratio,
                seconds_to_end=seconds_to_end,
                phase=phase,
                max_pair_cost=self._rebalance_pair_cost_cap(market.timeframe, phase),
            )
            if soft_eq is not None:
                intents.append(soft_eq)
            # Keep paired ladder quoting active even when imbalanced. Paired
            # entries preserve net delta and improve queue coverage across the
            # traversal range; only one-sided alpha is suppressed.
            allow_alpha_entries = False

        # Keep pair-entry laddering active deeper into the window; only stop
        # pair expansion near the hard time guard.
        pair_entry_guard_seconds = self._time_guard_seconds(market.timeframe)
        alpha_entry_guard_seconds = self._alpha_entry_guard_seconds(market.timeframe)
        if seconds_to_end <= pair_entry_guard_seconds:
            return intents
        alpha_entries_blocked = seconds_to_end <= alpha_entry_guard_seconds

        pair_cost = ask_up + ask_down
        pair_cost_all_in = pair_cost + fee_up + fee_down + (2.0 * self.config.directional_slippage_buffer)
        # Do not block on touch/top all-in alone. In volatile books, top can be
        # expensive while deeper maker levels remain viable.

        fluctuation_regime = self._fluctuation_regime(
            timeframe=market.timeframe,
            phase=phase,
            primary_motion=primary_motion,
            secondary_motion=secondary_motion,
        )
        fair_signal = self._fair_value_signal(
            snapshot=snapshot,
            fair_up=fair_up,
            primary_motion=primary_motion,
            secondary_motion=secondary_motion,
            fluctuation_regime=fluctuation_regime,
        )
        hold_queue_mode = bool(fluctuation_regime.get("hold_queue", 0.0) > 0.5)
        min_quote_dwell_seconds = float(fluctuation_regime.get("min_dwell_seconds", 0.0))
        regime_extra_levels = int(max(0.0, fluctuation_regime.get("extra_levels", 0.0)))
        volatility_score = float(fluctuation_regime.get("volatility_score", 0.0))
        if deep_mode_5m:
            regime_extra_levels += 2
            hold_queue_mode = True
            if phase == "open":
                min_quote_dwell_seconds = max(min_quote_dwell_seconds, 2.8)
            elif phase == "middle":
                min_quote_dwell_seconds = max(min_quote_dwell_seconds, 2.0)
        fast_iteration_mode = self._fast_iteration_mode()
        quote_refresh_seconds = self._quote_refresh_seconds(market.timeframe)
        quote_max_age_seconds = self._quote_max_age_seconds(market.timeframe)
        if deep_mode_5m:
            # Keep deeper 5m ladders in queue longer to reduce churn and retain
            # queue priority during rapid swings.
            quote_refresh_seconds = max(0.40, quote_refresh_seconds * 1.20)
            quote_max_age_seconds = max(0.80, quote_max_age_seconds * 1.80)
        if hold_queue_mode:
            if fast_iteration_mode:
                quote_refresh_seconds *= self.fast_iteration_hold_refresh_multiplier
                quote_max_age_seconds *= self.fast_iteration_hold_max_age_multiplier
                min_quote_dwell_seconds *= self.fast_iteration_hold_dwell_scale
            else:
                quote_refresh_seconds *= 1.35
                quote_max_age_seconds *= 1.55
        if min_quote_dwell_seconds > 0:
            min_quote_dwell_seconds = max(0.30, min_quote_dwell_seconds)
        pair_cost_governor_cap = self._pair_cost_governor_cap(
            timeframe=market.timeframe,
            phase=phase,
            rolling_pair_cost_avg=rolling_pair_cost_avg,
            rolling_pair_cost_samples=rolling_pair_cost_samples,
        )
        if deep_mode_5m:
            pair_cost_governor_cap = min(
                pair_cost_governor_cap,
                self._deep_mode_governor_ceiling(rolling_pair_cost_avg),
            )
        thrive_entry_bonus, thrive_governor_bonus = self._volatility_thrive_cap_bonus(
            timeframe=market.timeframe,
            phase=phase,
            fluctuation_regime=fluctuation_regime,
            rolling_pair_cost_avg=rolling_pair_cost_avg,
            rolling_pair_cost_samples=rolling_pair_cost_samples,
            deep_mode_5m=deep_mode_5m,
        )
        if thrive_governor_bonus > 0:
            pair_cost_governor_cap = min(
                1.0400,
                pair_cost_governor_cap + thrive_governor_bonus,
            )

        up_taker_entry = self._entry_from_ask(ask_up)
        down_taker_entry = self._entry_from_ask(ask_down)
        up_maker_entry = self._maker_entry_from_book(snapshot.primary_book)
        down_maker_entry = self._maker_entry_from_book(snapshot.secondary_book)
        up_maker_touch = self._maker_touch_entry(snapshot.primary_book)
        down_maker_touch = self._maker_touch_entry(snapshot.secondary_book)
        up_maker_ladder = self._maker_ladder_from_book(
            book=snapshot.primary_book,
            timeframe=market.timeframe,
            phase=phase,
            motion=primary_motion,
            extra_levels=regime_extra_levels,
        )
        down_maker_ladder = self._maker_ladder_from_book(
            book=snapshot.secondary_book,
            timeframe=market.timeframe,
            phase=phase,
            motion=secondary_motion,
            extra_levels=regime_extra_levels,
        )
        up_quote_entry = up_maker_touch if up_maker_touch > 0 else up_taker_entry
        down_quote_entry = down_maker_touch if down_maker_touch > 0 else down_taker_entry
        if up_quote_entry <= 0 or down_quote_entry <= 0:
            return intents
        quote_fee_up = per_share_fee(up_quote_entry, snapshot.primary_fee.base_fee)
        quote_fee_down = per_share_fee(down_quote_entry, snapshot.secondary_fee.base_fee)
        pair_quote_cost = up_quote_entry + down_quote_entry
        pair_quote_cost_all_in = pair_quote_cost + quote_fee_up + quote_fee_down + (
            2.0 * self.config.directional_slippage_buffer
        )

        market_cap = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        dominant_notional = max(
            primary_inventory * max(snapshot.primary_book.mid, ask_up, 0.01),
            secondary_inventory * max(snapshot.secondary_book.mid, ask_down, 0.01),
        )
        remaining_cap = max(0.0, market_cap - dominant_notional)
        min_size = snapshot.market.order_min_size
        min_alpha_notional = min_size * max(0.01, min(up_quote_entry, down_quote_entry))
        if remaining_cap < min_alpha_notional:
            return intents

        bankroll_scale = clamp(self.config.bankroll_usdc / 250.0, 0.10, 1.0)
        pair_entry_cap = self._pair_entry_cost_cap(market.timeframe, phase)
        if deep_mode_5m:
            pair_entry_cap = self._deep_mode_pair_entry_cap(
                pair_entry_cap=pair_entry_cap,
                governor_cap=pair_cost_governor_cap,
            )
        if thrive_entry_bonus > 0:
            pair_entry_cap = min(1.0350, pair_entry_cap + thrive_entry_bonus)
        swing_short_metric = float(fluctuation_regime.get("swing_short", 0.0))
        flip_rate_metric = float(fluctuation_regime.get("flip_rate_max", 0.0))
        anticipatory_mode = (
            market.timeframe == Timeframe.FIVE_MIN
            and phase in {"open", "middle"}
            and (deep_mode_5m or flip_rate_metric >= 0.09 or swing_short_metric >= 0.015)
        )
        anticipatory_extreme = anticipatory_mode and (flip_rate_metric >= 0.17 or swing_short_metric >= 0.028)
        pair_candidate_levels: list[dict[str, float]] = []
        if phase in {"open", "middle"} and up_maker_ladder and down_maker_ladder:
            pair_index_plan = self._pair_ladder_index_plan(
                primary_levels=len(up_maker_ladder),
                secondary_levels=len(down_maker_ladder),
                anticipatory=anticipatory_mode,
                extreme=anticipatory_extreme,
            )
            for primary_level_idx, secondary_level_idx in pair_index_plan:
                primary_pair_price = up_maker_ladder[primary_level_idx]
                secondary_pair_price = down_maker_ladder[secondary_level_idx]
                pair_level_cost = primary_pair_price + secondary_pair_price
                pair_level_cost_all_in = (
                    pair_level_cost
                    + per_share_fee(primary_pair_price, snapshot.primary_fee.base_fee)
                    + per_share_fee(secondary_pair_price, snapshot.secondary_fee.base_fee)
                    + (2.0 * self.config.directional_slippage_buffer)
                )
                if pair_level_cost > pair_entry_cap:
                    continue
                if pair_level_cost_all_in >= self.absurd_pair_cost_guard:
                    continue
                if pair_level_cost_all_in > pair_cost_governor_cap:
                    continue
                pair_candidate_levels.append(
                    {
                        "primary_price": primary_pair_price,
                        "secondary_price": secondary_pair_price,
                        "pair_cost": pair_level_cost,
                        "pair_cost_all_in": pair_level_cost_all_in,
                        "level": float(len(pair_candidate_levels) + 1),
                        "primary_level": float(primary_level_idx + 1),
                        "secondary_level": float(secondary_level_idx + 1),
                        "anticipatory_offset": float(abs(primary_level_idx - secondary_level_idx)),
                    }
                )
            if pair_candidate_levels:
                tick = max(0.01, self.maker_entry_tick)
                primary_touch = float(up_maker_ladder[0])
                secondary_touch = float(down_maker_ladder[0])
                filtered_levels: list[dict[str, float]] = []
                for level in pair_candidate_levels:
                    primary_price = float(level["primary_price"])
                    secondary_price = float(level["secondary_price"])
                    primary_gap_ticks = int(round(max(0.0, primary_touch - primary_price) / tick))
                    secondary_gap_ticks = int(round(max(0.0, secondary_touch - secondary_price) / tick))
                    touch_gap_sum = primary_gap_ticks + secondary_gap_ticks
                    anticipatory_offset = int(level.get("anticipatory_offset", 0.0))
                    max_gap_allowed = 3 if anticipatory_extreme else 2
                    if market.timeframe == Timeframe.FIVE_MIN and phase in {"open", "middle"}:
                        max_gap_allowed += int(
                            max(0.0, min(2.0, volatility_score * 2.0))
                        )
                    if max(primary_gap_ticks, secondary_gap_ticks) > max_gap_allowed:
                        continue
                    max_offset_allowed = 1
                    if anticipatory_extreme:
                        max_offset_allowed = 2
                    elif market.timeframe == Timeframe.FIVE_MIN and volatility_score >= 0.75:
                        max_offset_allowed = 2
                    if anticipatory_offset > max_offset_allowed:
                        continue
                    level["primary_gap_ticks"] = float(primary_gap_ticks)
                    level["secondary_gap_ticks"] = float(secondary_gap_ticks)
                    level["touch_gap_sum"] = float(touch_gap_sum)
                    # Fill-first ranking:
                    # prioritize levels that sit closest to touch and only then
                    # optimize pair-cost.
                    fill_score = (
                        10.0
                        - (2.4 * touch_gap_sum)
                        - (0.9 * anticipatory_offset)
                        - (max(0.0, float(level["pair_cost_all_in"]) - 0.9900) * 40.0)
                    )
                    level["fill_score"] = fill_score
                    filtered_levels.append(level)
                if filtered_levels:
                    filtered_levels.sort(
                        key=lambda item: (
                            -float(item.get("fill_score", 0.0)),
                            float(item.get("pair_cost_all_in", 9.99)),
                        )
                    )
                    pair_candidate_levels = filtered_levels

        pair_post_only = len(pair_candidate_levels) > 0
        pair_tif = TimeInForce.GTC if pair_post_only else TimeInForce.IOC
        pair_execution_style = "resting_maker_pair_ladder" if pair_post_only else "taker_ioc_pair"
        if not pair_candidate_levels:
            if up_taker_entry <= 0 or down_taker_entry <= 0:
                return intents
            taker_pair_cost = up_taker_entry + down_taker_entry
            taker_pair_cost_all_in = (
                taker_pair_cost
                + per_share_fee(up_taker_entry, snapshot.primary_fee.base_fee)
                + per_share_fee(down_taker_entry, snapshot.secondary_fee.base_fee)
                + (2.0 * self.config.directional_slippage_buffer)
            )
            if (
                taker_pair_cost > pair_entry_cap
                or taker_pair_cost_all_in >= self.absurd_pair_cost_guard
                or taker_pair_cost_all_in > pair_cost_governor_cap
            ):
                pair_candidate_levels = []
            else:
                pair_candidate_levels = [
                    {
                        "primary_price": up_taker_entry,
                        "secondary_price": down_taker_entry,
                        "pair_cost": taker_pair_cost,
                        "pair_cost_all_in": taker_pair_cost_all_in,
                        "level": 1.0,
                        "primary_level": 1.0,
                        "secondary_level": 1.0,
                        "anticipatory_offset": 0.0,
                    }
                ]

        if pair_candidate_levels:
            level_costs = [float(level["pair_cost"]) for level in pair_candidate_levels]
            max_level_cost = max(level_costs)
            min_pair_notional = min_size * max(0.01, max_level_cost)
            if remaining_cap < min_pair_notional:
                return intents
            max_size_by_cap = remaining_cap / max(0.01, max_level_cost)
            if max_size_by_cap < min_size:
                return intents
            pair_best_all_in = min(float(level["pair_cost_all_in"]) for level in pair_candidate_levels)
            pair_expected_edge = max(0.0, 1.0 - pair_best_all_in)
            edge_boost = 1.0 + min(1.50, pair_expected_edge * 120.0)
            base_pair_notional = max(min_pair_notional, 4.0 + (6.0 * bankroll_scale))
            if fast_iteration_mode:
                clipped_pair_notional = min(
                    self.fast_iteration_pair_notional_cap,
                    base_pair_notional * self.fast_iteration_notional_scale,
                )
                base_pair_notional = max(min_pair_notional, clipped_pair_notional)
            capture_fraction = self._volatility_capture_fraction(
                timeframe=market.timeframe,
                phase=phase,
                primary_motion=primary_motion,
                secondary_motion=secondary_motion,
            )
            swing = max(
                self._motion_metric(primary_motion, "ask_swing"),
                self._motion_metric(secondary_motion, "ask_swing"),
            )
            aggression = 1.0 + min(1.0, max(0.0, swing) * 14.0)
            target_pair_notional = min(
                remaining_cap * capture_fraction,
                base_pair_notional * edge_boost * aggression,
            )
            if market.timeframe == Timeframe.FIVE_MIN and phase in {"open", "middle"}:
                thrive_capture = clamp(capture_fraction + (0.10 * volatility_score), 0.20, 0.92)
                thrive_multiplier = 1.0 + (0.45 * volatility_score)
                target_pair_notional = min(
                    remaining_cap * thrive_capture,
                    target_pair_notional * thrive_multiplier,
                )
            if deep_mode_5m:
                if phase == "open":
                    capture_fraction = max(capture_fraction, 0.78)
                else:
                    capture_fraction = max(capture_fraction, 0.66)
                target_pair_notional = min(
                    remaining_cap * capture_fraction,
                    target_pair_notional * 1.45,
                )

            # For 5m markets, maintain a minimum two-level ladder whenever cap
            # allows it; one-level quoting misses a lot of transient prints.
            if market.timeframe == Timeframe.FIVE_MIN and phase in {"open", "middle"}:
                if fast_iteration_mode:
                    min_levels = 1
                    if volatility_score >= 0.60 and len(pair_candidate_levels) >= 2:
                        min_levels = max(min_levels, 2)
                    if swing >= 0.04 and len(pair_candidate_levels) >= 2:
                        min_levels = 2
                    if regime_extra_levels >= 2 and len(pair_candidate_levels) >= 3:
                        min_levels = 3
                else:
                    min_levels = 2
                    if swing >= 0.04 and len(pair_candidate_levels) >= 3:
                        min_levels = 3
                    if regime_extra_levels >= 1 and len(pair_candidate_levels) >= 3:
                        min_levels = max(min_levels, 3)
                    if regime_extra_levels >= 2 and len(pair_candidate_levels) >= 4:
                        min_levels = max(min_levels, 4)
                    if volatility_score >= 0.75 and len(pair_candidate_levels) >= 4:
                        min_levels = max(min_levels, 4)
                if anticipatory_mode and len(pair_candidate_levels) >= 2:
                    anticipatory_min = 2 if fast_iteration_mode else 3
                    min_levels = max(min_levels, anticipatory_min)
                if anticipatory_extreme and len(pair_candidate_levels) >= 4:
                    anticipatory_extreme_min = 4 if fast_iteration_mode else 5
                    min_levels = max(min_levels, anticipatory_extreme_min)
                if deep_mode_5m and len(pair_candidate_levels) >= 3:
                    # Deep mode should keep multiple paired levels resting so
                    # fast traversals fill into our queue on both sides.
                    deep_min_levels = 2
                    if volatility_score >= 0.45 and len(pair_candidate_levels) >= 3:
                        deep_min_levels = 3
                    if volatility_score >= 0.70 and len(pair_candidate_levels) >= 4:
                        deep_min_levels = 4
                    if volatility_score >= 0.85 and len(pair_candidate_levels) >= 5:
                        deep_min_levels = 5
                    if regime_extra_levels >= 2 and len(pair_candidate_levels) >= 6:
                        deep_min_levels = max(deep_min_levels, 6)
                    min_levels = max(min_levels, deep_min_levels)
                if len(pair_candidate_levels) >= min_levels:
                    min_notional_for_levels = (
                        sum(float(level["pair_cost"]) for level in pair_candidate_levels[:min_levels]) * min_size
                    )
                    target_pair_notional = max(target_pair_notional, min_notional_for_levels)
            total_size = min(max_size_by_cap, max(min_size, target_pair_notional / max(0.01, max_level_cost)))
            if total_size < min_size:
                return intents

            level_count = min(len(pair_candidate_levels), max(1, int(total_size / min_size)))
            if deep_mode_5m:
                max_deep_levels = 5
                if volatility_score >= 0.70:
                    max_deep_levels = 6
                if volatility_score >= 0.85:
                    max_deep_levels = 7
                if regime_extra_levels >= 2:
                    max_deep_levels = max(max_deep_levels, 6)
                if anticipatory_extreme:
                    max_deep_levels = max(max_deep_levels, 8)
                level_count = min(level_count, max_deep_levels)
            selected_levels: list[dict[str, float]] = []
            size_per_level = 0.0
            while level_count > 0:
                selected_levels = pair_candidate_levels[:level_count]
                denom = sum(float(level["pair_cost"]) for level in selected_levels)
                if denom <= 0:
                    level_count -= 1
                    continue
                size_per_level = min(total_size / level_count, remaining_cap / denom)
                pair_lot_step = min_size
                if market.timeframe == Timeframe.FIVE_MIN:
                    pair_lot_step = max(min_size, 1.0)
                size_per_level = self._size_to_lot(size_per_level, pair_lot_step)
                if size_per_level >= min_size:
                    break
                level_count -= 1
            if level_count <= 0 or size_per_level < min_size:
                return intents

            total_levels = len(selected_levels)
            for level_index, level in enumerate(selected_levels):
                primary_pair_price = float(level["primary_price"])
                secondary_pair_price = float(level["secondary_price"])
                level_pair_cost = float(level["pair_cost"])
                level_pair_cost_all_in = float(level["pair_cost_all_in"])
                level_edge = max(0.0, 1.0 - level_pair_cost_all_in)
                pair_group_id = f"{market.market_id}:pair_l{level_index + 1}"

                primary_metadata = {
                    "strategy": "pair-complete-set",
                    "intent_type": "pair_entry_primary",
                    "picked_side": "primary",
                    "p_fair": fair_up,
                    "edge_net": edge_up,
                    "best_ask": ask_up,
                    "maker_entry": up_maker_entry,
                    "maker_touch_entry": up_maker_touch,
                    "taker_entry": up_taker_entry,
                    "execution_style": pair_execution_style,
                    "quote_refresh_seconds": quote_refresh_seconds,
                    "quote_max_age_seconds": quote_max_age_seconds,
                    "hold_queue": hold_queue_mode,
                    "min_quote_dwell_seconds": min_quote_dwell_seconds,
                    "fee_bps": snapshot.primary_fee.base_fee,
                    "order_min_size": market.order_min_size,
                    "seconds_to_end": seconds_to_end,
                    "timeframe": market.timeframe.value,
                    "phase": phase,
                    "pair_cost_snapshot": pair_cost,
                    "pair_cost_snapshot_all_in": pair_cost_all_in,
                    "pair_cost_quote": level_pair_cost,
                    "pair_cost_quote_all_in": level_pair_cost_all_in,
                    "pair_discount_snapshot": 1.0 - pair_cost,
                    "pair_unit_cost": level_pair_cost,
                    "pair_edge_all_in": level_edge,
                    "opposite_token_id": market.secondary_token_id,
                    "opposite_entry": secondary_pair_price,
                    "opposite_fee_bps": snapshot.secondary_fee.base_fee,
                    "ladder_level": level_index + 1,
                    "ladder_total_levels": total_levels,
                    "primary_ladder_index": int(level.get("primary_level", 1.0)),
                    "secondary_ladder_index": int(level.get("secondary_level", 1.0)),
                    "anticipatory_offset": float(level.get("anticipatory_offset", 0.0)),
                    "primary_gap_ticks": float(level.get("primary_gap_ticks", 0.0)),
                    "secondary_gap_ticks": float(level.get("secondary_gap_ticks", 0.0)),
                    "touch_gap_sum": float(level.get("touch_gap_sum", 0.0)),
                    "fill_score": float(level.get("fill_score", 0.0)),
                    "anticipatory_mode": anticipatory_mode,
                    "fair_implied_up": float(fair_signal.get("implied_up", 0.5)),
                    "fair_dislocation_up": float(fair_signal.get("dislocation", 0.0)),
                    "fair_confidence": float(fair_signal.get("confidence", 0.5)),
                    "fair_bias_side": str(fair_signal.get("bias_side") or ""),
                    "pair_group_id": pair_group_id,
                    "pair_leg_role": "primary",
                    "quote_level_id": f"pair_primary_l{level_index + 1}",
                    "primary_ask_swing": self._motion_metric(primary_motion, "ask_swing"),
                    "secondary_ask_swing": self._motion_metric(secondary_motion, "ask_swing"),
                    "primary_flip_rate": max(
                        self._motion_metric(primary_motion, "ask_flip_rate"),
                        self._motion_metric(primary_motion, "mid_flip_rate"),
                    ),
                    "secondary_flip_rate": max(
                        self._motion_metric(secondary_motion, "ask_flip_rate"),
                        self._motion_metric(secondary_motion, "mid_flip_rate"),
                    ),
                    "fluctuation_swing_short": fluctuation_regime.get("swing_short", 0.0),
                    "fluctuation_flip_rate_max": fluctuation_regime.get("flip_rate_max", 0.0),
                    "fluctuation_extra_levels": fluctuation_regime.get("extra_levels", 0.0),
                    "fluctuation_volatility_score": fluctuation_regime.get("volatility_score", 0.0),
                    "thrive_entry_bonus": thrive_entry_bonus,
                    "thrive_governor_bonus": thrive_governor_bonus,
                    "governor_pair_cost_cap": pair_cost_governor_cap,
                    "rolling_pair_cost_avg": rolling_pair_cost_avg,
                    "rolling_pair_cost_samples": rolling_pair_cost_samples,
                    "fast_iteration_mode": fast_iteration_mode,
                    "primary_motion_samples": int(self._motion_metric(primary_motion, "samples")),
                    "secondary_motion_samples": int(self._motion_metric(secondary_motion, "samples")),
                }
                secondary_metadata = {
                    "strategy": "pair-complete-set",
                    "intent_type": "pair_completion",
                    "picked_side": "secondary",
                    "p_fair": fair_down,
                    "edge_net": edge_down,
                    "best_ask": ask_down,
                    "maker_entry": down_maker_entry,
                    "maker_touch_entry": down_maker_touch,
                    "taker_entry": down_taker_entry,
                    "execution_style": pair_execution_style,
                    "quote_refresh_seconds": quote_refresh_seconds,
                    "quote_max_age_seconds": quote_max_age_seconds,
                    "hold_queue": hold_queue_mode,
                    "min_quote_dwell_seconds": min_quote_dwell_seconds,
                    "fee_bps": snapshot.secondary_fee.base_fee,
                    "order_min_size": market.order_min_size,
                    "seconds_to_end": seconds_to_end,
                    "timeframe": market.timeframe.value,
                    "phase": phase,
                    "pair_cost_snapshot": pair_cost,
                    "pair_cost_snapshot_all_in": pair_cost_all_in,
                    "pair_cost_quote": level_pair_cost,
                    "pair_cost_quote_all_in": level_pair_cost_all_in,
                    "pair_discount_snapshot": 1.0 - pair_cost,
                    "pair_unit_cost": level_pair_cost,
                    "pair_edge_all_in": level_edge,
                    "opposite_token_id": market.primary_token_id,
                    "opposite_entry": primary_pair_price,
                    "opposite_fee_bps": snapshot.primary_fee.base_fee,
                    "ladder_level": level_index + 1,
                    "ladder_total_levels": total_levels,
                    "primary_ladder_index": int(level.get("primary_level", 1.0)),
                    "secondary_ladder_index": int(level.get("secondary_level", 1.0)),
                    "anticipatory_offset": float(level.get("anticipatory_offset", 0.0)),
                    "primary_gap_ticks": float(level.get("primary_gap_ticks", 0.0)),
                    "secondary_gap_ticks": float(level.get("secondary_gap_ticks", 0.0)),
                    "touch_gap_sum": float(level.get("touch_gap_sum", 0.0)),
                    "fill_score": float(level.get("fill_score", 0.0)),
                    "anticipatory_mode": anticipatory_mode,
                    "fair_implied_up": float(fair_signal.get("implied_up", 0.5)),
                    "fair_dislocation_up": float(fair_signal.get("dislocation", 0.0)),
                    "fair_confidence": float(fair_signal.get("confidence", 0.5)),
                    "fair_bias_side": str(fair_signal.get("bias_side") or ""),
                    "pair_group_id": pair_group_id,
                    "pair_leg_role": "secondary",
                    "quote_level_id": f"pair_secondary_l{level_index + 1}",
                    "primary_ask_swing": self._motion_metric(primary_motion, "ask_swing"),
                    "secondary_ask_swing": self._motion_metric(secondary_motion, "ask_swing"),
                    "primary_flip_rate": max(
                        self._motion_metric(primary_motion, "ask_flip_rate"),
                        self._motion_metric(primary_motion, "mid_flip_rate"),
                    ),
                    "secondary_flip_rate": max(
                        self._motion_metric(secondary_motion, "ask_flip_rate"),
                        self._motion_metric(secondary_motion, "mid_flip_rate"),
                    ),
                    "fluctuation_swing_short": fluctuation_regime.get("swing_short", 0.0),
                    "fluctuation_flip_rate_max": fluctuation_regime.get("flip_rate_max", 0.0),
                    "fluctuation_extra_levels": fluctuation_regime.get("extra_levels", 0.0),
                    "fluctuation_volatility_score": fluctuation_regime.get("volatility_score", 0.0),
                    "thrive_entry_bonus": thrive_entry_bonus,
                    "thrive_governor_bonus": thrive_governor_bonus,
                    "governor_pair_cost_cap": pair_cost_governor_cap,
                    "rolling_pair_cost_avg": rolling_pair_cost_avg,
                    "rolling_pair_cost_samples": rolling_pair_cost_samples,
                    "fast_iteration_mode": fast_iteration_mode,
                    "primary_motion_samples": int(self._motion_metric(primary_motion, "samples")),
                    "secondary_motion_samples": int(self._motion_metric(secondary_motion, "samples")),
                }
                first_primary = edge_up <= edge_down
                if abs(edge_up - edge_down) < 1e-9:
                    first_primary = primary_pair_price <= secondary_pair_price
                if first_primary:
                    intents.append(
                        OrderIntent(
                            market_id=market.market_id,
                            token_id=market.primary_token_id,
                            side=Side.BUY,
                            price=primary_pair_price,
                            size=size_per_level,
                            tif=pair_tif,
                            post_only=pair_post_only,
                            engine="engine_pair_arb",
                            expected_edge=level_edge,
                            metadata=primary_metadata,
                        )
                    )
                    intents.append(
                        OrderIntent(
                            market_id=market.market_id,
                            token_id=market.secondary_token_id,
                            side=Side.BUY,
                            price=secondary_pair_price,
                            size=size_per_level,
                            tif=pair_tif,
                            post_only=pair_post_only,
                            engine="engine_pair_arb",
                            expected_edge=level_edge,
                            metadata=secondary_metadata,
                        )
                    )
                    continue
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=market.secondary_token_id,
                        side=Side.BUY,
                        price=secondary_pair_price,
                        size=size_per_level,
                        tif=pair_tif,
                        post_only=pair_post_only,
                        engine="engine_pair_arb",
                        expected_edge=level_edge,
                        metadata=secondary_metadata,
                    )
                )
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=market.primary_token_id,
                        side=Side.BUY,
                        price=primary_pair_price,
                        size=size_per_level,
                        tif=pair_tif,
                        post_only=pair_post_only,
                        engine="engine_pair_arb",
                        expected_edge=level_edge,
                        metadata=primary_metadata,
                    )
                )
            return intents

        if not allow_alpha_entries or alpha_entries_blocked:
            return intents

        side_candidates = [
            {
                "picked_side": "primary",
                "token_id": market.primary_token_id,
                "entry": up_taker_entry,
                "maker_entry": up_maker_entry,
                "ask": ask_up,
                "fee_bps": snapshot.primary_fee.base_fee,
                "fair": fair_up,
                "edge": edge_up,
                "opposite_label": "secondary",
                "opposite_token_id": market.secondary_token_id,
                "opposite_ask": ask_down,
                "opposite_fee_bps": snapshot.secondary_fee.base_fee,
                "opposite_fair": fair_down,
                "learned_pair_price": learned_primary_pair_price,
                "learned_success_rate": learned_primary_success_rate,
                "learned_samples": learned_primary_samples,
            },
            {
                "picked_side": "secondary",
                "token_id": market.secondary_token_id,
                "entry": down_taker_entry,
                "maker_entry": down_maker_entry,
                "ask": ask_down,
                "fee_bps": snapshot.secondary_fee.base_fee,
                "fair": fair_down,
                "edge": edge_down,
                "opposite_label": "primary",
                "opposite_token_id": market.primary_token_id,
                "opposite_ask": ask_up,
                "opposite_fee_bps": snapshot.primary_fee.base_fee,
                "opposite_fair": fair_up,
                "learned_pair_price": learned_secondary_pair_price,
                "learned_success_rate": learned_secondary_success_rate,
                "learned_samples": learned_secondary_samples,
            },
        ]
        if preferred_entry_side in {"primary", "secondary"}:
            side_candidates.sort(key=lambda item: 0 if item["picked_side"] == preferred_entry_side else 1)
        else:
            side_candidates.sort(key=lambda item: float(item["edge"]), reverse=True)

        fair_confidence = float(fair_signal.get("confidence", 0.5))
        fair_dislocation = float(fair_signal.get("dislocation", 0.0))
        fair_bias_side = str(fair_signal.get("bias_side") or "")
        force_bias_side = (
            fair_bias_side in {"primary", "secondary"}
            and fair_confidence >= 0.55
            and abs(fair_dislocation) >= self.fair_bias_threshold
        )
        value_buffer = 0.0015 + ((1.0 - fair_confidence) * 0.0060)

        alpha_choice: dict[str, object] | None = None
        for candidate in side_candidates:
            if force_bias_side and str(candidate["picked_side"]) != fair_bias_side:
                continue
            edge_floor = self._alpha_edge_floor(
                side=str(candidate["picked_side"]),
                learned_primary_pair_price=learned_primary_pair_price,
                learned_primary_success_rate=learned_primary_success_rate,
                learned_primary_samples=learned_primary_samples,
                learned_secondary_pair_price=learned_secondary_pair_price,
                learned_secondary_success_rate=learned_secondary_success_rate,
                learned_secondary_samples=learned_secondary_samples,
            )
            edge_floor = max(edge_floor, 0.0012 + ((1.0 - fair_confidence) * 0.0045))
            candidate_entry = float(candidate["entry"])
            candidate_fair = float(candidate["fair"])
            entry_value_ceiling = max(0.01, candidate_fair - value_buffer)
            if candidate_entry > entry_value_ceiling:
                continue
            if float(candidate["edge"]) < edge_floor:
                continue
            projected_opposite_entry = self._project_opposite_entry(
                float(candidate["opposite_ask"]),
                float(candidate["opposite_fair"]),
            )
            if projected_opposite_entry <= 0:
                continue
            projected_pair_cost_price = (
                float(candidate["entry"])
                + projected_opposite_entry
                + (2.0 * self.config.directional_slippage_buffer)
            )
            projected_pair_cost_all_in = (
                float(candidate["entry"])
                + projected_opposite_entry
                + per_share_fee(float(candidate["entry"]), int(candidate["fee_bps"]))
                + per_share_fee(projected_opposite_entry, int(candidate["opposite_fee_bps"]))
                + (2.0 * self.config.directional_slippage_buffer)
            )
            if projected_pair_cost_price > self.max_entry_projected_completion_cost:
                continue
            if projected_pair_cost_all_in > self.max_entry_projected_completion_cost_all_in:
                continue
            if projected_pair_cost_all_in > pair_cost_governor_cap:
                continue
            alpha_choice = dict(candidate)
            alpha_choice["projected_pair_cost_price"] = projected_pair_cost_price
            alpha_choice["projected_opposite_entry"] = projected_opposite_entry
            alpha_choice["projected_pair_cost_all_in"] = projected_pair_cost_all_in
            alpha_choice["fair_confidence"] = fair_confidence
            alpha_choice["fair_dislocation"] = fair_dislocation
            alpha_choice["fair_bias_side"] = fair_bias_side
            alpha_choice["fair_entry_ceiling"] = entry_value_ceiling
            break

        if alpha_choice is None:
            return intents

        alpha_taker_price = float(alpha_choice["entry"])
        alpha_maker_price = float(alpha_choice.get("maker_entry") or 0.0)
        alpha_post_only = phase in {"open", "middle"} and alpha_maker_price > 0
        alpha_tif = TimeInForce.GTC if alpha_post_only else TimeInForce.IOC
        alpha_execution_style = "resting_maker_alpha_ladder" if alpha_post_only else "taker_ioc_alpha"
        if str(alpha_choice["picked_side"]) == "primary":
            alpha_book = snapshot.primary_book
        else:
            alpha_book = snapshot.secondary_book

        alpha_prices: list[float]
        if alpha_post_only:
            alpha_motion = primary_motion if str(alpha_choice["picked_side"]) == "primary" else secondary_motion
            alpha_prices = self._maker_ladder_from_book(
                book=alpha_book,
                timeframe=market.timeframe,
                phase=phase,
                motion=alpha_motion,
                extra_levels=regime_extra_levels,
            )
            if not alpha_prices and alpha_maker_price > 0:
                alpha_prices = [alpha_maker_price]
        else:
            alpha_prices = [alpha_taker_price]
        if not alpha_prices:
            return intents

        max_alpha_price = max(alpha_prices)
        alpha_min_notional = min_size * max(max_alpha_price, 0.01)
        if remaining_cap < alpha_min_notional:
            return intents
        alpha_max_size = remaining_cap / max(max_alpha_price, 0.01)
        if alpha_max_size < min_size:
            return intents
        alpha_notional_target = min(
            remaining_cap * (0.35 if fast_iteration_mode else 0.40),
            max(
                alpha_min_notional,
                min(
                    self.fast_iteration_alpha_notional_cap,
                    (2.5 + (4.0 * bankroll_scale)) * self.fast_iteration_notional_scale,
                ),
            )
            if fast_iteration_mode
            else max(alpha_min_notional, 2.5 + (4.0 * bankroll_scale)),
        )
        alpha_total_size = min(alpha_max_size, max(min_size, alpha_notional_target / max(max_alpha_price, 0.01)))
        if alpha_total_size < min_size:
            return intents

        level_count = min(len(alpha_prices), max(1, int(alpha_total_size / min_size)))
        if deep_mode_5m:
            level_count = min(level_count, 2)
        selected_prices: list[float] = []
        alpha_size_per_level = 0.0
        while level_count > 0:
            selected_prices = alpha_prices[:level_count]
            denom = sum(max(0.01, p) for p in selected_prices)
            if denom <= 0:
                level_count -= 1
                continue
            alpha_size_per_level = min(alpha_total_size / level_count, remaining_cap / denom)
            alpha_size_per_level = self._size_to_lot(alpha_size_per_level, min_size)
            if alpha_size_per_level >= min_size:
                break
            level_count -= 1
        if level_count <= 0 or alpha_size_per_level < min_size:
            return intents

        total_levels = len(selected_prices)
        for level_index, alpha_price in enumerate(selected_prices):
            fee_bps = int(alpha_choice["fee_bps"])
            opposite_fee_bps = int(alpha_choice["opposite_fee_bps"])
            projected_opposite_entry = float(alpha_choice["projected_opposite_entry"])
            projected_pair_cost_price = alpha_price + projected_opposite_entry + (2.0 * self.config.directional_slippage_buffer)
            projected_pair_cost_all_in = (
                alpha_price
                + projected_opposite_entry
                + per_share_fee(alpha_price, fee_bps)
                + per_share_fee(projected_opposite_entry, opposite_fee_bps)
                + (2.0 * self.config.directional_slippage_buffer)
            )
            edge_net = (
                float(alpha_choice["fair"])
                - alpha_price
                - per_share_fee(alpha_price, fee_bps)
                - self.config.directional_slippage_buffer
            )
            intents.append(
                OrderIntent(
                    market_id=market.market_id,
                    token_id=str(alpha_choice["token_id"]),
                    side=Side.BUY,
                    price=alpha_price,
                    size=alpha_size_per_level,
                    tif=alpha_tif,
                    post_only=alpha_post_only,
                    engine="engine_pair_arb",
                    expected_edge=max(0.0, edge_net),
                    metadata={
                        "strategy": "pair-sequenced-alpha",
                        "intent_type": "alpha_entry",
                        "picked_side": str(alpha_choice["picked_side"]),
                        "p_fair": float(alpha_choice["fair"]),
                        "edge_net": edge_net,
                        "best_ask": float(alpha_choice["ask"]),
                        "maker_entry": alpha_maker_price,
                        "maker_touch_entry": alpha_prices[0],
                        "taker_entry": alpha_taker_price,
                        "execution_style": alpha_execution_style,
                        "quote_refresh_seconds": quote_refresh_seconds,
                        "quote_max_age_seconds": quote_max_age_seconds,
                        "hold_queue": hold_queue_mode,
                        "min_quote_dwell_seconds": min_quote_dwell_seconds,
                        "fee_bps": fee_bps,
                        "order_min_size": market.order_min_size,
                        "seconds_to_end": seconds_to_end,
                        "timeframe": market.timeframe.value,
                        "phase": phase,
                        "pair_cost_snapshot": pair_cost,
                        "pair_cost_snapshot_all_in": pair_cost_all_in,
                        "pair_cost_hint": projected_pair_cost_price,
                        "projected_pair_cost_price": projected_pair_cost_price,
                        "projected_pair_cost_all_in": projected_pair_cost_all_in,
                        "opposite_token_id": str(alpha_choice["opposite_token_id"]),
                        "opposite_entry": projected_opposite_entry,
                        "opposite_fee_bps": opposite_fee_bps,
                        "projected_opposite_improvement": max(
                            0.0, float(alpha_choice["opposite_ask"]) - projected_opposite_entry
                        ),
                        "fair_implied_up": float(fair_signal.get("implied_up", 0.5)),
                        "fair_dislocation_up": float(alpha_choice.get("fair_dislocation", 0.0)),
                        "fair_confidence": float(alpha_choice.get("fair_confidence", 0.5)),
                        "fair_bias_side": str(alpha_choice.get("fair_bias_side") or ""),
                        "fair_entry_ceiling": float(alpha_choice.get("fair_entry_ceiling", alpha_price)),
                        "fluctuation_swing_short": fluctuation_regime.get("swing_short", 0.0),
                        "fluctuation_flip_rate_max": fluctuation_regime.get("flip_rate_max", 0.0),
                        "fluctuation_extra_levels": fluctuation_regime.get("extra_levels", 0.0),
                        "fluctuation_volatility_score": fluctuation_regime.get("volatility_score", 0.0),
                        "thrive_entry_bonus": thrive_entry_bonus,
                        "thrive_governor_bonus": thrive_governor_bonus,
                        "governor_pair_cost_cap": pair_cost_governor_cap,
                        "rolling_pair_cost_avg": rolling_pair_cost_avg,
                        "rolling_pair_cost_samples": rolling_pair_cost_samples,
                        "fast_iteration_mode": fast_iteration_mode,
                        "ladder_level": level_index + 1,
                        "ladder_total_levels": total_levels,
                        "quote_level_id": f"alpha_{alpha_choice['picked_side']}_l{level_index + 1}",
                    },
                )
            )
        return intents

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import MarketSnapshot, OrderIntent, Side, TimeInForce, Timeframe
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

    def _time_guard_seconds(self, timeframe: Timeframe) -> float:
        if timeframe == Timeframe.FIVE_MIN:
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
            if small_bankroll:
                return max(55.0, float(self.config.directional_min_time_left_5m) * 0.60)
            return max(base, 100.0)
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
        max_pair_cost: float | None = None,
    ) -> OrderIntent | None:
        if abs(net_delta) <= 0:
            return None

        if net_delta > 0:
            # Too many primary shares; buy secondary to neutralize.
            token_id = snapshot.market.secondary_token_id
            ask = snapshot.secondary_book.best_ask
            fair_side = 1.0 - fair_probability
            fee_bps = snapshot.secondary_fee.base_fee
            picked_side = "secondary"
            token_price_hint = max(snapshot.secondary_book.mid, ask, 0.01)
            dominant_avg = primary_avg_entry if primary_avg_entry > 0 else max(snapshot.primary_book.mid, 0.01)
        else:
            token_id = snapshot.market.primary_token_id
            ask = snapshot.primary_book.best_ask
            fair_side = fair_probability
            fee_bps = snapshot.primary_fee.base_fee
            picked_side = "primary"
            token_price_hint = max(snapshot.primary_book.mid, ask, 0.01)
            dominant_avg = secondary_avg_entry if secondary_avg_entry > 0 else max(snapshot.secondary_book.mid, 0.01)

        if ask <= 0:
            return None

        entry = self._entry_from_ask(ask)
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
            return None
        size = max(min_size, required)

        # Equalizers are risk-reducing intents; do not suppress them using
        # market-cap heuristics. Cash/risk checks at execution time handle
        # affordability and hard limits.
        _ = token_price_hint

        fee = per_share_fee(ask, fee_bps)
        edge_net = fair_side - ask - fee - self.config.directional_slippage_buffer
        return OrderIntent(
            market_id=snapshot.market.market_id,
            token_id=token_id,
            side=Side.BUY,
            price=entry,
            size=size,
            tif=TimeInForce.IOC,
            post_only=False,
            engine="engine_pair_arb",
            expected_edge=50.0 if forced else 15.0,
            metadata={
                "strategy": "pair-arb",
                "intent_type": "equalize_forced" if forced else "equalize",
                "picked_side": picked_side,
                "p_fair": fair_side,
                "edge_net": edge_net,
                "best_ask": ask,
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
        now: datetime | None = None,
        preferred_entry_side: str | None = None,
        learned_primary_pair_price: float | None = None,
        learned_primary_success_rate: float | None = None,
        learned_primary_samples: int = 0,
        learned_secondary_pair_price: float | None = None,
        learned_secondary_success_rate: float | None = None,
        learned_secondary_samples: int = 0,
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
        granularity_ratio = snapshot.market.order_min_size / max(snapshot.market.order_min_size, gross_inventory)
        effective_target_ratio = max(self.target_naked_ratio, min(0.35, granularity_ratio * 0.75))
        effective_hard_ratio = max(self.hard_naked_ratio, min(0.60, effective_target_ratio + 0.10))

        # Keep the near-expiry time guard for fresh entries, but still allow
        # inventory-reducing equalization when we already hold one-sided risk.
        if inside_time_guard and gross_inventory <= 0:
            return []

        # First priority: eliminate large one-sided exposure quickly.
        if gross_inventory > 0 and naked_ratio >= effective_hard_ratio:
            must_force = seconds_to_end <= self._force_equalizer_seconds(market.timeframe)
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
                    max_pair_cost=None,
                )
                return [forced] if forced is not None else []

        intents: list[OrderIntent] = []
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
                max_pair_cost=self.target_rebalance_pair_cost,
            )
            if soft_eq is not None:
                intents.append(soft_eq)

        # Always neutralize first when one-sided inventory is elevated.
        if gross_inventory > 0 and naked_ratio >= effective_target_ratio and intents:
            return intents

        # Do not open fresh pair entries late in the interval.
        if seconds_to_end <= self._alpha_entry_guard_seconds(market.timeframe):
            return intents

        pair_cost = ask_up + ask_down
        pair_cost_all_in = pair_cost + fee_up + fee_down + (2.0 * self.config.directional_slippage_buffer)

        if pair_cost_all_in >= self.absurd_pair_cost_guard:
            return intents

        up_entry = self._entry_from_ask(ask_up)
        down_entry = self._entry_from_ask(ask_down)
        if up_entry <= 0 or down_entry <= 0:
            return intents

        market_cap = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        dominant_notional = max(
            primary_inventory * max(snapshot.primary_book.mid, ask_up, 0.01),
            secondary_inventory * max(snapshot.secondary_book.mid, ask_down, 0.01),
        )
        remaining_cap = max(0.0, market_cap - dominant_notional)
        min_size = snapshot.market.order_min_size
        min_alpha_notional = min_size * max(0.01, min(up_entry, down_entry))
        if remaining_cap < min_alpha_notional:
            return intents

        bankroll_scale = clamp(self.config.bankroll_usdc / 250.0, 0.10, 1.0)
        pair_expected_edge = max(0.0, 1.0 - pair_cost_all_in)
        pair_unit_cost = max(0.01, up_entry + down_entry)
        primary_metadata = {
            "strategy": "pair-complete-set",
            "intent_type": "pair_entry_primary",
            "picked_side": "primary",
            "p_fair": fair_up,
            "edge_net": edge_up,
            "best_ask": ask_up,
            "fee_bps": snapshot.primary_fee.base_fee,
            "order_min_size": market.order_min_size,
            "seconds_to_end": seconds_to_end,
            "timeframe": market.timeframe.value,
            "pair_cost_snapshot": pair_cost,
            "pair_cost_snapshot_all_in": pair_cost_all_in,
            "pair_discount_snapshot": 1.0 - pair_cost,
            "pair_unit_cost": pair_unit_cost,
            "pair_edge_all_in": pair_expected_edge,
            "opposite_token_id": market.secondary_token_id,
            "opposite_entry": down_entry,
            "opposite_fee_bps": snapshot.secondary_fee.base_fee,
        }
        secondary_metadata = {
            "strategy": "pair-complete-set",
            "intent_type": "pair_completion",
            "picked_side": "secondary",
            "p_fair": fair_down,
            "edge_net": edge_down,
            "best_ask": ask_down,
            "fee_bps": snapshot.secondary_fee.base_fee,
            "order_min_size": market.order_min_size,
            "seconds_to_end": seconds_to_end,
            "timeframe": market.timeframe.value,
            "pair_cost_snapshot": pair_cost,
            "pair_cost_snapshot_all_in": pair_cost_all_in,
            "pair_discount_snapshot": 1.0 - pair_cost,
            "pair_unit_cost": pair_unit_cost,
            "pair_edge_all_in": pair_expected_edge,
            "opposite_token_id": market.primary_token_id,
            "opposite_entry": up_entry,
            "opposite_fee_bps": snapshot.primary_fee.base_fee,
        }

        if pair_cost <= self.max_entry_pair_cost:
            min_pair_notional = min_size * pair_unit_cost
            if remaining_cap < min_pair_notional:
                return intents
            max_size_by_cap = remaining_cap / pair_unit_cost
            if max_size_by_cap < min_size:
                return intents
            pair_edge = max(0.0, 1.0 - pair_cost_all_in)
            edge_boost = 1.0 + min(1.50, pair_edge * 120.0)
            base_pair_notional = max(min_pair_notional, 4.0 + (6.0 * bankroll_scale))
            target_pair_notional = min(remaining_cap * 0.25, base_pair_notional * edge_boost)
            size = min(max_size_by_cap, max(min_size, target_pair_notional / pair_unit_cost))
            if size < min_size:
                return intents
            first_primary = ask_up <= ask_down
            if first_primary:
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=market.primary_token_id,
                        side=Side.BUY,
                        price=up_entry,
                        size=size,
                        tif=TimeInForce.IOC,
                        post_only=False,
                        engine="engine_pair_arb",
                        expected_edge=pair_expected_edge,
                        metadata=primary_metadata,
                    )
                )
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=market.secondary_token_id,
                        side=Side.BUY,
                        price=down_entry,
                        size=size,
                        tif=TimeInForce.IOC,
                        post_only=False,
                        engine="engine_pair_arb",
                        expected_edge=pair_expected_edge,
                        metadata=secondary_metadata,
                    )
                )
                return intents

            intents.append(
                OrderIntent(
                    market_id=market.market_id,
                    token_id=market.secondary_token_id,
                    side=Side.BUY,
                    price=down_entry,
                    size=size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_pair_arb",
                    expected_edge=pair_expected_edge,
                    metadata=secondary_metadata,
                )
            )
            intents.append(
                OrderIntent(
                    market_id=market.market_id,
                    token_id=market.primary_token_id,
                    side=Side.BUY,
                    price=up_entry,
                    size=size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_pair_arb",
                    expected_edge=pair_expected_edge,
                    metadata=primary_metadata,
                )
            )
            return intents

        side_candidates = [
            {
                "picked_side": "primary",
                "token_id": market.primary_token_id,
                "entry": up_entry,
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
                "entry": down_entry,
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

        alpha_choice: dict[str, object] | None = None
        for candidate in side_candidates:
            edge_floor = self._alpha_edge_floor(
                side=str(candidate["picked_side"]),
                learned_primary_pair_price=learned_primary_pair_price,
                learned_primary_success_rate=learned_primary_success_rate,
                learned_primary_samples=learned_primary_samples,
                learned_secondary_pair_price=learned_secondary_pair_price,
                learned_secondary_success_rate=learned_secondary_success_rate,
                learned_secondary_samples=learned_secondary_samples,
            )
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
            alpha_choice = dict(candidate)
            alpha_choice["projected_pair_cost_price"] = projected_pair_cost_price
            alpha_choice["projected_opposite_entry"] = projected_opposite_entry
            alpha_choice["projected_pair_cost_all_in"] = projected_pair_cost_all_in
            break

        if alpha_choice is None:
            return intents

        alpha_price = float(alpha_choice["entry"])
        alpha_min_notional = min_size * max(alpha_price, 0.01)
        if remaining_cap < alpha_min_notional:
            return intents
        alpha_max_size = remaining_cap / max(alpha_price, 0.01)
        if alpha_max_size < min_size:
            return intents
        alpha_notional_target = min(
            remaining_cap * 0.40,
            max(alpha_min_notional, 2.5 + (4.0 * bankroll_scale)),
        )
        alpha_size = min(alpha_max_size, max(min_size, alpha_notional_target / max(alpha_price, 0.01)))
        if alpha_size < min_size:
            return intents

        intents.append(
            OrderIntent(
                market_id=market.market_id,
                token_id=str(alpha_choice["token_id"]),
                side=Side.BUY,
                price=alpha_price,
                size=alpha_size,
                tif=TimeInForce.IOC,
                post_only=False,
                engine="engine_pair_arb",
                expected_edge=max(0.0, float(alpha_choice["edge"])),
                metadata={
                    "strategy": "pair-sequenced-alpha",
                    "intent_type": "alpha_entry",
                    "picked_side": str(alpha_choice["picked_side"]),
                    "p_fair": float(alpha_choice["fair"]),
                    "edge_net": float(alpha_choice["edge"]),
                    "best_ask": float(alpha_choice["ask"]),
                    "fee_bps": int(alpha_choice["fee_bps"]),
                    "order_min_size": market.order_min_size,
                    "seconds_to_end": seconds_to_end,
                    "timeframe": market.timeframe.value,
                    "pair_cost_snapshot": pair_cost,
                    "pair_cost_snapshot_all_in": pair_cost_all_in,
                    "pair_cost_hint": float(alpha_choice["projected_pair_cost_price"]),
                    "projected_pair_cost_price": float(alpha_choice["projected_pair_cost_price"]),
                    "projected_pair_cost_all_in": float(alpha_choice["projected_pair_cost_all_in"]),
                    "opposite_token_id": str(alpha_choice["opposite_token_id"]),
                    "opposite_entry": float(alpha_choice["projected_opposite_entry"]),
                    "opposite_fee_bps": int(alpha_choice["opposite_fee_bps"]),
                    "projected_opposite_improvement": max(
                        0.0, float(alpha_choice["opposite_ask"]) - float(alpha_choice["projected_opposite_entry"])
                    ),
                },
            )
        )
        return intents

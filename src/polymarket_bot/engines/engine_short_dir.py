from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import MarketSnapshot, OrderIntent, Side, TimeInForce, Timeframe
from polymarket_bot.pricing import clamp, per_share_fee, round_tick


@dataclass
class ShortDirectionalEngine:
    config: BotConfig

    def compute_fair_probability(
        self,
        market_prob: float,
        spot_return: float,
        imbalance: float,
        seconds_to_end: float,
        timeframe: Timeframe,
    ) -> float:
        horizon = 300.0 if timeframe == Timeframe.FIVE_MIN else 900.0
        time_factor = clamp(seconds_to_end / horizon, 0.0, 1.0)
        ret_edge = clamp(spot_return * 8.0, -0.12, 0.12)
        imbalance_edge = clamp(imbalance * 0.08, -0.08, 0.08)
        decay = 0.6 + 0.4 * time_factor
        delta = (ret_edge + imbalance_edge) * decay
        return clamp(market_prob + delta, 0.01, 0.99)

    def _candidate_buy(
        self,
        *,
        fair_probability: float,
        ask: float,
        fee_bps: int,
        alpha_mode: bool = True,
    ) -> tuple[float, float] | None:
        if ask <= 0:
            return None
        fair = clamp(fair_probability, 0.01, 0.99)
        fee = per_share_fee(ask, fee_bps)
        edge_net = fair - ask - fee - self.config.directional_slippage_buffer
        edge_trigger = max(0.003, self.config.directional_threshold * (0.20 if alpha_mode else 0.08))
        if edge_net <= edge_trigger:
            return None
        entry = round_tick(min(0.99, ask + 0.01))
        return entry, edge_net

    def _candidate_sell_exit(
        self,
        *,
        fair_probability: float,
        avg_entry_price: float,
        bid: float,
        fee_bps: int,
    ) -> tuple[float, float] | None:
        if bid <= 0 or avg_entry_price <= 0:
            return None
        fair = clamp(fair_probability, 0.01, 0.99)
        fee = per_share_fee(bid, fee_bps)
        realized_edge = bid - avg_entry_price - fee - self.config.directional_slippage_buffer
        fair_dislocation = bid - fair
        # Profit-only quick exit: close only when realized edge is positive and market is no longer cheap.
        if realized_edge < 0.003:
            return None
        if fair_dislocation < 0.005:
            return None
        entry = round_tick(max(0.01, bid - 0.01))
        return entry, realized_edge

    def generate(
        self,
        snapshot: MarketSnapshot,
        fair_probability: float,
        primary_inventory: float = 0.0,
        secondary_inventory: float = 0.0,
        primary_avg_entry: float = 0.0,
        secondary_avg_entry: float = 0.0,
        now: datetime | None = None,
    ) -> list[OrderIntent]:
        market = snapshot.market
        if market.timeframe not in {Timeframe.FIVE_MIN, Timeframe.FIFTEEN_MIN}:
            return []

        now = now or datetime.now(tz=timezone.utc)
        seconds_to_end = (market.end_time - now).total_seconds()
        if seconds_to_end <= 0:
            return []

        if market.timeframe == Timeframe.FIVE_MIN and seconds_to_end < self.config.directional_min_time_left_5m:
            return []
        if market.timeframe == Timeframe.FIFTEEN_MIN and seconds_to_end < self.config.directional_min_time_left_15m:
            return []

        p_fair = clamp(fair_probability, 0.01, 0.99)
        intents: list[OrderIntent] = []
        market_budget = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        market_mid = max(0.02, (snapshot.primary_book.mid + (1.0 - snapshot.secondary_book.mid)) / 2.0)
        rebalance_band = max(market.order_min_size, (market_budget / market_mid) * 0.70)
        gross_inventory = max(0.0, primary_inventory) + max(0.0, secondary_inventory)
        net_delta = primary_inventory - secondary_inventory
        imbalance_ratio = abs(net_delta) / max(market.order_min_size, gross_inventory)
        target_net_gross_ratio = 0.18
        hard_net_gross_ratio = 0.28
        rebalance_required = gross_inventory >= (market.order_min_size * 2.0) and imbalance_ratio >= hard_net_gross_ratio
        rebalance_target_side = ""
        if net_delta > rebalance_band or (
            gross_inventory >= (market.order_min_size * 2.0) and imbalance_ratio > target_net_gross_ratio and net_delta > 0
        ):
            rebalance_target_side = "secondary"
        elif net_delta < -rebalance_band or (
            gross_inventory >= (market.order_min_size * 2.0) and imbalance_ratio > target_net_gross_ratio and net_delta < 0
        ):
            rebalance_target_side = "primary"

        buy_candidates: list[tuple[str, str, float, float, int, float, float]] = []

        primary_candidate = self._candidate_buy(
            fair_probability=p_fair,
            ask=snapshot.primary_book.best_ask,
            fee_bps=snapshot.primary_fee.base_fee,
            alpha_mode=True,
        )
        if primary_candidate is not None:
            entry, edge_net = primary_candidate
            buy_candidates.append(
                (
                    "primary",
                    market.primary_token_id,
                    entry,
                    edge_net,
                    snapshot.primary_fee.base_fee,
                    snapshot.primary_book.best_bid,
                    snapshot.primary_book.best_ask,
                )
            )

        secondary_candidate = self._candidate_buy(
            fair_probability=1.0 - p_fair,
            ask=snapshot.secondary_book.best_ask,
            fee_bps=snapshot.secondary_fee.base_fee,
            alpha_mode=True,
        )
        if secondary_candidate is not None:
            entry, edge_net = secondary_candidate
            buy_candidates.append(
                (
                    "secondary",
                    market.secondary_token_id,
                    entry,
                    edge_net,
                    snapshot.secondary_fee.base_fee,
                    snapshot.secondary_book.best_bid,
                    snapshot.secondary_book.best_ask,
                )
            )

        dual_mode_added = False
        dual_alpha_floor = max(0.010, self.config.directional_threshold * 0.25)
        if (
            len(buy_candidates) == 2
            and not rebalance_required
            and imbalance_ratio <= target_net_gross_ratio
            and all(candidate[3] >= dual_alpha_floor for candidate in buy_candidates)
        ):
            dual_mode_added = True
            for side_label, token_id, entry, edge_net, fee_bps, bid, ask in buy_candidates:
                confidence = clamp(edge_net / 0.08, 0.0, 1.0)
                target_notional = min(
                    self.config.directional_notional * 0.65,
                    max(1.0, market_budget * (0.18 + 0.16 * confidence)),
                )
                size = max(market.order_min_size, target_notional / max(entry, 0.01))
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=token_id,
                        side=Side.BUY,
                        price=entry,
                        size=size,
                        tif=TimeInForce.IOC,
                        post_only=False,
                        engine="engine_short_dir",
                        expected_edge=edge_net,
                        metadata={
                            "strategy": "short-directional",
                            "intent_type": "dual_alpha",
                            "timeframe": market.timeframe.value,
                            "p_fair": p_fair,
                            "picked_side": side_label,
                            "fee_bps": fee_bps,
                            "edge_net": edge_net,
                            "best_bid": bid,
                            "best_ask": ask,
                            "seconds_to_end": seconds_to_end,
                            "inventory_primary": primary_inventory,
                            "inventory_secondary": secondary_inventory,
                            "net_delta": net_delta,
                            "gross_inventory": gross_inventory,
                            "net_gross_ratio": imbalance_ratio,
                            "target_net_gross_ratio": target_net_gross_ratio,
                            "order_min_size": market.order_min_size,
                        },
                    )
                )

        # Alpha leg: buy most mispriced side (suppressed if forced rebalance would be worsened).
        if buy_candidates and not dual_mode_added:
            buy_candidates.sort(key=lambda x: x[3], reverse=True)
            side_label, token_id, entry, edge_net, fee_bps, bid, ask = buy_candidates[0]
            if rebalance_required and rebalance_target_side and side_label != rebalance_target_side:
                pass
            else:
                confidence = clamp(edge_net / 0.08, 0.0, 1.0)
                target_notional = min(
                    self.config.directional_notional,
                    max(1.0, market_budget * (0.50 + 0.40 * confidence)),
                )
                size = max(market.order_min_size, target_notional / max(entry, 0.01))
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=token_id,
                        side=Side.BUY,
                        price=entry,
                        size=size,
                        tif=TimeInForce.IOC,
                        post_only=False,
                        engine="engine_short_dir",
                        expected_edge=edge_net,
                        metadata={
                            "strategy": "short-directional",
                            "intent_type": "alpha",
                            "timeframe": market.timeframe.value,
                            "p_fair": p_fair,
                            "picked_side": side_label,
                            "fee_bps": fee_bps,
                            "edge_net": edge_net,
                            "best_bid": bid,
                            "best_ask": ask,
                            "seconds_to_end": seconds_to_end,
                            "inventory_primary": primary_inventory,
                            "inventory_secondary": secondary_inventory,
                            "net_delta": net_delta,
                            "gross_inventory": gross_inventory,
                            "net_gross_ratio": imbalance_ratio,
                            "target_net_gross_ratio": target_net_gross_ratio,
                            "order_min_size": market.order_min_size,
                        },
                    )
                )

        # Equalizer leg: if inventory drifts too directional, bias to opposite side.
        if rebalance_target_side:
            if rebalance_target_side == "primary":
                rebalance_candidate = self._candidate_buy(
                    fair_probability=p_fair,
                    ask=snapshot.primary_book.best_ask,
                    fee_bps=snapshot.primary_fee.base_fee,
                    alpha_mode=False,
                )
                token_id = market.primary_token_id
                fee_bps = snapshot.primary_fee.base_fee
                best_bid = snapshot.primary_book.best_bid
                best_ask = snapshot.primary_book.best_ask
            else:
                rebalance_candidate = self._candidate_buy(
                    fair_probability=1.0 - p_fair,
                    ask=snapshot.secondary_book.best_ask,
                    fee_bps=snapshot.secondary_fee.base_fee,
                    alpha_mode=False,
                )
                token_id = market.secondary_token_id
                fee_bps = snapshot.secondary_fee.base_fee
                best_bid = snapshot.secondary_book.best_bid
                best_ask = snapshot.secondary_book.best_ask
            if rebalance_candidate is not None:
                entry, edge_net = rebalance_candidate
                imbalance_excess = max(0.0, abs(net_delta) - rebalance_band)
                ratio_excess = max(0.0, imbalance_ratio - target_net_gross_ratio)
                excess_factor = clamp(
                    max(imbalance_excess / max(1.0, rebalance_band), ratio_excess / max(0.05, target_net_gross_ratio)),
                    0.2,
                    1.4,
                )
                target_notional = min(
                    self.config.directional_notional,
                    max(1.0, market_budget * (0.30 + 0.35 * excess_factor)),
                )
                size = max(market.order_min_size, target_notional / max(entry, 0.01))
                intent_type = "rebalance_forced" if rebalance_required else "rebalance"
                priority_edge = edge_net + (0.025 if rebalance_required else 0.0)
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=token_id,
                        side=Side.BUY,
                        price=entry,
                        size=size,
                        tif=TimeInForce.IOC,
                        post_only=False,
                        engine="engine_short_dir",
                        expected_edge=priority_edge,
                        metadata={
                            "strategy": "short-directional",
                            "intent_type": intent_type,
                            "timeframe": market.timeframe.value,
                            "p_fair": p_fair,
                            "picked_side": rebalance_target_side,
                            "fee_bps": fee_bps,
                            "edge_net": edge_net,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "seconds_to_end": seconds_to_end,
                            "inventory_primary": primary_inventory,
                            "inventory_secondary": secondary_inventory,
                            "rebalance_band": rebalance_band,
                            "net_delta": net_delta,
                            "gross_inventory": gross_inventory,
                            "net_gross_ratio": imbalance_ratio,
                            "target_net_gross_ratio": target_net_gross_ratio,
                            "order_min_size": market.order_min_size,
                        },
                    )
                )

        # Fast exit leg: monetize short-term mispricing pops or reduce adverse inventory.
        primary_exit = self._candidate_sell_exit(
            fair_probability=p_fair,
            avg_entry_price=primary_avg_entry,
            bid=snapshot.primary_book.best_bid,
            fee_bps=snapshot.primary_fee.base_fee,
        )
        if primary_exit is not None and primary_inventory >= market.order_min_size:
            exit_price, exit_edge = primary_exit
            exit_size = min(primary_inventory, max(market.order_min_size, (market_budget * 0.50) / max(exit_price, 0.01)))
            intents.append(
                OrderIntent(
                    market_id=market.market_id,
                    token_id=market.primary_token_id,
                    side=Side.SELL,
                    price=exit_price,
                    size=exit_size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_short_dir",
                    expected_edge=exit_edge,
                    metadata={
                        "strategy": "short-directional",
                        "intent_type": "take_profit_exit",
                        "timeframe": market.timeframe.value,
                        "p_fair": p_fair,
                        "picked_side": "primary",
                        "fee_bps": snapshot.primary_fee.base_fee,
                        "edge_net": exit_edge,
                        "best_bid": snapshot.primary_book.best_bid,
                        "best_ask": snapshot.primary_book.best_ask,
                        "seconds_to_end": seconds_to_end,
                        "avg_entry": primary_avg_entry,
                        "inventory_primary": primary_inventory,
                        "net_delta": net_delta,
                        "gross_inventory": gross_inventory,
                        "net_gross_ratio": imbalance_ratio,
                        "target_net_gross_ratio": target_net_gross_ratio,
                        "order_min_size": market.order_min_size,
                    },
                )
            )

        secondary_exit = self._candidate_sell_exit(
            fair_probability=1.0 - p_fair,
            avg_entry_price=secondary_avg_entry,
            bid=snapshot.secondary_book.best_bid,
            fee_bps=snapshot.secondary_fee.base_fee,
        )
        if secondary_exit is not None and secondary_inventory >= market.order_min_size:
            exit_price, exit_edge = secondary_exit
            exit_size = min(secondary_inventory, max(market.order_min_size, (market_budget * 0.50) / max(exit_price, 0.01)))
            intents.append(
                OrderIntent(
                    market_id=market.market_id,
                    token_id=market.secondary_token_id,
                    side=Side.SELL,
                    price=exit_price,
                    size=exit_size,
                    tif=TimeInForce.IOC,
                    post_only=False,
                    engine="engine_short_dir",
                    expected_edge=exit_edge,
                    metadata={
                        "strategy": "short-directional",
                        "intent_type": "take_profit_exit",
                        "timeframe": market.timeframe.value,
                        "p_fair": 1.0 - p_fair,
                        "picked_side": "secondary",
                        "fee_bps": snapshot.secondary_fee.base_fee,
                        "edge_net": exit_edge,
                        "best_bid": snapshot.secondary_book.best_bid,
                        "best_ask": snapshot.secondary_book.best_ask,
                        "seconds_to_end": seconds_to_end,
                        "avg_entry": secondary_avg_entry,
                        "inventory_secondary": secondary_inventory,
                        "net_delta": net_delta,
                        "gross_inventory": gross_inventory,
                        "net_gross_ratio": imbalance_ratio,
                        "target_net_gross_ratio": target_net_gross_ratio,
                        "order_min_size": market.order_min_size,
                    },
                )
            )

        if not intents:
            return []

        # Rebalance/exit must outrank fresh alpha when inventory control requires it.
        rank = {
            "take_profit_exit": 0,
            "rebalance_forced": 1,
            "rebalance": 2,
            "dual_alpha": 3,
            "alpha": 4,
        }
        intents.sort(key=lambda x: (rank.get(str(x.metadata.get("intent_type") or "alpha"), 9), -x.expected_edge))
        capped: list[OrderIntent] = []
        seen: set[tuple[str, str]] = set()
        for intent in intents:
            key = (intent.token_id, intent.side.value)
            if key in seen:
                continue
            seen.add(key)
            capped.append(intent)
            if len(capped) >= 3:
                break
        return capped

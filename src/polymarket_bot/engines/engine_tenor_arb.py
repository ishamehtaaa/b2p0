from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from polymarket_bot.config import BotConfig
from polymarket_bot.models import MarketSnapshot, OrderIntent, Side, TimeInForce, Timeframe
from polymarket_bot.pricing import clamp, directional_ev_per_share, per_share_fee, round_tick


@dataclass
class TenorDislocationEngine:
    config: BotConfig

    def generate(
        self,
        snapshots: list[MarketSnapshot],
        fair_by_market: dict[str, float],
        now: datetime | None = None,
    ) -> list[OrderIntent]:
        now = now or datetime.now(tz=timezone.utc)
        by_tf: dict[Timeframe, MarketSnapshot] = {}
        for snapshot in snapshots:
            current = by_tf.get(snapshot.market.timeframe)
            if current is None or snapshot.market.end_time < current.market.end_time:
                by_tf[snapshot.market.timeframe] = snapshot

        one_h = by_tf.get(Timeframe.ONE_HOUR)
        if one_h is None:
            return []
        one_h_fair = fair_by_market.get(one_h.market.market_id)
        if one_h_fair is None:
            return []

        intents: list[OrderIntent] = []
        for short_tf in (Timeframe.FIVE_MIN, Timeframe.FIFTEEN_MIN):
            short = by_tf.get(short_tf)
            if short is None:
                continue
            short_fair = fair_by_market.get(short.market.market_id)
            if short_fair is None:
                continue
            seconds_left = (short.market.end_time - now).total_seconds()
            if short_tf == Timeframe.FIVE_MIN and seconds_left < self.config.directional_min_time_left_5m:
                continue
            if short_tf == Timeframe.FIFTEEN_MIN and seconds_left < self.config.directional_min_time_left_15m:
                continue

            dislocation = one_h_fair - short_fair
            trigger = max(0.02, self.config.tenor_dislocation_threshold * 0.5)
            if abs(dislocation) < trigger:
                continue

            if dislocation > 0:
                # Short tenor underpriced vs 1h reference: buy primary.
                token_id = short.market.primary_token_id
                fair = clamp(short_fair + dislocation * 0.4, 0.01, 0.99)
                bid = short.primary_book.best_bid
                ask = short.primary_book.best_ask
                fee_bps = short.primary_fee.base_fee
                side_name = "primary"
            else:
                # Short tenor overpriced vs 1h reference: buy secondary.
                token_id = short.market.secondary_token_id
                fair = clamp((1.0 - short_fair) + abs(dislocation) * 0.4, 0.01, 0.99)
                bid = short.secondary_book.best_bid
                ask = short.secondary_book.best_ask
                fee_bps = short.secondary_fee.base_fee
                side_name = "secondary"

            required_discount = per_share_fee(fair, fee_bps) + self.config.directional_slippage_buffer + 0.002
            entry = round_tick(fair - required_discount)
            if bid > 0:
                entry = max(entry, round_tick(bid + 0.01))
            if ask > 0:
                entry = min(entry, round_tick(ask - 0.01))
            if ask > 0 and bid > 0 and entry >= ask:
                continue

            ev = directional_ev_per_share(
                fair_probability=min(0.99, fair),
                entry_price=entry,
                base_fee_bps=fee_bps,
                slippage_buffer=self.config.directional_slippage_buffer,
            )
            if ev <= 0:
                continue

            market_budget = self.config.bankroll_usdc * self.config.max_market_exposure_pct
            target_notional = min(self.config.tenor_arb_notional, max(1.0, market_budget * 0.50))
            size = max(short.market.order_min_size, target_notional / max(entry, 0.01))
            intents.append(
                OrderIntent(
                    market_id=short.market.market_id,
                    token_id=token_id,
                    side=Side.BUY,
                    price=entry,
                    size=size,
                    tif=TimeInForce.GTC,
                    post_only=True,
                    engine="engine_tenor_arb",
                    expected_edge=ev,
                    metadata={
                        "strategy": "tenor-dislocation",
                        "short_tf": short_tf.value,
                        "one_h_market": one_h.market.market_id,
                        "one_h_fair": one_h_fair,
                        "short_fair": short_fair,
                        "dislocation": dislocation,
                        "picked_side": side_name,
                    },
                )
            )
        return intents

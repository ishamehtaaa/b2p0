from __future__ import annotations

from dataclasses import dataclass

from polymarket_bot.config import BotConfig
from polymarket_bot.models import MarketSnapshot, OrderIntent, Side, TimeInForce, Timeframe
from polymarket_bot.pricing import clamp, per_share_fee, round_tick


@dataclass
class HourMakerEngine:
    config: BotConfig

    def generate(
        self,
        snapshot: MarketSnapshot,
        inventory_shares: float,
        fair_probability: float,
    ) -> list[OrderIntent]:
        market = snapshot.market
        if market.timeframe != Timeframe.ONE_HOUR:
            return []
        if snapshot.primary_fee.base_fee >= self.config.fee_disable_maker_bps:
            return []

        fair = clamp(fair_probability, 0.02, 0.98)
        spread = max(snapshot.primary_book.spread, self.config.maker_min_spread)
        if spread <= 0:
            return []

        bid_depth, ask_depth = snapshot.primary_book.top_depth_usdc(levels=2)
        # Keep a lightweight sanity floor so dead books are skipped without blocking viable markets.
        if bid_depth < 10.0 and ask_depth < 10.0:
            return []

        tick = 0.01
        fee_edge = per_share_fee(fair, snapshot.primary_fee.base_fee)
        # Cap extreme spreads to avoid quoting at useless edges when books are 0.01/0.99 wide.
        effective_spread = min(0.20, max(0.02, spread))
        half_spread = clamp((effective_spread / 2.0) + fee_edge + 0.004, 0.01, 0.12)

        inv_ratio = clamp(inventory_shares / max(1.0, self.config.maker_max_inventory_shares), -1.0, 1.0)
        # Positive inventory skews quotes lower to reduce long inventory.
        skew = inv_ratio * 0.02

        bid_price = round_tick(fair - half_spread - skew, tick=tick)
        ask_price = round_tick(fair + half_spread - skew, tick=tick)

        best_ask = snapshot.primary_book.best_ask
        best_bid = snapshot.primary_book.best_bid
        if best_ask > 0 and bid_price >= best_ask:
            bid_price = round_tick(best_ask - tick, tick=tick)
        if best_bid > 0 and ask_price <= best_bid:
            ask_price = round_tick(best_bid + tick, tick=tick)

        if ask_price - bid_price < tick:
            return []

        market_budget = self.config.bankroll_usdc * self.config.max_market_exposure_pct
        per_quote_notional = min(self.config.maker_quote_notional, max(1.0, market_budget * 0.45))
        bid_size = max(snapshot.market.order_min_size, per_quote_notional / max(bid_price, 0.01))
        metadata = {
            "strategy": "1h-maker",
            "inventory_shares": inventory_shares,
            "fair_probability": fair,
            "bid_depth_top2_usdc": bid_depth,
            "ask_depth_top2_usdc": ask_depth,
        }
        edge = max(0.0, (ask_price - bid_price) / 2.0 - fee_edge)

        intents = [
            OrderIntent(
                market_id=market.market_id,
                token_id=market.primary_token_id,
                side=Side.BUY,
                price=bid_price,
                size=bid_size,
                tif=TimeInForce.GTC,
                post_only=True,
                engine="engine_1h_maker",
                expected_edge=edge,
                metadata={**metadata, "quote_side": "bid"},
            )
        ]

        # Ask quotes require inventory in the token. Skip until position exists.
        if inventory_shares > 0:
            ask_size_target = max(snapshot.market.order_min_size, per_quote_notional / max(ask_price, 0.01))
            ask_size = min(ask_size_target, inventory_shares)
            if ask_size >= snapshot.market.order_min_size:
                intents.append(
                    OrderIntent(
                        market_id=market.market_id,
                        token_id=market.primary_token_id,
                        side=Side.SELL,
                        price=ask_price,
                        size=ask_size,
                        tif=TimeInForce.GTC,
                        post_only=True,
                        engine="engine_1h_maker",
                        expected_edge=edge,
                        metadata={**metadata, "quote_side": "ask"},
                    )
                )

        return intents

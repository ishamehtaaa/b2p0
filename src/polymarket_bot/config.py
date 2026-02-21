from __future__ import annotations

from dataclasses import dataclass
import os


ALL_TAGS = (102892, 102467, 102175)
DEFAULT_TAGS = ALL_TAGS


@dataclass(frozen=True)
class BotConfig:
    mode: str
    gamma_url: str
    data_api_url: str
    clob_url: str
    clob_ws_url: str
    polygon_rpc_url: str
    btc_spot_url: str
    btc_spot_ws_url: str
    database_path: str
    api_timeout_seconds: float

    enabled_tags: tuple[int, ...]
    poll_interval_seconds: float
    fee_poll_interval_seconds: float
    max_markets_per_tag: int
    max_trade_markets_5m: int
    max_trade_markets_15m: int
    max_trade_markets_1h: int

    bankroll_usdc: float
    canary_cap_pct: float
    max_daily_dd_pct: float
    max_total_exposure_pct: float
    max_market_exposure_pct: float
    max_consecutive_exec_errors: int
    stale_feed_seconds: float
    pair_max_intents_per_cycle: int
    use_reference_trader_learning: bool

    maker_min_spread: float
    maker_min_depth_usdc_top2: float
    maker_quote_notional: float
    maker_max_inventory_shares: float
    fee_disable_maker_bps: int

    directional_threshold: float
    directional_notional: float
    directional_slippage_buffer: float
    directional_min_time_left_5m: int
    directional_min_time_left_15m: int

    tenor_dislocation_threshold: float
    tenor_arb_notional: float

    allowed_region: str
    excluded_provinces: tuple[str, ...]
    bot_region: str
    bot_province: str

    poly_private_key: str
    poly_proxy_address: str
    poly_proxy_factory_address: str
    poly_chain_id: int
    poly_signature_type: int | None

    log_level: str

    @property
    def live_mode(self) -> bool:
        return self.mode.lower() == "live"

    @property
    def paper_mode(self) -> bool:
        return not self.live_mode

    @property
    def live_permitted(self) -> bool:
        if not self.live_mode:
            return True
        if self.bot_region.upper() != self.allowed_region.upper():
            return False
        if self.bot_province.upper() in set(self.excluded_provinces):
            return False
        return True


def load_config() -> BotConfig:
    raw_reference_learning = os.getenv("ENABLE_REFERENCE_LEARNING", "").strip().lower()
    use_reference_learning = raw_reference_learning in {"1", "true", "yes", "y", "on"}
    raw_signature_type = os.getenv("POLY_SIGNATURE_TYPE", "").strip()
    parsed_signature_type: int | None = None
    if raw_signature_type:
        try:
            parsed_signature_type = int(raw_signature_type)
        except ValueError:
            parsed_signature_type = None

    # Runtime profile: pair-arb with RNJD fair value and bounded short-tenor exposure.
    return BotConfig(
        mode=os.getenv("BOT_MODE", "paper").strip().lower(),
        gamma_url="https://gamma-api.polymarket.com",
        data_api_url="https://data-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
        clob_ws_url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        polygon_rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        btc_spot_url="https://api.coinbase.com/v2/prices/BTC-USD/spot",
        btc_spot_ws_url="wss://stream.binance.com:9443/ws/btcusdt@trade",
        database_path=os.getenv("BOT_DB_PATH", "data/bot.db"),
        api_timeout_seconds=3.0,
        enabled_tags=DEFAULT_TAGS,
        poll_interval_seconds=1.0,
        fee_poll_interval_seconds=60.0,
        max_markets_per_tag=120,
        max_trade_markets_5m=3,
        max_trade_markets_15m=3,
        max_trade_markets_1h=3,
        bankroll_usdc=5000.0,
        canary_cap_pct=0.10,
        max_daily_dd_pct=0.25,
        max_total_exposure_pct=0.95,
        max_market_exposure_pct=0.60,
        max_consecutive_exec_errors=3,
        stale_feed_seconds=5.0,
        pair_max_intents_per_cycle=max(2, int(os.getenv("PAIR_MAX_INTENTS", "18"))),
        use_reference_trader_learning=use_reference_learning,
        maker_min_spread=0.01,
        maker_min_depth_usdc_top2=250.0,
        maker_quote_notional=25.0,
        maker_max_inventory_shares=300.0,
        fee_disable_maker_bps=800,
        directional_threshold=0.035,
        directional_notional=25.0,
        directional_slippage_buffer=0.002,
        directional_min_time_left_5m=90,
        directional_min_time_left_15m=120,
        tenor_dislocation_threshold=0.06,
        tenor_arb_notional=20.0,
        allowed_region="CA",
        excluded_provinces=("ON",),
        bot_region=os.getenv("BOT_REGION", "CA").upper(),
        bot_province=os.getenv("BOT_PROVINCE", "BC").upper(),
        poly_private_key=os.getenv("POLY_PRIVATE_KEY", os.getenv("PRIVATE_KEY", "")),
        poly_proxy_address=os.getenv("POLY_PROXY_ADDRESS", ""),
        poly_proxy_factory_address=os.getenv(
            "POLY_PROXY_FACTORY_ADDRESS",
            "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052",
        ),
        poly_chain_id=137,
        poly_signature_type=parsed_signature_type,
        log_level="INFO",
    )

# Polymarket Pair-Arb Bot

Lean BTC/ETH short-tenor bot with a pair-arbitrage engine:

- Uses Binance BTC ticks + RNJD fair-value estimation.
- Detects underpriced binary legs on Polymarket.
- Enters the single underpriced side first (`alpha_entry`) when fair-value edge is positive.
- Rapidly equalizes to the opposite side after entry (forced/soft equalizers) so one-sided exposure stays brief.

The strategy and risk parameters are locked in code.  
`.env` is only for runtime essentials.

## Quick Start

```bash
cp .env.example .env
python3 -m pip install websocket-client
python3 bot.py run --mode paper --bankroll 100
```

Run continuous loop:

```bash
python3 bot.py run --mode paper --bankroll 100
```

Live mode:

```bash
python3 bot.py run --mode live --bankroll 100
```

## Minimal `.env`

```bash
BOT_MODE=paper
POLY_PRIVATE_KEY=
# PRIVATE_KEY=  # fallback alias if you prefer docs naming
POLY_PROXY_ADDRESS=
BOT_REGION=CA
BOT_PROVINCE=BC
BOT_DB_PATH=data/bot.db
PAIR_MAX_INTENTS=18
```

## Notes

- Ontario is blocked by compliance gate.
- No strategy guarantees profitability.
- `Ctrl+C` once = graceful stop, twice = force exit.
- There is no `--once` and no `fees` command.
- Live mode runs an auth preflight at startup and exits immediately if key/proxy/signature settings are invalid.
- Spot feed uses Binance WebSocket (`btcusdt@trade`) in strict mode (no REST fallback).
- CLOB order books use Polymarket market-channel WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) in strict mode (no REST fallback); markets without live WS books are skipped.
- Pair strategy runs on 5m/15m/1h up/down markets, but only enters while enough time remains before expiry.
- Settlement automation attempts `MERGE` (complete-set realization) and periodic `REDEEM`; if the live client lacks those methods, the bot logs and continues trading.

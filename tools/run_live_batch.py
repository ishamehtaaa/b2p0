#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fallback_load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        _fallback_load_dotenv(env_path)


def _safe_remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _latest_equity(db_path: Path) -> float | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT current_equity FROM risk_state ORDER BY ts DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return float(row[0])


def _equity_snapshot(db_path: Path, bankroll: float) -> tuple[float, float, float, float]:
    if not db_path.exists():
        return (bankroll, bankroll, bankroll, bankroll)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT current_equity FROM risk_state ORDER BY ts ASC").fetchall()
    finally:
        conn.close()
    if not rows:
        return (bankroll, bankroll, bankroll, bankroll)
    vals = [float(r[0]) for r in rows]
    return (vals[0], vals[-1], min(vals), max(vals))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated live bot sessions and summarize each.")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--bankroll", type=float, default=50.0)
    parser.add_argument("--session-seconds", type=int, default=80)
    parser.add_argument("--stop-loss", type=float, default=3.0, help="Stop session when equity <= bankroll-stop-loss")
    parser.add_argument("--timeframe", choices=("all", "5m", "15m", "1h"), default="5m")
    parser.add_argument("--prefix", default="/tmp/live_batch")
    args = parser.parse_args()

    if args.sessions <= 0:
        raise SystemExit("--sessions must be > 0")
    if args.bankroll <= 0:
        raise SystemExit("--bankroll must be > 0")
    if args.session_seconds < 10:
        raise SystemExit("--session-seconds must be >= 10")
    if args.stop_loss < 0:
        raise SystemExit("--stop-loss must be >= 0")

    _load_dotenv()
    stop_loss_equity = float(args.bankroll) - float(args.stop_loss)
    summary_path = Path(f"{args.prefix}_summary.json")
    summary: list[dict[str, object]] = []

    base_env = os.environ.copy()
    if args.timeframe == "5m":
        base_env["BOT_ENABLED_TAGS"] = "102892"
        base_env["MAX_TRADE_MARKETS_15M"] = "0"
        base_env["MAX_TRADE_MARKETS_1H"] = "0"
    elif args.timeframe == "15m":
        base_env["BOT_ENABLED_TAGS"] = "102467"
        base_env["MAX_TRADE_MARKETS_5M"] = "0"
        base_env["MAX_TRADE_MARKETS_1H"] = "0"
    elif args.timeframe == "1h":
        base_env["BOT_ENABLED_TAGS"] = "102175"
        base_env["MAX_TRADE_MARKETS_5M"] = "0"
        base_env["MAX_TRADE_MARKETS_15M"] = "0"

    for i in range(1, int(args.sessions) + 1):
        db_path = Path(f"{args.prefix}_{i}.db")
        log_path = Path(f"{args.prefix}_{i}.log")
        _safe_remove(db_path)
        _safe_remove(log_path)

        env = dict(base_env)
        env["BOT_DB_PATH"] = str(db_path)

        stop_reason = "natural_exit"
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                ["python", "bot.py", "run", "--mode", "live", "--bankroll", str(args.bankroll)],
                cwd=ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            started = time.time()
            while proc.poll() is None:
                elapsed = time.time() - started
                if elapsed >= float(args.session_seconds):
                    stop_reason = "max_session_seconds"
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    break

                equity = _latest_equity(db_path)
                if equity is not None and equity <= stop_loss_equity:
                    stop_reason = "stop_loss_equity"
                    try:
                        proc.send_signal(signal.SIGINT)
                    except Exception:
                        pass
                    break
                time.sleep(1.0)

            if proc.poll() is None:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)

            rc = proc.returncode
            if stop_reason == "natural_exit" and rc not in (0, None):
                stop_reason = f"exit_{rc}"

        conn = sqlite3.connect(db_path)
        try:
            orders = _scalar(conn, "SELECT COUNT(*) FROM orders")
            fills = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE filled_size > 0")
            buy_orders = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE lower(side)=?", ("buy",))
            sell_orders = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE lower(side)=?", ("sell",))
            buy_fills = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE lower(side)=? AND filled_size > 0", ("buy",))
            sell_fills = _scalar(conn, "SELECT COUNT(*) FROM orders WHERE lower(side)=? AND filled_size > 0", ("sell",))
            markets = [str(r[0]) for r in conn.execute("SELECT DISTINCT market_id FROM orders WHERE filled_size > 0 ORDER BY market_id").fetchall()]
            timeframes = [str(r[0]) for r in conn.execute("SELECT DISTINCT timeframe FROM market_snapshots ORDER BY timeframe").fetchall()]
        finally:
            conn.close()

        equity_start, equity_end, equity_min, equity_max = _equity_snapshot(db_path, float(args.bankroll))
        pnl = equity_end - float(args.bankroll)
        item = {
            "session": i,
            "stop_reason": stop_reason,
            "orders": orders,
            "fills": fills,
            "buy_orders": buy_orders,
            "sell_orders": sell_orders,
            "buy_fills": buy_fills,
            "sell_fills": sell_fills,
            "market_count": len(markets),
            "markets": markets,
            "timeframes_seen": timeframes,
            "equity_start": equity_start,
            "equity_end": equity_end,
            "equity_min": equity_min,
            "equity_max": equity_max,
            "pnl": pnl,
            "db": str(db_path),
            "log": str(log_path),
        }
        summary.append(item)
        print(
            "session={session} stop={stop} orders={orders} fills={fills} buys={buys} sells={sells} "
            "markets={mkts} timeframes={tfs} pnl={pnl:.4f}".format(
                session=i,
                stop=stop_reason,
                orders=orders,
                fills=fills,
                buys=buy_orders,
                sells=sell_orders,
                mkts=len(markets),
                tfs=",".join(timeframes) if timeframes else "none",
                pnl=pnl,
            ),
            flush=True,
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_file {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

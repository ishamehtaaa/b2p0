from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import sqlite3
from typing import Any

from polymarket_bot.models import MarketSnapshot, OrderIntent, OrderResult, RiskState


class Storage:
    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
              ts TEXT NOT NULL,
              market_id TEXT NOT NULL,
              tag_id INTEGER NOT NULL,
              timeframe TEXT NOT NULL,
              question TEXT NOT NULL,
              primary_token_id TEXT NOT NULL,
              secondary_token_id TEXT NOT NULL,
              primary_bid REAL NOT NULL,
              primary_ask REAL NOT NULL,
              primary_mid REAL NOT NULL,
              primary_spread REAL NOT NULL,
              secondary_bid REAL NOT NULL,
              secondary_ask REAL NOT NULL,
              secondary_mid REAL NOT NULL,
              primary_fee_bps INTEGER NOT NULL,
              secondary_fee_bps INTEGER NOT NULL,
              imbalance REAL NOT NULL,
              liquidity_num REAL NOT NULL,
              volume_24h REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
              ts TEXT NOT NULL,
              order_id TEXT NOT NULL,
              market_id TEXT NOT NULL,
              token_id TEXT NOT NULL,
              side TEXT NOT NULL,
              price REAL NOT NULL,
              size REAL NOT NULL,
              status TEXT NOT NULL,
              filled_size REAL NOT NULL,
              filled_price REAL NOT NULL,
              fee_paid REAL NOT NULL,
              engine TEXT NOT NULL,
              mode TEXT NOT NULL,
              metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS risk_events (
              ts TEXT NOT NULL,
              kind TEXT NOT NULL,
              details TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS risk_state (
              ts TEXT NOT NULL,
              current_equity REAL NOT NULL,
              daily_drawdown_pct REAL NOT NULL,
              total_exposure REAL NOT NULL,
              halted INTEGER NOT NULL,
              halted_reason TEXT NOT NULL,
              consecutive_exec_errors INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pair_learning_events (
              ts TEXT NOT NULL,
              market_id TEXT NOT NULL,
              timeframe TEXT NOT NULL,
              entry_side TEXT NOT NULL,
              sec_bucket INTEGER NOT NULL,
              pair_price_cost REAL NOT NULL,
              hedge_delay_seconds REAL NOT NULL,
              success INTEGER NOT NULL,
              source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pair_learning_stats (
              timeframe TEXT NOT NULL,
              entry_side TEXT NOT NULL,
              sec_bucket INTEGER NOT NULL,
              samples INTEGER NOT NULL,
              successes INTEGER NOT NULL,
              ewma_pair_price REAL NOT NULL,
              ewma_hedge_delay REAL NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (timeframe, entry_side, sec_bucket)
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record_snapshot(self, snapshot: MarketSnapshot) -> None:
        self.conn.execute(
            """
            INSERT INTO market_snapshots (
              ts, market_id, tag_id, timeframe, question,
              primary_token_id, secondary_token_id,
              primary_bid, primary_ask, primary_mid, primary_spread,
              secondary_bid, secondary_ask, secondary_mid,
              primary_fee_bps, secondary_fee_bps, imbalance,
              liquidity_num, volume_24h
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(tz=timezone.utc).isoformat(),
                snapshot.market.market_id,
                snapshot.market.tag_id,
                snapshot.market.timeframe.value,
                snapshot.market.question,
                snapshot.market.primary_token_id,
                snapshot.market.secondary_token_id,
                snapshot.primary_book.best_bid,
                snapshot.primary_book.best_ask,
                snapshot.primary_mid,
                snapshot.primary_book.spread,
                snapshot.secondary_book.best_bid,
                snapshot.secondary_book.best_ask,
                snapshot.secondary_mid,
                snapshot.primary_fee.base_fee,
                snapshot.secondary_fee.base_fee,
                snapshot.imbalance,
                snapshot.market.liquidity_num,
                snapshot.market.volume_24h,
            ),
        )
        self.conn.commit()

    def record_order(self, intent: OrderIntent, result: OrderResult, mode: str) -> None:
        metadata_payload = dict(intent.metadata)
        metadata_payload["_execution"] = {
            "status": result.status,
            "filled_size": result.filled_size,
            "filled_price": result.filled_price,
            "fee_paid": result.fee_paid,
            "raw": result.raw,
        }
        self.conn.execute(
            """
            INSERT INTO orders (
              ts, order_id, market_id, token_id, side, price, size, status,
              filled_size, filled_price, fee_paid, engine, mode, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(tz=timezone.utc).isoformat(),
                result.order_id,
                result.market_id,
                result.token_id,
                result.side.value,
                result.price,
                result.size,
                result.status,
                result.filled_size,
                result.filled_price,
                result.fee_paid,
                result.engine,
                mode,
                json.dumps(metadata_payload, separators=(",", ":"), default=str),
            ),
        )
        self.conn.commit()

    def record_risk_event(self, kind: str, details: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO risk_events (ts, kind, details) VALUES (?, ?, ?)",
            (
                datetime.now(tz=timezone.utc).isoformat(),
                kind,
                json.dumps(details, separators=(",", ":"), default=str),
            ),
        )
        self.conn.commit()

    def record_risk_state(self, state: RiskState) -> None:
        self.conn.execute(
            """
            INSERT INTO risk_state (
              ts, current_equity, daily_drawdown_pct, total_exposure, halted,
              halted_reason, consecutive_exec_errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(tz=timezone.utc).isoformat(),
                state.current_equity,
                state.daily_drawdown_pct,
                state.total_exposure,
                1 if state.halted else 0,
                state.halted_reason,
                state.consecutive_exec_errors,
            ),
        )
        self.conn.commit()

    def record_pair_learning_event(
        self,
        *,
        market_id: str,
        timeframe: str,
        entry_side: str,
        sec_bucket: int,
        pair_price_cost: float,
        hedge_delay_seconds: float,
        success: bool,
        source: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO pair_learning_events (
              ts, market_id, timeframe, entry_side, sec_bucket, pair_price_cost,
              hedge_delay_seconds, success, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(tz=timezone.utc).isoformat(),
                market_id,
                timeframe,
                entry_side,
                int(sec_bucket),
                float(pair_price_cost),
                float(hedge_delay_seconds),
                1 if success else 0,
                source,
            ),
        )
        self.conn.commit()

    def upsert_pair_learning_stat(
        self,
        *,
        timeframe: str,
        entry_side: str,
        sec_bucket: int,
        samples: int,
        successes: int,
        ewma_pair_price: float,
        ewma_hedge_delay: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO pair_learning_stats (
              timeframe, entry_side, sec_bucket, samples, successes,
              ewma_pair_price, ewma_hedge_delay, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(timeframe, entry_side, sec_bucket) DO UPDATE SET
              samples=excluded.samples,
              successes=excluded.successes,
              ewma_pair_price=excluded.ewma_pair_price,
              ewma_hedge_delay=excluded.ewma_hedge_delay,
              updated_at=excluded.updated_at
            """,
            (
                timeframe,
                entry_side,
                int(sec_bucket),
                int(samples),
                int(successes),
                float(ewma_pair_price),
                float(ewma_hedge_delay),
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def load_pair_learning_stats(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT timeframe, entry_side, sec_bucket, samples, successes,
                   ewma_pair_price, ewma_hedge_delay
            FROM pair_learning_stats
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_pair_learning_data(self) -> None:
        self.conn.execute("DELETE FROM pair_learning_events")
        self.conn.execute("DELETE FROM pair_learning_stats")
        self.conn.commit()

    def report(self, window_hours: int) -> dict[str, Any]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
        rows = self.conn.execute(
            """
            SELECT ts, engine, token_id, side, filled_size, filled_price, fee_paid
            FROM orders
            WHERE ts >= ? AND filled_size > 0
            ORDER BY ts ASC
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        latest_mids = self._latest_token_mids()
        per_engine: dict[str, dict[str, Any]] = {}
        states: dict[tuple[str, str], tuple[float, float]] = {}  # (engine, token) -> (qty, avg_cost)
        realized_samples: dict[str, list[float]] = {}

        for row in rows:
            engine = str(row["engine"])
            token = str(row["token_id"])
            side = str(row["side"]).lower()
            size = float(row["filled_size"] or 0.0)
            price = float(row["filled_price"] or 0.0)
            fee = float(row["fee_paid"] or 0.0)
            if size <= 0:
                continue

            metric = per_engine.setdefault(
                engine,
                {
                    "fills": 0,
                    "buy_notional": 0.0,
                    "sell_notional": 0.0,
                    "fees": 0.0,
                    "closed_trades": 0,
                    "closed_pnl": 0.0,
                    "open_unrealized_pnl_est": 0.0,
                    "total_pnl_est": 0.0,
                },
            )
            metric["fills"] += 1
            metric["fees"] += fee
            key = (engine, token)
            qty, avg_cost = states.get(key, (0.0, 0.0))

            if side == "buy":
                notional = size * price
                metric["buy_notional"] += notional
                total_cost = (avg_cost * qty) + notional
                qty += size
                avg_cost = total_cost / qty if qty > 0 else 0.0
                metric["closed_pnl"] -= fee
            else:
                notional = size * price
                metric["sell_notional"] += notional
                close_size = min(size, max(0.0, qty))
                trade_pnl = 0.0
                if close_size > 0:
                    trade_pnl = (price - avg_cost) * close_size
                    qty -= close_size
                    if qty <= 1e-9:
                        qty = 0.0
                        avg_cost = 0.0
                metric["closed_trades"] += 1
                metric["closed_pnl"] += trade_pnl - fee
                realized_samples.setdefault(engine, []).append(trade_pnl - fee)

            states[key] = (qty, avg_cost)

        for (engine, token), (qty, avg_cost) in states.items():
            if qty <= 1e-9:
                continue
            mid = latest_mids.get(token, 0.5)
            unreal = (mid - avg_cost) * qty
            per_engine[engine]["open_unrealized_pnl_est"] += unreal

        for metric in per_engine.values():
            metric["closed_pnl"] = round(float(metric["closed_pnl"]), 6)
            metric["open_unrealized_pnl_est"] = round(float(metric["open_unrealized_pnl_est"]), 6)
            metric["total_pnl_est"] = round(metric["closed_pnl"] + metric["open_unrealized_pnl_est"], 6)
            metric["buy_notional"] = round(float(metric["buy_notional"]), 6)
            metric["sell_notional"] = round(float(metric["sell_notional"]), 6)
            metric["fees"] = round(float(metric["fees"]), 6)

        per_engine_rows = sorted(
            [{"engine": engine, **metrics} for engine, metrics in per_engine.items()],
            key=lambda x: x["total_pnl_est"],
            reverse=True,
        )

        all_samples: list[float] = []
        for items in realized_samples.values():
            all_samples.extend(items)
        proof = self._proof_gate(per_engine_rows, all_samples)

        latest_risk = self.conn.execute(
            "SELECT * FROM risk_state ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return {
            "window_hours": window_hours,
            "per_engine": per_engine_rows,
            "proof": proof,
            "latest_risk": dict(latest_risk) if latest_risk else {},
        }

    def _latest_token_mids(self) -> dict[str, float]:
        rows = self.conn.execute(
            """
            SELECT ts, primary_token_id, primary_mid, secondary_token_id, secondary_mid
            FROM market_snapshots
            ORDER BY ts DESC
            """
        ).fetchall()
        mids: dict[str, float] = {}
        for row in rows:
            p_token = str(row["primary_token_id"])
            s_token = str(row["secondary_token_id"])
            p_mid = float(row["primary_mid"] or 0.0)
            s_mid = float(row["secondary_mid"] or 0.0)
            if p_token not in mids and p_mid > 0:
                mids[p_token] = p_mid
            if s_token not in mids and s_mid > 0:
                mids[s_token] = s_mid
        return mids

    def _proof_gate(self, per_engine_rows: list[dict[str, Any]], realized_samples: list[float]) -> dict[str, Any]:
        total_fills = int(sum(int(row["fills"]) for row in per_engine_rows))
        total_closed = int(sum(int(row["closed_trades"]) for row in per_engine_rows))
        total_closed_pnl = float(sum(float(row["closed_pnl"]) for row in per_engine_rows))
        total_pnl_est = float(sum(float(row["total_pnl_est"]) for row in per_engine_rows))

        n = len(realized_samples)
        mean = 0.0
        std = 0.0
        lower95 = 0.0
        if n > 0:
            mean = sum(realized_samples) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in realized_samples) / (n - 1)
            std = math.sqrt(max(0.0, variance))
            lower95 = mean - 1.96 * (std / math.sqrt(n))
        else:
            lower95 = mean

        # Hard pass criteria for "proof of profitability", not just one lucky minute.
        min_fills = 80
        min_closed = 30
        passed = (
            total_fills >= min_fills
            and total_closed >= min_closed
            and total_closed_pnl > 0
            and total_pnl_est > 0
            and lower95 > 0
        )
        return {
            "passed": passed,
            "criteria": {
                "min_fills": min_fills,
                "min_closed_trades": min_closed,
                "closed_pnl_positive": True,
                "total_pnl_est_positive": True,
                "realized_trade_mean_ci95_lower_gt_zero": True,
            },
            "summary": {
                "fills": total_fills,
                "closed_trades": total_closed,
                "closed_pnl": round(total_closed_pnl, 6),
                "total_pnl_est": round(total_pnl_est, 6),
                "realized_trade_mean": round(mean, 6),
                "realized_trade_ci95_lower": round(lower95, 6),
            },
        }

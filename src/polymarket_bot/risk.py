from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from polymarket_bot.config import BotConfig
from polymarket_bot.models import (
    OrderIntent,
    OrderResult,
    PositionState,
    RiskState,
    Side,
)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class InventorySyncResult:
    cash_before: float
    cash_after: float
    token_size_changes: dict[str, tuple[float, float]]
    unknown_market_tokens: tuple[str, ...]


class RiskManager:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.day_anchor = datetime.now(tz=timezone.utc).date()
        self.day_start_equity = config.bankroll_usdc
        self.cash = config.bankroll_usdc
        self.positions: dict[str, PositionState] = {}
        self.market_exposure: dict[str, float] = {}
        self.consecutive_exec_errors = 0
        self.halted = False
        self.halted_reason = ""
        self.last_data_at = datetime.now(tz=timezone.utc)

    def _roll_day(self, now: datetime) -> None:
        if now.date() == self.day_anchor:
            return
        self.day_anchor = now.date()
        self.day_start_equity = self.current_equity()
        self.halted = False
        self.halted_reason = ""
        self.consecutive_exec_errors = 0

    def update_data_heartbeat(self, now: datetime) -> None:
        self.last_data_at = now
        self._roll_day(now)

    def current_equity(self) -> float:
        return self.cash + sum(
            pos.size * pos.mark_price for pos in self.positions.values()
        )

    def total_exposure(self) -> float:
        return sum(self.market_exposure.values())

    def daily_drawdown_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        dd = (self.day_start_equity - self.current_equity()) / self.day_start_equity
        return max(0.0, dd)

    def state(self) -> RiskState:
        return RiskState(
            current_equity=self.current_equity(),
            daily_drawdown_pct=self.daily_drawdown_pct(),
            total_exposure=self.total_exposure(),
            halted=self.halted,
            halted_reason=self.halted_reason,
            consecutive_exec_errors=self.consecutive_exec_errors,
        )

    def can_place(self, intent: OrderIntent) -> RiskDecision:
        if self.halted and intent.side == Side.BUY:
            return RiskDecision(False, f"risk halted: {self.halted_reason}")
        if intent.side == Side.BUY:
            required_cash = max(0.0, intent.notional)
            if required_cash > (self.cash + 1e-9):
                return RiskDecision(False, "cash budget exceeded")
        return RiskDecision(True, "")

    def apply_fill(self, result: OrderResult) -> None:
        if not result.is_filled:
            return
        position = self.positions.get(
            result.token_id,
            PositionState(token_id=result.token_id, market_id=result.market_id),
        )

        if result.side == Side.BUY:
            total_cost = (
                position.average_price * position.size
                + result.filled_price * result.filled_size
            )
            new_size = position.size + result.filled_size
            position.average_price = total_cost / new_size if new_size > 0 else 0.0
            position.size = new_size
            self.cash -= result.filled_price * result.filled_size
            self.cash -= result.fee_paid
        else:
            sell_size = min(result.filled_size, position.size)
            position.size -= sell_size
            self.cash += result.filled_price * sell_size
            self.cash -= result.fee_paid
            if position.size <= 1e-9:
                position.size = 0.0
                position.average_price = 0.0

        self.positions[result.token_id] = position
        self._recompute_market_exposure()

    def sync_exchange_inventory(
        self,
        *,
        cash: float,
        token_sizes: dict[str, float],
        token_market_ids: dict[str, str],
    ) -> InventorySyncResult:
        cash_before = self.cash
        self.cash = max(0.0, float(cash))

        size_changes: dict[str, tuple[float, float]] = {}
        unknown_market_tokens: list[str] = []
        for token_id, raw_size in token_sizes.items():
            cleaned_token_id = str(token_id or "").strip()
            if not cleaned_token_id:
                continue
            new_size = max(0.0, float(raw_size))
            existing = self.positions.get(cleaned_token_id)
            market_id = (token_market_ids.get(cleaned_token_id) or "").strip()
            if existing is None:
                if not market_id:
                    if new_size > 1e-9:
                        unknown_market_tokens.append(cleaned_token_id)
                    continue
                existing = PositionState(
                    token_id=cleaned_token_id,
                    market_id=market_id,
                )
            elif market_id:
                existing.market_id = market_id

            previous_size = max(0.0, existing.size)
            if abs(previous_size - new_size) > 1e-9:
                size_changes[cleaned_token_id] = (previous_size, new_size)

            if new_size <= 1e-9:
                existing.size = 0.0
                existing.average_price = 0.0
            else:
                # If inventory appeared outside local fill accounting, keep a
                # conservative non-zero average so pair-cost logic does not
                # assume free inventory.
                if previous_size <= 1e-9 and existing.average_price <= 0:
                    existing.average_price = max(0.5, existing.mark_price)
                existing.size = new_size

            self.positions[cleaned_token_id] = existing

        self._recompute_market_exposure()
        return InventorySyncResult(
            cash_before=cash_before,
            cash_after=self.cash,
            token_size_changes=size_changes,
            unknown_market_tokens=tuple(sorted(set(unknown_market_tokens))),
        )

    def apply_merge(
        self,
        *,
        market_id: str,
        primary_token_id: str,
        secondary_token_id: str,
        pair_size: float,
        payout_per_pair: float = 1.0,
    ) -> float:
        requested = max(0.0, float(pair_size))
        if requested <= 0:
            return 0.0

        primary = self.positions.get(
            primary_token_id,
            PositionState(token_id=primary_token_id, market_id=market_id),
        )
        secondary = self.positions.get(
            secondary_token_id,
            PositionState(token_id=secondary_token_id, market_id=market_id),
        )
        reducible = min(requested, max(0.0, primary.size), max(0.0, secondary.size))
        if reducible <= 0:
            return 0.0

        primary.size -= reducible
        secondary.size -= reducible
        if primary.size <= 1e-9:
            primary.size = 0.0
            primary.average_price = 0.0
        if secondary.size <= 1e-9:
            secondary.size = 0.0
            secondary.average_price = 0.0

        self.positions[primary_token_id] = primary
        self.positions[secondary_token_id] = secondary
        self.cash += reducible * max(0.0, float(payout_per_pair))
        self._recompute_market_exposure()
        return reducible

    def mark_to_market(self, mark_prices: dict[str, float]) -> None:
        for token_id, position in self.positions.items():
            if token_id in mark_prices:
                position.mark_price = mark_prices[token_id]
        self._recompute_market_exposure()

    def _token_notionals_by_market(self) -> dict[str, dict[str, float]]:
        by_market: dict[str, dict[str, float]] = {}
        for token_id, position in self.positions.items():
            if position.size <= 0:
                continue
            market = by_market.setdefault(position.market_id, {})
            market[token_id] = max(0.0, position.notional)
        return by_market

    def _projected_market_exposure(self, intent: OrderIntent) -> float:
        by_market = self._token_notionals_by_market()
        token_notionals = dict(by_market.get(intent.market_id, {}))
        token_notionals[intent.token_id] = token_notionals.get(
            intent.token_id, 0.0
        ) + max(0.0, intent.notional)
        if not token_notionals:
            return 0.0
        # Two-sided binary inventory is partially hedged; cap by dominant leg not sum of both legs.
        return max(token_notionals.values())

    def _recompute_market_exposure(self) -> None:
        by_market = self._token_notionals_by_market()
        exposure: dict[str, float] = {}
        for market_id, token_notionals in by_market.items():
            if not token_notionals:
                continue
            exposure[market_id] = max(token_notionals.values())
        self.market_exposure = exposure

    def on_execution_error(self) -> None:
        self.consecutive_exec_errors += 1
        if self.consecutive_exec_errors >= self.config.max_consecutive_exec_errors:
            self.halt("consecutive execution errors")

    def on_execution_success(self) -> None:
        self.consecutive_exec_errors = 0

    def _maybe_resume_from_drawdown(self) -> None:
        if not self.halted:
            return
        if "daily drawdown threshold exceeded" not in self.halted_reason.lower():
            return
        # Hysteresis prevents thrashing if equity hovers around the threshold.
        resume_threshold = self.config.max_daily_dd_pct * 0.70
        if self.daily_drawdown_pct() <= resume_threshold:
            self.halted = False
            self.halted_reason = ""
            self.consecutive_exec_errors = 0

    def check_kill_switches(self, now: datetime) -> None:
        if self.halted:
            return

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halted_reason = reason

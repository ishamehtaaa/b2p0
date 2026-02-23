from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime, timezone

from polymarket_bot.clients_clob import ClobClient
from polymarket_bot.models import FeeInfo
from polymarket_bot.pricing import clamp


class SpotTracker:
    def __init__(self) -> None:
        self.points: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def update(self, now_ts: float, price: float) -> None:
        with self._lock:
            self.points.append((now_ts, price))
            while self.points and now_ts - self.points[0][0] > 900:
                self.points.popleft()

    def _snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self.points)

    def point_count(self) -> int:
        with self._lock:
            return len(self.points)

    def latest_age_seconds(self, now_ts: float | None = None) -> float:
        points = self._snapshot()
        if not points:
            return float("inf")
        current_ts = now_ts if now_ts is not None else time.time()
        return max(0.0, current_ts - points[-1][0])

    def price_at_or_before(
        self, target_ts: float, max_lookback_seconds: float = 600.0
    ) -> float | None:
        points = self._snapshot()
        if not points:
            return None
        if target_ts < points[0][0]:
            return None

        candidate: float | None = None
        candidate_ts = 0.0
        for ts, price in points:
            if ts <= target_ts:
                candidate = price
                candidate_ts = ts
            else:
                break
        if candidate is None:
            return None
        if target_ts - candidate_ts > max_lookback_seconds:
            return None
        return candidate

    def return_over_seconds(self, seconds: int) -> float:
        points = self._snapshot()
        if len(points) < 2:
            return 0.0
        latest_ts, latest_price = points[-1]
        for ts, price in reversed(points):
            if latest_ts - ts >= seconds:
                if price <= 0:
                    return 0.0
                return (latest_price - price) / price
        first_price = points[0][1]
        if first_price <= 0:
            return 0.0
        return (latest_price - first_price) / first_price

    def realized_volatility(self, seconds: int = 300) -> float:
        points = self._snapshot()
        if len(points) < 3:
            return 0.0
        latest_ts = points[-1][0]
        window = [
            (ts, price)
            for ts, price in points
            if latest_ts - ts <= seconds and price > 0
        ]
        if len(window) < 3:
            return 0.0
        returns: list[float] = []
        for i in range(1, len(window)):
            prev = window[i - 1][1]
            curr = window[i][1]
            if prev <= 0 or curr <= 0:
                continue
            returns.append(math.log(curr / prev))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(max(0.0, variance))

    def _rnjd_params(self, horizon_seconds: int) -> tuple[float, float] | None:
        points = self._snapshot()
        if len(points) < 20:
            return None

        latest_ts = points[-1][0]
        window = [(ts, px) for ts, px in points if latest_ts - ts <= 300 and px > 0]
        if len(window) < 20:
            return None

        returns: list[float] = []
        dts: list[float] = []
        for i in range(1, len(window)):
            dt = max(1e-3, window[i][0] - window[i - 1][0])
            prev = window[i - 1][1]
            curr = window[i][1]
            if prev <= 0 or curr <= 0:
                continue
            returns.append(math.log(curr / prev))
            dts.append(dt)
        if len(returns) < 10:
            return None

        scaled = [r / math.sqrt(dt) for r, dt in zip(returns, dts)]
        if not scaled:
            return None
        median_scaled = sorted(scaled)[len(scaled) // 2]
        abs_dev = [abs(x - median_scaled) for x in scaled]
        mad = sorted(abs_dev)[len(abs_dev) // 2]
        sigma_ps = max(1e-6, 1.4826 * mad)

        jump_mask: list[bool] = []
        for r, dt in zip(returns, dts):
            jump_mask.append(abs(r) > 3.0 * sigma_ps * math.sqrt(dt))

        total_time = max(1.0, sum(dts))
        jump_returns = [r for r, is_jump in zip(returns, jump_mask) if is_jump]
        cont_returns = [r for r, is_jump in zip(returns, jump_mask) if not is_jump]
        cont_dts = [dt for dt, is_jump in zip(dts, jump_mask) if not is_jump]

        lam = len(jump_returns) / total_time
        if cont_returns and cont_dts:
            mu_c = sum(cont_returns) / max(1e-6, sum(cont_dts))
            scaled_cont = [r / math.sqrt(dt) for r, dt in zip(cont_returns, cont_dts)]
            if len(scaled_cont) >= 2:
                mean_cont = sum(scaled_cont) / len(scaled_cont)
                var_c = sum((x - mean_cont) ** 2 for x in scaled_cont) / (
                    len(scaled_cont) - 1
                )
            else:
                var_c = sigma_ps**2
        else:
            mu_c = 0.0
            var_c = sigma_ps**2

        if jump_returns:
            mu_j = sum(jump_returns) / len(jump_returns)
            if len(jump_returns) >= 2:
                mean_j = mu_j
                var_j = sum((x - mean_j) ** 2 for x in jump_returns) / (
                    len(jump_returns) - 1
                )
            else:
                var_j = (abs(mu_j) * 0.5) ** 2
        else:
            mu_j = 0.0
            var_j = 0.0

        h = float(max(1, horizon_seconds))
        mu_h = (mu_c * h) + (lam * h * mu_j)
        var_h = max(1e-10, (var_c * h) + (lam * h * (var_j + mu_j * mu_j)))

        # Short-horizon momentum component to react faster in 5m / 15m windows.
        ret_8 = self.return_over_seconds(min(8, horizon_seconds))
        ret_20 = self.return_over_seconds(min(20, horizon_seconds))
        mu_h += 0.35 * ret_8 + 0.15 * ret_20
        return mu_h, var_h

    def rnjd_probability_above_log_return(
        self, horizon_seconds: int, target_log_return: float = 0.0
    ) -> float:
        params = self._rnjd_params(horizon_seconds)
        if params is None:
            return 0.5
        mu_h, var_h = params
        z = (mu_h - target_log_return) / math.sqrt(var_h)
        return clamp(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.01, 0.99)

    def rnjd_probability(self, horizon_seconds: int) -> float:
        return self.rnjd_probability_above_log_return(horizon_seconds, 0.0)


class BookMotionTracker:
    def __init__(self, max_window_seconds: float = 180.0) -> None:
        self.max_window_seconds = max(30.0, float(max_window_seconds))
        self._points: dict[str, deque[tuple[float, float, float, float]]] = {}
        self._lock = threading.Lock()

    def update_book(
        self,
        *,
        now_ts: float,
        token_id: str,
        best_bid: float,
        best_ask: float,
        mid: float,
    ) -> None:
        if not token_id:
            return
        bid = max(0.0, float(best_bid))
        ask = max(0.0, float(best_ask))
        mark = max(0.0, float(mid))
        with self._lock:
            bucket = self._points.setdefault(token_id, deque())
            bucket.append((float(now_ts), bid, ask, mark))
            cutoff = float(now_ts) - self.max_window_seconds
            while bucket and bucket[0][0] < cutoff:
                bucket.popleft()

    def summarize(
        self, *, token_id: str, now_ts: float, window_seconds: float
    ) -> dict[str, float]:
        if not token_id:
            return {}
        window = max(5.0, min(self.max_window_seconds, float(window_seconds)))
        cutoff = float(now_ts) - window
        with self._lock:
            raw = list(self._points.get(token_id, ()))
        if not raw:
            return {}
        points = [item for item in raw if item[0] >= cutoff]
        if not points:
            return {}

        asks = [ask for _, _, ask, _ in points if ask > 0]
        bids = [bid for _, bid, _, _ in points if bid > 0]
        mids = [mark for _, _, _, mark in points if mark > 0]
        summary: dict[str, float] = {"samples": float(len(points))}
        if asks:
            summary["ask_low"] = min(asks)
            summary["ask_high"] = max(asks)
            summary["ask_swing"] = max(0.0, summary["ask_high"] - summary["ask_low"])
        if bids:
            summary["bid_low"] = min(bids)
            summary["bid_high"] = max(bids)
            summary["bid_swing"] = max(0.0, summary["bid_high"] - summary["bid_low"])
        if len(mids) >= 2:
            summary["mid_drift"] = mids[-1] - mids[0]
        else:
            summary["mid_drift"] = 0.0

        # Detect rapid two-sided tape by combining short-window swings with
        # directional flip-rate in asks/mids.
        short_window = max(5.0, min(window, window * 0.35))
        short_cutoff = float(now_ts) - short_window
        short_points = [item for item in points if item[0] >= short_cutoff]
        short_asks = [ask for _, _, ask, _ in short_points if ask > 0]
        short_mids = [mark for _, _, _, mark in short_points if mark > 0]
        if short_asks:
            summary["ask_swing_short"] = max(0.0, max(short_asks) - min(short_asks))
        else:
            summary["ask_swing_short"] = 0.0
        if short_mids:
            summary["mid_swing_short"] = max(0.0, max(short_mids) - min(short_mids))
        else:
            summary["mid_swing_short"] = 0.0

        ask_flip_count, ask_flip_rate = self._flip_rate(points=points, field_index=2)
        mid_flip_count, mid_flip_rate = self._flip_rate(points=points, field_index=3)
        summary["ask_flip_count"] = float(ask_flip_count)
        summary["ask_flip_rate"] = ask_flip_rate
        summary["mid_flip_count"] = float(mid_flip_count)
        summary["mid_flip_rate"] = mid_flip_rate
        return summary

    @staticmethod
    def _flip_rate(
        points: list[tuple[float, float, float, float]], field_index: int
    ) -> tuple[int, float]:
        if len(points) < 3:
            return 0, 0.0
        prev_sign = 0
        flip_count = 0
        first_ts = float(points[0][0])
        last_ts = float(points[-1][0])
        for idx in range(1, len(points)):
            prev = float(points[idx - 1][field_index])
            curr = float(points[idx][field_index])
            if prev <= 0 or curr <= 0:
                continue
            delta = curr - prev
            if abs(delta) < 0.004:
                continue
            sign = 1 if delta > 0 else -1
            if prev_sign != 0 and sign != prev_sign:
                flip_count += 1
            prev_sign = sign
        duration = max(1.0, last_ts - first_ts)
        return flip_count, float(flip_count) / duration


class FeeCache:
    def __init__(self, refresh_seconds: float) -> None:
        self.refresh_seconds = refresh_seconds
        self.cache: dict[str, FeeInfo] = {}
        self._lock = threading.Lock()

    def get(self, token_id: str, clob: ClobClient) -> FeeInfo:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            current = self.cache.get(token_id)
        if current is not None:
            age = (now - current.fetched_at).total_seconds()
            if age <= self.refresh_seconds:
                return current
        fresh = clob.get_fee_rate(token_id, side="buy")
        with self._lock:
            self.cache[token_id] = fresh
        return fresh

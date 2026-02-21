from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_tick(price: float, tick: float = 0.01) -> float:
    ticks = round(price / tick)
    return max(tick, min(1.0 - tick, ticks * tick))


def per_share_fee(price: float, base_fee_bps: int) -> float:
    """
    Conservative fee estimate using p*(1-p) scaling:
      fee ~= (base_fee_bps / 10000) * p * (1-p)
    """
    p = clamp(price, 0.0, 1.0)
    return (base_fee_bps / 10000.0) * p * (1.0 - p)


def directional_ev_per_share(
    fair_probability: float,
    entry_price: float,
    base_fee_bps: int,
    slippage_buffer: float,
) -> float:
    fee = per_share_fee(entry_price, base_fee_bps)
    return fair_probability - entry_price - fee - slippage_buffer


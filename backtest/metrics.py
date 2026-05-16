"""Backtest metrics — adverse_selection_bps, fill_rate, markout, P&L.

Per ADR 0008 framing (Briola 2024-style modest claims):
    - Track markouts: for each policy fill at time t, mid_K_events_later - fill_price
    - Sign by side: bid fill -> profit if mid goes up; ask fill -> profit if mid down
    - adverse_markout = -mean(signed markout); positive = adverse to us
    - No Sharpe (we don't model inventory/capital/time)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    """Aggregates a single (ticker, policy) backtest run."""
    ticker: str
    policy_name: str
    quotes_posted: int
    fills: list  # list of PolicyFill (from backtest.policies)
    mids: np.ndarray  # per-event mid prices (precomputed)


def _signed_markout(fill, mid_future: int, tick_size: int = 100) -> float:
    """Signed mid markout in ticks. Positive = favorable to the policy.

    For a BID fill (side_sign=+1): we bought at fill_price; if mid moves UP, profit.
    For an ASK fill (side_sign=-1): we sold; if mid moves DOWN, profit.
    """
    diff_native = mid_future - fill.fill_price  # in price-int units
    return (diff_native * fill.side_sign) / tick_size  # in ticks


def summarise(result: BacktestResult, markout_k: int = 100,
              tick_size: int = 100) -> dict:
    """Compute the canonical metric dict for results.md per ADR 0008."""
    n_fills = len(result.fills)
    mid_avg = float(np.mean(result.mids)) if len(result.mids) > 0 else 1.0

    markouts: list[float] = []  # in ticks; positive = favorable
    fill_qty_total = 0
    for f in result.fills:
        # mid_K_events later — clip at end of stream
        future_idx = min(f.event_index + markout_k, len(result.mids) - 1) \
            if f.event_index >= 0 else 0
        if future_idx <= 0:
            continue  # immediate fill or beyond-stream; skip
        mid_future = int(result.mids[future_idx])
        m = _signed_markout(f, mid_future, tick_size)
        markouts.append(m)
        fill_qty_total += f.quantity

    mean_markout = float(np.mean(markouts)) if markouts else 0.0
    median_markout = float(np.median(markouts)) if markouts else 0.0

    # Adverse selection bps: a NEGATIVE signed markout is adverse to us. Convert
    # the mean adverse loss in ticks to bps of mid price.
    # adverse_markout_bps = (-mean_markout * tick_size) / mid_avg * 1e4
    # Positive value means we LOST X bps on average per fill.
    adverse_bps = float(-mean_markout * tick_size / mid_avg * 10_000)

    return {
        "ticker": result.ticker,
        "policy": result.policy_name,
        "quotes_posted": result.quotes_posted,
        "fills": n_fills,
        "fill_qty": fill_qty_total,
        "fill_rate": (n_fills / max(result.quotes_posted, 1)),
        "markout_mean_ticks": mean_markout,
        "markout_median_ticks": median_markout,
        "adverse_selection_bps": adverse_bps,
        "gross_pnl_ticks": float(np.sum(markouts)) if markouts else 0.0,
        "trade_count": n_fills,
    }

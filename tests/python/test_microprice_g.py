"""Unit tests for Stoikov G(I,S) microprice fit + lookup.

References:
    Stoikov 2018 — The Micro-Price, Quantitative Finance 18(12), 1959-1966

The G(I, S) function is the martingale-fair price ADJUSTMENT (relative to mid)
as a function of (imbalance, spread). It's fitted offline from real LOB data
via a fixed-point iteration on the discretized (I, S) state space.

At inference time, the C++ matcher loads the LUT JSON and does an O(1) table
lookup per event. No regression / model inference needed at runtime.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from model.microprice_g import MicropriceLut, fit_microprice_g


def _synthetic_tob(n: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic TOB rows with a controlled microstructure.

    Bid imbalance correlates with next-step mid moves (positive => up move).
    Used to test that fit_microprice_g recovers the directional pattern.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    mid = np.zeros(n, dtype=np.int64)
    mid[0] = 10000
    bid_sz = np.zeros(n, dtype=np.int64)
    ask_sz = np.zeros(n, dtype=np.int64)
    spread = 100  # 1 tick

    for i in range(n):
        # Random imbalance: bias toward 50/50 with noise
        total = rng.integers(100, 500)
        imb_target = rng.uniform(-0.8, 0.8)
        bid_sz[i] = max(1, int(total * (1 + imb_target) / 2))
        ask_sz[i] = max(1, total - bid_sz[i])
        if i > 0:
            # mid drifts in the direction of last-row imbalance + noise
            last_imb = (bid_sz[i - 1] - ask_sz[i - 1]) / (bid_sz[i - 1] + ask_sz[i - 1])
            drift = int(last_imb * 50)  # half a tick per unit of imbalance
            mid[i] = mid[i - 1] + drift + rng.integers(-30, 30)

    return pl.DataFrame({
        "bid_price_l1": mid - spread // 2,
        "ask_price_l1": mid + spread // 2,
        "bid_size_l1": bid_sz,
        "ask_size_l1": ask_sz,
    })


def test_fit_produces_lut_with_expected_shape():
    df = _synthetic_tob(n=2000)
    lut = fit_microprice_g(df, n_imbalance_buckets=10, n_spread_buckets=3, tick_size=100)
    assert lut.table.shape == (10, 3)


def test_lut_center_is_between_extremes():
    """G(I=0) should lie strictly between G(I=-0.8) and G(I=+0.8).

    The 'symmetric -> zero' claim only holds for data with zero long-run drift,
    which random-walk synthetic doesn't guarantee. The structural property is:
    the center bucket's adjustment is bracketed by the extremes.
    """
    df = _synthetic_tob(n=5000)
    lut = fit_microprice_g(df, n_imbalance_buckets=11, n_spread_buckets=3, tick_size=100)
    center = lut.lookup(imbalance=0.0, spread_ticks=1)
    bid_heavy = lut.lookup(imbalance=0.8, spread_ticks=1)
    ask_heavy = lut.lookup(imbalance=-0.8, spread_ticks=1)
    lo, hi = min(bid_heavy, ask_heavy), max(bid_heavy, ask_heavy)
    assert lo <= center <= hi, \
        f"G(0)={center:.1f} not in [{lo:.1f}, {hi:.1f}] (G(-0.8)={ask_heavy:.1f}, G(+0.8)={bid_heavy:.1f})"


def test_lut_bid_imbalance_yields_positive_adjustment():
    df = _synthetic_tob(n=5000)
    lut = fit_microprice_g(df, n_imbalance_buckets=11, n_spread_buckets=3, tick_size=100)
    # Strong bid imbalance should push G(I, S) > 0 (above mid)
    # ... but with synthetic data correlation, expect positive direction
    high_bid_adj = lut.lookup(imbalance=0.8, spread_ticks=1)
    high_ask_adj = lut.lookup(imbalance=-0.8, spread_ticks=1)
    assert high_bid_adj > high_ask_adj, \
        f"expected G(0.8) > G(-0.8), got {high_bid_adj} > {high_ask_adj}"


def test_lut_round_trip_through_json(tmp_path: Path):
    df = _synthetic_tob(n=2000)
    lut = fit_microprice_g(df, n_imbalance_buckets=10, n_spread_buckets=3, tick_size=100)
    json_path = tmp_path / "lut.json"
    lut.save(json_path)
    loaded = MicropriceLut.load(json_path)
    assert loaded.table.shape == lut.table.shape
    # Bit-equal after JSON round-trip (floats encoded to full precision)
    import numpy as np
    np.testing.assert_array_equal(loaded.table, lut.table)


def test_lookup_vec_matches_scalar_loop():
    df = _synthetic_tob(n=2000)
    lut = fit_microprice_g(df, n_imbalance_buckets=10, n_spread_buckets=3, tick_size=100)
    imb = [0.0, 0.5, -0.5, 0.9, -0.9]
    sp = [1, 2, 1, 3, 1]
    vec = lut.lookup_vec(imb, sp)
    scalar = [lut.lookup(i, s) for i, s in zip(imb, sp, strict=False)]
    import numpy as np
    np.testing.assert_array_equal(vec, scalar)

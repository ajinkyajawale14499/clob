"""Unit tests for extended microstructure features (synthetic data, no data-mark)."""

import polars as pl

from data.features.ofi import multi_level_ofi
from data.features.queue import queue_depletion_ewma
from data.features.spread import spread_ticks, spread_zscore
from data.features.trades import signed_trade_flow, trade_flow_imbalance
from data.features.vol import realized_vol


def _trades(rows: list[tuple[int, int]]) -> pl.DataFrame:
    """Helper: build a Trades frame from [(aggressor_side, size), ...]."""
    return pl.DataFrame({
        "ts_ns": pl.Series(range(len(rows)), dtype=pl.Datetime("ns")),
        "order_id": list(range(1, len(rows) + 1)),
        "size": [s for _, s in rows],
        "price": [10000 + i for i in range(len(rows))],
        "aggressor_side": pl.Series([a for a, _ in rows], dtype=pl.Int8),
    })


def test_signed_trade_flow_all_buys_positive():
    df = _trades([(1, 10), (1, 20), (1, 30)])
    out = signed_trade_flow(df, window=3)
    assert out["signed_trade_flow_3"][-1] == 60


def test_signed_trade_flow_all_sells_negative():
    df = _trades([(-1, 10), (-1, 20), (-1, 30)])
    out = signed_trade_flow(df, window=3)
    assert out["signed_trade_flow_3"][-1] == -60


def test_signed_trade_flow_mixed_cancels():
    df = _trades([(1, 50), (-1, 30), (1, 20)])
    out = signed_trade_flow(df, window=3)
    # 50 - 30 + 20 = 40
    assert out["signed_trade_flow_3"][-1] == 40


def test_tfi_all_buys_one():
    df = _trades([(1, 10), (1, 20), (1, 30)])
    out = trade_flow_imbalance(df, window=3)
    assert out["tfi_3"][-1] == 1.0


def test_tfi_all_sells_minus_one():
    df = _trades([(-1, 10), (-1, 20), (-1, 30)])
    out = trade_flow_imbalance(df, window=3)
    assert out["tfi_3"][-1] == -1.0


def test_tfi_balanced_zero():
    df = _trades([(1, 50), (-1, 50)])
    out = trade_flow_imbalance(df, window=2)
    assert out["tfi_2"][-1] == 0.0


def test_tfi_in_bounds_for_any_window():
    df = _trades([(1, 100), (-1, 25), (1, 200), (-1, 75), (1, 30)])
    out = trade_flow_imbalance(df, window=3)
    last3 = out["tfi_3"].drop_nulls()
    assert (last3 >= -1.0).all() and (last3 <= 1.0).all()


def test_signed_trade_flow_rolling_window_size_2():
    df = _trades([(1, 10), (1, 20), (-1, 30), (1, 40)])
    out = signed_trade_flow(df, window=2)
    # cumulative pairs: [null, 30, -10, 10]
    vals = out["signed_trade_flow_2"].to_list()
    assert vals == [None, 30, -10, 10]


# ------- realized_vol -------

def _tob_rows(mids: list[int], tick: int = 100) -> pl.DataFrame:
    """Build a minimal TOB frame with controlled mid prices (bid_l1 + ask_l1)/2."""
    return pl.DataFrame({
        "bid_price_l1": [m - tick // 2 for m in mids],
        "ask_price_l1": [m + tick // 2 for m in mids],
        "bid_size_l1": [100] * len(mids),
        "ask_size_l1": [100] * len(mids),
    })


def test_realized_vol_zero_when_mid_constant():
    df = _tob_rows([10000] * 10)
    out = realized_vol(df, window=5)
    assert out["realized_vol_5"][-1] == 0.0


def test_realized_vol_positive_when_mid_moves():
    df = _tob_rows([10000, 10100, 10200, 10300, 10400])  # +100 each step
    out = realized_vol(df, window=4)
    # ret^2 each step = 100^2 = 10000. Sum over last 4 = 40000. sqrt = 200.
    assert out["realized_vol_4"][-1] == 200.0


# ------- spread_ticks + spread_zscore -------

def test_spread_ticks_basic():
    df = _tob_rows([10000], tick=100)
    out = spread_ticks(df, tick_size=100)
    # bid=9950 ask=10050 -> spread = 100 = 1 tick
    assert out["spread_ticks"][0] == 1.0


def test_spread_zscore_zero_when_spread_constant():
    df = _tob_rows([10000] * 10)
    out = spread_zscore(df, window=5)
    # constant spread -> std=0 -> z-score = NaN/Inf-like; polars yields null
    # We accept null OR 0; assert it's not finite-positive
    last = out["spread_zscore_5"][-1]
    assert last is None or last == 0.0 or last != last  # NaN check


# ------- queue_depletion_ewma -------

def test_queue_depletion_positive_when_bid_grows():
    df = pl.DataFrame({
        "bid_price_l1": [10000] * 5, "ask_price_l1": [10100] * 5,
        "bid_size_l1": [100, 110, 120, 130, 140],
        "ask_size_l1": [100, 100, 100, 100, 100],
    })
    out = queue_depletion_ewma(df, alpha=0.5)
    assert out["queue_depletion_bid"][-1] > 0


def test_queue_depletion_negative_when_ask_shrinks():
    df = pl.DataFrame({
        "bid_price_l1": [10000] * 5, "ask_price_l1": [10100] * 5,
        "bid_size_l1": [100] * 5,
        "ask_size_l1": [100, 80, 60, 40, 20],
    })
    out = queue_depletion_ewma(df, alpha=0.5)
    assert out["queue_depletion_ask"][-1] < 0


# ------- multi_level_ofi -------

def test_multi_level_ofi_zero_on_constant_book():
    df = pl.DataFrame({
        f"{side}_size_l{lv}": [100] * 10
        for side in ("bid", "ask") for lv in range(1, 11)
    })
    out = multi_level_ofi(df, levels=[2, 3, 4, 5], window=5)
    assert out["mlofi_l2_l5_w5"][-1] == 0


def test_multi_level_ofi_positive_when_bid_l2_l5_grows():
    n = 10
    df = pl.DataFrame({
        **{f"bid_size_l{lv}": [100 + i * 10 for i in range(n)] for lv in [2, 3, 4, 5]},
        **{f"ask_size_l{lv}": [100] * n for lv in [2, 3, 4, 5]},
        **{f"bid_size_l{lv}": [100] * n for lv in [1, 6, 7, 8, 9, 10]},
        **{f"ask_size_l{lv}": [100] * n for lv in [1, 6, 7, 8, 9, 10]},
    })
    out = multi_level_ofi(df, levels=[2, 3, 4, 5], window=5)
    assert out["mlofi_l2_l5_w5"][-1] > 0

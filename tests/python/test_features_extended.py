"""Unit tests for extended microstructure features (synthetic data, no data-mark)."""

import polars as pl

from data.features.trades import signed_trade_flow, trade_flow_imbalance


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

"""Unit tests for label generator (synthetic TOB, no data-mark)."""

import polars as pl

from model.labels import class_share, make_labels_3class


def _tob_from_mids(mids: list[int], spread: int = 100) -> pl.DataFrame:
    """Build a minimal TOB with controlled mid prices."""
    return pl.DataFrame({
        "bid_price_l1": [m - spread // 2 for m in mids],
        "ask_price_l1": [m + spread // 2 for m in mids],
        "bid_size_l1": [100] * len(mids),
        "ask_size_l1": [100] * len(mids),
    })


def test_up_when_mid_rises_one_tick():
    # mid: 10000 -> 10100 (one tick = 100)
    df = _tob_from_mids([10000, 10100])
    out = make_labels_3class(df, k_events=1, deadband_ticks=1)
    assert out.height == 1
    assert out["label"][0] == 2  # Up


def test_down_when_mid_falls_one_tick():
    df = _tob_from_mids([10100, 10000])
    out = make_labels_3class(df, k_events=1, deadband_ticks=1)
    assert out["label"][0] == 0  # Down


def test_stable_when_mid_unchanged():
    df = _tob_from_mids([10000, 10000])
    out = make_labels_3class(df, k_events=1, deadband_ticks=1)
    assert out["label"][0] == 1  # Stable


def test_stable_with_deadband_2_when_one_tick_move():
    """deadband=2 means moves < 2 ticks are Stable. 1-tick move -> Stable."""
    df = _tob_from_mids([10000, 10100])
    out = make_labels_3class(df, k_events=1, deadband_ticks=2)
    assert out["label"][0] == 1  # Stable (1-tick < 2-tick deadband)


def test_k_events_2_lookahead():
    df = _tob_from_mids([10000, 10000, 10200])
    out = make_labels_3class(df, k_events=2, deadband_ticks=1)
    # mid[0]=10000, mid[2]=10200 -> +2 ticks -> Up
    assert out.height == 1
    assert out["label"][0] == 2


def test_trailing_k_rows_dropped():
    df = _tob_from_mids([10000] * 10)
    out = make_labels_3class(df, k_events=3, deadband_ticks=1)
    assert out.height == 7  # 10 - 3


def test_class_share_returns_fractions():
    labels = pl.Series("label", [0, 0, 1, 1, 1, 2], dtype=pl.Int8)
    shares = class_share(labels)
    assert shares[0] == 2 / 6
    assert shares[1] == 3 / 6
    assert shares[2] == 1 / 6


def test_class_share_handles_missing_classes():
    labels = pl.Series("label", [1, 1, 1, 1], dtype=pl.Int8)
    shares = class_share(labels)
    assert shares == {0: 0.0, 1: 1.0, 2: 0.0}


def test_k_events_zero_raises():
    df = _tob_from_mids([10000, 10000])
    try:
        make_labels_3class(df, k_events=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for k_events=0")

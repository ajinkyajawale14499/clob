import polars as pl

from data.features.imbalance import imbalance
from data.features.microprice import microprice
from data.features.ofi import order_flow_imbalance


def test_microprice_balanced_book() -> None:
    df = pl.DataFrame({
        "bid_price_l1": [100], "bid_size_l1": [10],
        "ask_price_l1": [102], "ask_size_l1": [10],
    })
    out = microprice(df)
    assert out["microprice"][0] == 101.0


def test_microprice_thicker_bid_tilts_toward_ask() -> None:
    df = pl.DataFrame({
        "bid_price_l1": [100], "bid_size_l1": [90],
        "ask_price_l1": [102], "ask_size_l1": [10],
    })
    out = microprice(df)
    # (90*102 + 10*100) / 100 = 101.8
    assert abs(out["microprice"][0] - 101.8) < 1e-9


def test_ofi_zero_when_no_change() -> None:
    df = pl.DataFrame({
        "ts_ns": pl.Series(
            [1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=pl.Datetime("ns")
        ),
        "bid_size_l1": [10, 10, 10],
        "ask_size_l1": [10, 10, 10],
    })
    out = order_flow_imbalance(df.sort("ts_ns"), window="5s")
    assert out["ofi"].to_list() == [0, 0, 0]


def test_ofi_positive_when_bid_size_grows() -> None:
    df = pl.DataFrame({
        "ts_ns": pl.Series(
            [1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=pl.Datetime("ns")
        ),
        "bid_size_l1": [10, 20, 20],
        "ask_size_l1": [10, 10, 10],
    })
    out = order_flow_imbalance(df.sort("ts_ns"), window="5s")
    assert out["ofi"][-1] == 10


def test_imbalance_symmetric() -> None:
    df = pl.DataFrame({"bid_size_l1": [10], "ask_size_l1": [10]})
    out = imbalance(df)
    assert out["imbalance"][0] == 0.0


def test_imbalance_one_sided_bid() -> None:
    df = pl.DataFrame({"bid_size_l1": [100], "ask_size_l1": [0]})
    out = imbalance(df)
    assert out["imbalance"][0] == 1.0


def test_imbalance_one_sided_ask() -> None:
    df = pl.DataFrame({"bid_size_l1": [0], "ask_size_l1": [100]})
    out = imbalance(df)
    assert out["imbalance"][0] == -1.0


def test_imbalance_both_empty_is_null() -> None:
    df = pl.DataFrame({"bid_size_l1": [0], "ask_size_l1": [0]})
    out = imbalance(df)
    # Both sides empty -> undefined -> NULL (NOT 0)
    assert out["imbalance"][0] is None


def test_imbalance_integer_division_promotes_to_float() -> None:
    df = pl.DataFrame({"bid_size_l1": [3], "ask_size_l1": [5]})
    out = imbalance(df)
    assert abs(out["imbalance"][0] - (-0.25)) < 1e-9

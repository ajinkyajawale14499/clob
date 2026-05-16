import polars as pl

from data.features.microprice import microprice


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

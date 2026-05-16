from pathlib import Path

import polars as pl
import pytest

from data.ingestion.binance_bookticker import load_binance_bookticker

pytestmark = pytest.mark.data  # CI: skipped via `pytest -m "not data"`

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"


def _find_file() -> Path:
    files = list(SAMPLE_DIR.glob("*-bookTicker-*.csv"))
    assert files, "Binance bookTicker file missing in data/raw/"
    return files[0]


def test_binance_bookticker_returns_tob_frame() -> None:
    df = load_binance_bookticker(_find_file())
    assert df.columns == [
        "ts_ns",
        "bid_price_l1",
        "bid_size_l1",
        "ask_price_l1",
        "ask_size_l1",
    ]
    assert df.height > 1000  # busy crypto pair -> many updates per day


def test_binance_ts_is_datetime() -> None:
    df = load_binance_bookticker(_find_file())
    assert df["ts_ns"].dtype == pl.Datetime("ns")


def test_binance_prices_are_int64() -> None:
    df = load_binance_bookticker(_find_file())
    assert df["bid_price_l1"].dtype == pl.Int64
    assert df["ask_price_l1"].dtype == pl.Int64
    # BTC at ~$50k = 5e12 in 1e-8 ticks. Sanity: well above 1e10.
    assert df["bid_price_l1"].mean() > 1e10


def test_binance_bid_never_crosses_ask() -> None:
    df = load_binance_bookticker(_find_file())
    crossed = df.filter(pl.col("bid_price_l1") >= pl.col("ask_price_l1"))
    assert crossed.is_empty()

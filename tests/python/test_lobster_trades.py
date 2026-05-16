from pathlib import Path

import polars as pl
import pytest

from data.ingestion.lobster_message import load_lobster_messages
from data.ingestion.lobster_trades import extract_trades

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"
ALL_TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def _present_tickers() -> list[str]:
    return [t for t in ALL_TICKERS if list(SAMPLE_DIR.glob(f"{t}_*_message_*.csv"))]


if not _present_tickers():
    pytest.skip(
        "No LOBSTER files in data/raw/ — see README quickstart.",
        allow_module_level=True,
    )


@pytest.fixture(params=_present_tickers())
def ticker(request) -> str:
    return request.param


def test_trades_have_aggressor_sign(ticker: str) -> None:
    msg = load_lobster_messages(next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
    trades = extract_trades(msg)
    assert trades.columns == ["ts_ns", "order_id", "size", "price", "aggressor_side"]
    assert set(trades["aggressor_side"].unique().to_list()).issubset({-1, 1})
    # LOBSTER 2012-06-21 day sample: every stock has >1k visible+hidden trades.
    assert trades.height > 1000, f"{ticker}: only {trades.height} trades"


def test_trades_are_subset_of_messages(ticker: str) -> None:
    msg = load_lobster_messages(next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
    trades = extract_trades(msg)
    expected = msg.filter(pl.col("event_type").is_in([4, 5])).height
    assert trades.height == expected


def test_trades_preserve_chronological_order(ticker: str) -> None:
    msg = load_lobster_messages(next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
    trades = extract_trades(msg)
    assert trades["ts_ns"].is_sorted()


def test_trades_aggressor_opposite_of_resting_direction(ticker: str) -> None:
    """LOBSTER convention: on exec rows, `direction` is the SIGN OF THE RESTING ORDER
    (maker), so aggressor side is `-direction`. Verify the flip happened."""
    msg = load_lobster_messages(next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
    raw_execs = msg.filter(pl.col("event_type").is_in([4, 5]))
    trades = extract_trades(msg)
    # aggressor_side should equal -side from raw_execs, row by row.
    paired = raw_execs.select(pl.col("side").alias("resting_side")).hstack(
        trades.select(pl.col("aggressor_side"))
    )
    assert (paired["aggressor_side"] == -paired["resting_side"]).all()

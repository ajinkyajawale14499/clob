from pathlib import Path

import polars as pl
import pytest

from data.ingestion.lobster_message import load_lobster_messages
from data.ingestion.lobster_orderbook import (
    join_messages_orderbook,
    load_lobster_orderbook,
)

pytestmark = pytest.mark.data  # CI: skipped via `pytest -m "not data"`

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"

# LOBSTER 2012-06-21 sample stocks (https://lobsterdata.com/info/DataSamples.php).
# Tests run against every stock the local fixture has — missing stocks are
# silently skipped (test_*_for_ticker collects only what's on disk).
ALL_TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def _present_tickers() -> list[str]:
    return [t for t in ALL_TICKERS if list(SAMPLE_DIR.glob(f"{t}_*_message_*.csv"))]


def _paths(ticker: str) -> tuple[Path, Path]:
    msg = next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv"))
    book = next(SAMPLE_DIR.glob(f"{ticker}_*_orderbook_*.csv"))
    return msg, book


# Skip the whole module cleanly if no LOBSTER data is available locally.
if not _present_tickers():
    pytest.skip(
        "No LOBSTER files in data/raw/ — see README quickstart.",
        allow_module_level=True,
    )


@pytest.fixture(params=_present_tickers())
def ticker(request) -> str:
    return request.param


def test_load_lobster_messages_returns_polars_frame(ticker: str) -> None:
    msg_path, _ = _paths(ticker)
    df = load_lobster_messages(msg_path)
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["ts_ns", "event_type", "order_id", "side", "size", "price"]
    assert df.height > 100


def test_lobster_ts_is_datetime(ticker: str) -> None:
    """ts_ns must be Datetime so polars rolling_*_by works without 'i' suffix."""
    msg_path, _ = _paths(ticker)
    df = load_lobster_messages(msg_path)
    assert df["ts_ns"].dtype == pl.Datetime("ns")


def test_lobster_prices_are_int64_ticks(ticker: str) -> None:
    msg_path, _ = _paths(ticker)
    df = load_lobster_messages(msg_path)
    assert df["price"].dtype == pl.Int64
    assert df["price"].min() > 1_000  # not sub-dollar nonsense


def test_lobster_sides_are_signed(ticker: str) -> None:
    msg_path, _ = _paths(ticker)
    df = load_lobster_messages(msg_path)
    sides = set(df["side"].unique().to_list())
    assert sides.issubset({-1, 1})


def test_load_lobster_orderbook_returns_tob_frame(ticker: str) -> None:
    _, book_path = _paths(ticker)
    df = load_lobster_orderbook(book_path, n_levels=10)
    # 40 cols: bid_price_l1..l10, bid_size_l1..l10, ask_*_l1..l10
    assert df.height > 100
    assert "bid_price_l1" in df.columns
    assert "ask_size_l5" in df.columns
    assert df["bid_price_l1"].dtype == pl.Int64
    # Best bid must be < best ask for every row in a healthy market.
    crossed = df.filter(pl.col("bid_price_l1") >= pl.col("ask_price_l1"))
    assert crossed.is_empty(), f"{ticker}: found {crossed.height} crossed rows"


def test_join_messages_orderbook_row_count_matches(ticker: str) -> None:
    msg_path, book_path = _paths(ticker)
    msg = load_lobster_messages(msg_path)
    book = load_lobster_orderbook(book_path, n_levels=10)
    joined = join_messages_orderbook(msg, book)
    assert joined.height == msg.height
    assert joined.height == book.height
    # joined has the message columns AND the book TOB columns
    assert "event_type" in joined.columns
    assert "bid_price_l1" in joined.columns

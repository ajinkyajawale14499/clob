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


def _find_msg_file() -> Path:
    files = list(SAMPLE_DIR.glob("*_message_*.csv"))  # plain CSV after unzip
    assert files, f"No LOBSTER message file in {SAMPLE_DIR} — run Task 0.0 Step 2"
    return files[0]


def test_load_lobster_messages_returns_polars_frame() -> None:
    df = load_lobster_messages(_find_msg_file())
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["ts_ns", "event_type", "order_id", "side", "size", "price"]
    assert df.height > 100


def test_lobster_ts_is_datetime() -> None:
    """ts_ns must be Datetime so polars rolling_*_by works without 'i' suffix."""
    df = load_lobster_messages(_find_msg_file())
    assert df["ts_ns"].dtype == pl.Datetime("ns")


def test_lobster_prices_are_int64_ticks() -> None:
    df = load_lobster_messages(_find_msg_file())
    assert df["price"].dtype == pl.Int64
    assert df["price"].min() > 1_000  # not sub-dollar nonsense


def test_lobster_sides_are_signed() -> None:
    df = load_lobster_messages(_find_msg_file())
    sides = set(df["side"].unique().to_list())
    assert sides.issubset({-1, 1})


def _find_book_file() -> Path:
    files = list(SAMPLE_DIR.glob("*_orderbook_*.csv"))
    assert files, "No LOBSTER orderbook file in data/raw/"
    return files[0]


def test_load_lobster_orderbook_returns_tob_frame() -> None:
    df = load_lobster_orderbook(_find_book_file(), n_levels=10)
    # 40 cols: bid_price_l1..l10, bid_size_l1..l10, ask_*_l1..l10
    assert df.height > 100
    assert "bid_price_l1" in df.columns
    assert "ask_size_l5" in df.columns
    assert df["bid_price_l1"].dtype == pl.Int64
    # Best bid must be < best ask for every row in a healthy market.
    crossed = df.filter(pl.col("bid_price_l1") >= pl.col("ask_price_l1"))
    assert crossed.is_empty(), f"Found {crossed.height} crossed rows"


def test_join_messages_orderbook_row_count_matches() -> None:
    msg = load_lobster_messages(_find_msg_file())
    book = load_lobster_orderbook(_find_book_file(), n_levels=10)
    joined = join_messages_orderbook(msg, book)
    assert joined.height == msg.height
    assert joined.height == book.height
    # joined has the message columns AND the book TOB columns
    assert "event_type" in joined.columns
    assert "bid_price_l1" in joined.columns

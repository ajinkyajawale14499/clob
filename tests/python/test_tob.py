from pathlib import Path

import polars as pl
import pytest

from data.ingestion.binance_bookticker import load_binance_bookticker
from data.ingestion.lobster_message import load_lobster_messages
from data.ingestion.lobster_orderbook import (
    join_messages_orderbook,
    load_lobster_orderbook,
)
from data.tob.unified import binance_to_tob, lobster_to_tob

pytestmark = pytest.mark.data  # CI: skipped via `pytest -m "not data"`

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"

TOB_COLUMNS = ["ts_ns", "bid_price_l1", "bid_size_l1", "ask_price_l1", "ask_size_l1"]


def test_lobster_to_tob_matches_schema() -> None:
    msg = load_lobster_messages(next(SAMPLE_DIR.glob("*_message_*.csv")))
    book = load_lobster_orderbook(next(SAMPLE_DIR.glob("*_orderbook_*.csv")), n_levels=10)
    joined = join_messages_orderbook(msg, book)
    tob = lobster_to_tob(joined)
    assert tob.columns == TOB_COLUMNS
    assert tob.height == joined.height


def test_binance_to_tob_matches_schema() -> None:
    bt = load_binance_bookticker(next(SAMPLE_DIR.glob("*-bookTicker-*.csv")))
    tob = binance_to_tob(bt)
    assert tob.columns == TOB_COLUMNS


def test_unified_tob_is_sorted_by_ts() -> None:
    msg = load_lobster_messages(next(SAMPLE_DIR.glob("*_message_*.csv")))
    book = load_lobster_orderbook(next(SAMPLE_DIR.glob("*_orderbook_*.csv")), n_levels=10)
    tob = lobster_to_tob(join_messages_orderbook(msg, book))
    diffs = tob["ts_ns"].diff().drop_nulls()
    # All non-negative -> sorted.
    assert (diffs >= pl.duration(nanoseconds=0)).all()

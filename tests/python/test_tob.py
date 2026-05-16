from pathlib import Path

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

ALL_TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def _present_tickers() -> list[str]:
    return [t for t in ALL_TICKERS if list(SAMPLE_DIR.glob(f"{t}_*_message_*.csv"))]


def _paths(ticker: str) -> tuple[Path, Path]:
    msg = next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv"))
    book = next(SAMPLE_DIR.glob(f"{ticker}_*_orderbook_*.csv"))
    return msg, book


if not _present_tickers():
    pytest.skip(
        "No LOBSTER files in data/raw/ — see README quickstart.",
        allow_module_level=True,
    )


@pytest.fixture(params=_present_tickers())
def ticker(request) -> str:
    return request.param


def test_lobster_to_tob_matches_schema(ticker: str) -> None:
    msg_path, book_path = _paths(ticker)
    msg = load_lobster_messages(msg_path)
    book = load_lobster_orderbook(book_path, n_levels=10)
    joined = join_messages_orderbook(msg, book)
    tob = lobster_to_tob(joined)
    assert tob.columns == TOB_COLUMNS
    assert tob.height == joined.height


def test_binance_to_tob_matches_schema() -> None:
    bt = load_binance_bookticker(next(SAMPLE_DIR.glob("*-bookTicker-*.csv")))
    tob = binance_to_tob(bt)
    assert tob.columns == TOB_COLUMNS


def test_unified_tob_is_sorted_by_ts(ticker: str) -> None:
    msg_path, book_path = _paths(ticker)
    msg = load_lobster_messages(msg_path)
    book = load_lobster_orderbook(book_path, n_levels=10)
    tob = lobster_to_tob(join_messages_orderbook(msg, book))
    assert tob["ts_ns"].is_sorted()

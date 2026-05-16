from pathlib import Path

import polars as pl
import pytest

from data.ingestion.lobster_message import load_lobster_messages

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

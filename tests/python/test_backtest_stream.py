"""Smoke test for backtest/stream.py against real LOBSTER (data-marked)."""

from collections import Counter
from pathlib import Path

import pytest

# backtest.stream imports clob_py — skip if the C++ bindings aren't built.
pytest.importorskip("clob_py")
from backtest.stream import stream_lobster_events

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"


def _msg_path() -> Path:
    files = list(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))
    return files[0] if files else None


@pytest.mark.skipif(_msg_path() is None, reason="LOBSTER AAPL missing")
def test_stream_yields_only_add_limit_and_cancel():
    counts: Counter[str] = Counter()
    for ev in stream_lobster_events(_msg_path()):
        counts[ev[1]] += 1  # ev = (ts, op, *args)
    # AAPL day has ~70% add_limit, ~25% cancels, rest are execs (skipped).
    assert counts["add_limit"] > 100_000
    assert counts["cancel"] > 10_000
    # No other ops should appear.
    assert set(counts.keys()) <= {"add_limit", "cancel"}


@pytest.mark.skipif(_msg_path() is None, reason="LOBSTER AAPL missing")
def test_stream_chronological_order():
    """ts_ns must be non-decreasing."""
    prev = None
    for i, (ts, *_) in enumerate(stream_lobster_events(_msg_path())):
        if prev is not None:
            assert ts >= prev, f"row {i}: ts {ts} < prev {prev}"
        prev = ts
        if i > 10_000:
            break  # 10k rows is enough to catch ordering bugs

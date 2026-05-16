"""LOBSTER message -> Engine operation stream.

Each LOBSTER event_type maps to a clob_py Engine call:
  1 = new limit order      -> Engine.add_limit(order_id, side, price, size)
  2 = partial cancel       -> approximated as Engine.cancel (v1 limitation;
                              ADR 0009 caveat)
  3 = full cancel          -> Engine.cancel
  4 = visible execution    -> SKIP (matched implicitly by add_limit's cross)
  5 = hidden execution     -> SKIP (hidden orders never enter our Book)
  6 = cross / auction      -> SKIP (out of scope)
  7 = trading halt         -> SKIP

The output is a generator of (ts_ns, op_name, *args) tuples. The driver in
backtest/driver.py dispatches these to Engine.{add_limit,cancel}.
"""

from collections.abc import Iterator
from pathlib import Path

import clob_py

from data.ingestion.lobster_message import load_lobster_messages


def stream_lobster_events(path: Path) -> Iterator[tuple]:
    """Yields (ts_ns, op_name, *args) tuples in chronological order.

    op_name is one of {"add_limit", "cancel"}. args match the Engine method's
    parameter order.
    """
    msg = load_lobster_messages(path)
    # Iterate as a list of dicts — slower than columnar, but the backtest is
    # already Python-bound. Total ~400k rows per stock.
    for row in msg.iter_rows(named=True):
        ts = row["ts_ns"]
        et = row["event_type"]
        if et == 1:
            side = clob_py.Side.Bid if row["side"] == 1 else clob_py.Side.Ask
            yield (ts, "add_limit", row["order_id"], side, row["price"], row["size"])
        elif et in (2, 3):
            yield (ts, "cancel", row["order_id"])
        # 4, 5, 6, 7: skip

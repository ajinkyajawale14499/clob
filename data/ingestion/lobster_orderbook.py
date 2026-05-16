"""LOBSTER orderbook-file ingestion — top-N snapshots per event.

Each row of the orderbook file corresponds 1:1 with a row of the message file.
Columns (no header): ask_price_1, ask_size_1, bid_price_1, bid_size_1,
                     ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
"""

from pathlib import Path

import polars as pl


def load_lobster_orderbook(path: Path, *, n_levels: int) -> pl.DataFrame:
    """Parse a LOBSTER orderbook CSV into per-level columns named
    bid_price_l<i>, bid_size_l<i>, ask_price_l<i>, ask_size_l<i> for i in 1..N."""
    # Generate column names matching the LOBSTER native column ordering
    raw_names: list[str] = []
    for level in range(1, n_levels + 1):
        raw_names += [
            f"ask_price_{level}_raw",
            f"ask_size_{level}_raw",
            f"bid_price_{level}_raw",
            f"bid_size_{level}_raw",
        ]

    raw = pl.read_csv(
        path,
        has_header=False,
        new_columns=raw_names,
        schema_overrides={c: pl.Int64 for c in raw_names},
    )

    # Rename to the canonical schema: bid_price_l1, bid_size_l1, ask_price_l1, ...
    projections: list[pl.Expr] = []
    for level in range(1, n_levels + 1):
        projections += [
            pl.col(f"bid_price_{level}_raw").alias(f"bid_price_l{level}"),
            pl.col(f"bid_size_{level}_raw").alias(f"bid_size_l{level}"),
            pl.col(f"ask_price_{level}_raw").alias(f"ask_price_l{level}"),
            pl.col(f"ask_size_{level}_raw").alias(f"ask_size_l{level}"),
        ]
    return raw.select(projections)


def join_messages_orderbook(messages: pl.DataFrame, orderbook: pl.DataFrame) -> pl.DataFrame:
    """LOBSTER guarantees 1:1 row correspondence between message and orderbook files.
    We use horizontal concat (by row index) to align."""
    assert messages.height == orderbook.height, (
        f"row mismatch: messages={messages.height}, orderbook={orderbook.height}"
    )
    return pl.concat([messages, orderbook], how="horizontal")

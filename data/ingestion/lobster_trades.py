"""LOBSTER trade extraction from message file.

Event types from https://lobsterdata.com/info/DataStructure.php:
  4 = visible execution (a marketable order takes a resting one)
  5 = hidden execution (LOBSTER's L2 doesn't see hidden orders; the trade still prints)

LOBSTER's `direction` column on exec rows is the sign of the RESTING order
(the maker), so the AGGRESSOR side is `-direction`. We flip here so downstream
features (signed_trade_flow, TFI) consume aggressor-signed sizes directly.
"""

from pathlib import Path

import polars as pl


def extract_trades(messages: pl.DataFrame) -> pl.DataFrame:
    """Filter event_type ∈ {4, 5} from messages; flip resting-direction → aggressor side.

    Output schema:
        ts_ns          Datetime("ns")
        order_id       Int64        — the RESTING order's id (LOBSTER convention)
        size           Int64        — shares filled
        price          Int64        — fill price in LOBSTER ticks ($0.0001)
        aggressor_side Int8         — +1 buy aggressor, -1 sell aggressor
    """
    return messages.filter(pl.col("event_type").is_in([4, 5])).select(
        pl.col("ts_ns"),
        pl.col("order_id"),
        pl.col("size"),
        pl.col("price"),
        (-pl.col("side")).cast(pl.Int8).alias("aggressor_side"),
    )


def extract_trades_from_path(path: Path) -> pl.DataFrame:
    """Convenience: load + filter in one step."""
    from data.ingestion.lobster_message import load_lobster_messages
    return extract_trades(load_lobster_messages(path))

"""Unify LOBSTER and Binance into a single 5-column top-of-book schema.

Output columns (the contract for the feature library):
    ts_ns        Datetime("ns")
    bid_price_l1 Int64 (ticks)
    bid_size_l1  Int64
    ask_price_l1 Int64
    ask_size_l1  Int64
"""

import polars as pl

TOB_COLUMNS = ["ts_ns", "bid_price_l1", "bid_size_l1", "ask_price_l1", "ask_size_l1"]


def lobster_to_tob(joined: pl.DataFrame) -> pl.DataFrame:
    """`joined` is the message+orderbook merge from Task 1.3."""
    return joined.select(TOB_COLUMNS).sort("ts_ns")


def binance_to_tob(bookticker: pl.DataFrame) -> pl.DataFrame:
    """bookticker already has the right shape; just sort and select."""
    return bookticker.select(TOB_COLUMNS).sort("ts_ns")

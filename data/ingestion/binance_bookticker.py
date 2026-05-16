"""Binance public-data bookTicker ingestion.

Source: https://data.binance.vision/data/futures/um/daily/bookTicker/<SYM>/

Native schema (as of 2024+):
    update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
    transaction_time, event_time

Prices in Binance native: float USDT — we convert to int64 ticks at 1e-8 precision.
"""

from pathlib import Path

import polars as pl

PRICE_SCALE = 100_000_000  # 1 USDT = 1e8 ticks (8-decimal precision)
SIZE_SCALE = 100_000_000   # same for sizes; quantities are float BTC


def load_binance_bookticker(path: Path) -> pl.DataFrame:
    """Parse Binance futures bookTicker CSV -> normalized BBO frame."""
    raw = pl.read_csv(path)
    return raw.select(
        # transaction_time is ms since epoch; cast to Datetime("ns") via x 1e6
        (pl.col("transaction_time") * 1_000_000).cast(pl.Int64)
        .cast(pl.Datetime("ns"))
        .alias("ts_ns"),
        (pl.col("best_bid_price") * PRICE_SCALE).cast(pl.Int64).alias("bid_price_l1"),
        (pl.col("best_bid_qty") * SIZE_SCALE).cast(pl.Int64).alias("bid_size_l1"),
        (pl.col("best_ask_price") * PRICE_SCALE).cast(pl.Int64).alias("ask_price_l1"),
        (pl.col("best_ask_qty") * SIZE_SCALE).cast(pl.Int64).alias("ask_size_l1"),
    )

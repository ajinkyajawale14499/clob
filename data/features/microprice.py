"""Microprice — size-weighted midpoint.
microprice = (bid_size x ask_price + ask_size x bid_price) / (bid_size + ask_size)
Reference: Stoikov, "The Micro-Price" (2017).
"""

import polars as pl


def microprice(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (
                pl.col("bid_size_l1") * pl.col("ask_price_l1")
                + pl.col("ask_size_l1") * pl.col("bid_price_l1")
            )
            / (pl.col("bid_size_l1") + pl.col("ask_size_l1"))
        ).alias("microprice"),
    )

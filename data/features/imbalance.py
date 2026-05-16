"""Top-of-book size imbalance in [-1, +1], NULL when both sides empty."""

import polars as pl


def imbalance(df: pl.DataFrame) -> pl.DataFrame:
    total = pl.col("bid_size_l1") + pl.col("ask_size_l1")
    return df.with_columns(
        pl.when(total == 0)
        .then(None)
        .otherwise(
            (pl.col("bid_size_l1") - pl.col("ask_size_l1")).cast(pl.Float64) / total
        )
        .alias("imbalance"),
    )

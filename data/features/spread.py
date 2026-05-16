"""Spread features: spread_ticks + spread_zscore.

References:
    Pascual & Veredas 2010 — Spread regimes and short-term predictability
    Cont 2011 — Statistical modeling of high-frequency financial data
"""

import polars as pl


def spread_ticks(df: pl.DataFrame, *, tick_size: int = 100) -> pl.DataFrame:
    """Add `spread_ticks` = (ask_l1 - bid_l1) / tick_size.

    LOBSTER native: 1 tick = $0.0001 = 100 price-integer units. Default
    tick_size=100 matches LOBSTER; pass tick_size=1 for already-tick-scaled data.
    """
    return df.with_columns(
        ((pl.col("ask_price_l1") - pl.col("bid_price_l1")) / tick_size)
        .cast(pl.Float64)
        .alias("spread_ticks"),
    )


def spread_zscore(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Add `spread_zscore_<window>` = (spread - rolling_mean) / rolling_std.

    Captures spread regime: a wide spread in a normally-wide regime is uninformative,
    but a wide spread in a normally-tight regime predicts upcoming volatility.

    When the rolling std is 0 (constant-spread regime — common in liquid stocks
    where every event keeps spread at 1 tick), returns 0.0 rather than ±Inf so
    the ML pipeline can ingest the column without a fill-Inf step downstream.
    """
    col = f"spread_zscore_{window}"
    return df.with_columns(
        _sp=(pl.col("ask_price_l1") - pl.col("bid_price_l1")).cast(pl.Float64)
    ).with_columns(
        _mean=pl.col("_sp").rolling_mean(window_size=window),
        _std=pl.col("_sp").rolling_std(window_size=window),
    ).with_columns(
        pl.when((pl.col("_std") == 0) | pl.col("_std").is_null())
        .then(0.0)
        .otherwise((pl.col("_sp") - pl.col("_mean")) / pl.col("_std"))
        .alias(col)
    ).drop("_sp", "_mean", "_std")

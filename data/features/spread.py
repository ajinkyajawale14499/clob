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
    """
    col = f"spread_zscore_{window}"
    return df.with_columns(
        _sp=(pl.col("ask_price_l1") - pl.col("bid_price_l1")).cast(pl.Float64)
    ).with_columns(
        ((pl.col("_sp") - pl.col("_sp").rolling_mean(window_size=window))
         / pl.col("_sp").rolling_std(window_size=window)).alias(col)
    ).drop("_sp")

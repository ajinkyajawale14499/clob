"""Realized volatility feature: rolling sqrt(Σ(Δmid)²) over an event window.

References:
    Andersen & Bollerslev 1998 — Answering the skeptics: realized volatility
    Parkinson 1980 — extreme-value vol estimators
"""

import polars as pl


def realized_vol(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Add `realized_vol_<window>` = sqrt(Σ(Δmid)²) over the last `window` events.

    Assumes `df` has `bid_price_l1` and `ask_price_l1` (Int64 ticks). Computes
    mid as (bid + ask) / 2 (Float64 to retain precision), then diff -> sum of
    squares -> sqrt. NaN-clean: first `window-1` rows are null.
    """
    col = f"realized_vol_{window}"
    return df.with_columns(
        _mid=((pl.col("bid_price_l1") + pl.col("ask_price_l1")) / 2.0)
    ).with_columns(
        _ret=pl.col("_mid").diff().fill_null(0.0)
    ).with_columns(
        (pl.col("_ret") ** 2).rolling_sum(window_size=window).sqrt().alias(col)
    ).drop("_mid", "_ret")

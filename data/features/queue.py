"""Queue-depletion features: EWMA of Δbest-queue-size per event.

References:
    Moallemi & Yuan 2016 — Queue position valuation in a limit-order book
    Gould & Bonart 2016 — Queue imbalance as a price predictor
"""

import polars as pl


def queue_depletion_ewma(df: pl.DataFrame, *, alpha: float = 0.05) -> pl.DataFrame:
    """Add `queue_depletion_bid` + `queue_depletion_ask` = EWMA of Δ-per-event.

    Signed: positive = queue growing (wall building), negative = queue shrinking.
    Bid and ask tracked separately because their dynamics encode different signals
    (e.g., ask shrinking + bid stable => upward pressure on mid).

    Default alpha=0.05 -> ~20-event effective window. Tune via grid in W9 eval.
    """
    return df.with_columns(
        _d_bid=pl.col("bid_size_l1").diff().fill_null(0).cast(pl.Float64),
        _d_ask=pl.col("ask_size_l1").diff().fill_null(0).cast(pl.Float64),
    ).with_columns(
        pl.col("_d_bid").ewm_mean(alpha=alpha, adjust=False).alias("queue_depletion_bid"),
        pl.col("_d_ask").ewm_mean(alpha=alpha, adjust=False).alias("queue_depletion_ask"),
    ).drop("_d_bid", "_d_ask")

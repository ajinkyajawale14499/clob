"""Trade-flow features: signed_trade_flow + TFI (trade-flow imbalance).

References:
    Cont, Kukanov, Stoikov (2014) — Price Impact of Order Book Events
    Kolm, Turiel, Westray (2023) — Deep Order Flow Imbalance (multi-horizon)

Convention: trades input must have `aggressor_side` (+1 buy / -1 sell) per
`data/ingestion/lobster_trades.extract_trades`.
"""

import polars as pl


def signed_trade_flow(trades: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Add `signed_trade_flow_<window>` = rolling sum of (aggressor_side * size)
    over the last `window` trades. Positive -> net buying pressure."""
    col = f"signed_trade_flow_{window}"
    return trades.with_columns(
        (pl.col("aggressor_side").cast(pl.Int64) * pl.col("size"))
        .rolling_sum(window_size=window)
        .alias(col),
    )


def trade_flow_imbalance(trades: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Add `tfi_<window>` = (buy_vol - sell_vol) / total_vol over last `window`
    trades. Bounded in [-1, +1]. Returns 0.0 when the window has no trades."""
    col = f"tfi_{window}"
    buys = pl.when(pl.col("aggressor_side") == 1).then(pl.col("size")).otherwise(0)
    sells = pl.when(pl.col("aggressor_side") == -1).then(pl.col("size")).otherwise(0)
    return trades.with_columns(
        _buy=buys.rolling_sum(window_size=window),
        _sell=sells.rolling_sum(window_size=window),
    ).with_columns(
        pl.when((pl.col("_buy") + pl.col("_sell")) == 0)
        .then(0.0)
        .otherwise(
            (pl.col("_buy") - pl.col("_sell")).cast(pl.Float64)
            / (pl.col("_buy") + pl.col("_sell"))
        )
        .alias(col)
    ).drop("_buy", "_sell")

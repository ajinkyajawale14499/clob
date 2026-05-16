"""Order Flow Imbalance (OFI).
Reference: Cont, Kukanov, Stoikov (2014).
"""

import polars as pl


def order_flow_imbalance(df: pl.DataFrame, *, window: str) -> pl.DataFrame:
    """Add `ofi` column = rolling sum of (Δbid_size_l1 - Δask_size_l1) over `window`.

    Args:
        df: frame sorted by `ts_ns` (Datetime). Must contain bid_size_l1, ask_size_l1.
        window: polars duration string (e.g., "1s", "100ms"). Datetime by-column
            allows ns/ms/s/m.
    """
    return df.with_columns(
        bid_delta=pl.col("bid_size_l1").diff().fill_null(0),
        ask_delta=pl.col("ask_size_l1").diff().fill_null(0),
    ).with_columns(
        (pl.col("bid_delta") - pl.col("ask_delta"))
        .rolling_sum_by("ts_ns", window_size=window)
        .alias("ofi"),
    ).drop("bid_delta", "ask_delta")

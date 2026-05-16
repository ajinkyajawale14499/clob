"""Order Flow Imbalance (OFI) — L1 and multi-level variants.
References:
    Cont, Kukanov, Stoikov (2014) — Price Impact of Order Book Events
    Xu, Cont, Kim (2021) — Cross-Impact of Order Flow Imbalance in Equity Markets
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


def order_flow_imbalance_event(df: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Same as order_flow_imbalance but window is in EVENTS (Int), not duration.

    Used by the W9 training pipeline where features are indexed by message_id
    (not by wall-clock — see ADR 0007). Output column: `ofi_<window>`.
    """
    col = f"ofi_{window}"
    return df.with_columns(
        _bd=pl.col("bid_size_l1").diff().fill_null(0),
        _ad=pl.col("ask_size_l1").diff().fill_null(0),
    ).with_columns(
        (pl.col("_bd") - pl.col("_ad"))
        .rolling_sum(window_size=window)
        .alias(col),
    ).drop("_bd", "_ad")


def multi_level_ofi(df: pl.DataFrame, *, levels: list[int], window: int) -> pl.DataFrame:
    """Add `mlofi_l<lo>_l<hi>_w<window>` = rolling sum of Σ_lv (Δbid_lv - Δask_lv).

    Xu-Cont-Kim 2021 show ~30% R² lift over L1 OFI for short-horizon mid-price
    prediction. We pass `levels=[2,3,4,5]` for L2-L5; L1 is captured by
    `order_flow_imbalance_event`.
    """
    if not levels:
        raise ValueError("levels must be non-empty")
    lo, hi = min(levels), max(levels)
    col = f"mlofi_l{lo}_l{hi}_w{window}"
    bid_delta_expr = sum(
        pl.col(f"bid_size_l{lv}").diff().fill_null(0) for lv in levels
    )
    ask_delta_expr = sum(
        pl.col(f"ask_size_l{lv}").diff().fill_null(0) for lv in levels
    )
    return df.with_columns(
        (bid_delta_expr - ask_delta_expr)
        .rolling_sum(window_size=window)
        .alias(col),
    )

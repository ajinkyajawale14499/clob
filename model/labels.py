"""3-class mid-direction labels with 1-tick deadband.

Per ADR 0008 (Briola 2024 §3.1): binary sign() on raw LOBSTER mid is dominated
by stable-class noise (large-tick stocks 65-95% stable at K=10-100). The
3-class formulation with a 1-tick deadband is the literature default and
mirrors what production microstructure ML systems use.

Encoding:
    0 = Down   (mid[t+K] - mid[t] <= -θ ticks)
    1 = Stable (-θ ticks < delta < +θ ticks)
    2 = Up     (mid[t+K] - mid[t] >= +θ ticks)
"""

import polars as pl

from model.schema import LABEL_DEADBAND_TICKS, LABEL_TICK_SIZE


def make_labels_3class(
    tob: pl.DataFrame,
    *,
    k_events: int,
    deadband_ticks: int = LABEL_DEADBAND_TICKS,
    tick_size: int = LABEL_TICK_SIZE,
) -> pl.DataFrame:
    """Compute mid[t+K] - mid[t] in ticks; threshold via `deadband_ticks`.

    Drops the trailing K rows (no future mid available).

    Args:
        tob: must contain `bid_price_l1`, `ask_price_l1` (Int64).
        k_events: lookahead horizon in event count.
        deadband_ticks: |delta| < deadband_ticks -> Stable (label=1). Default 1.
        tick_size: LOBSTER native = 100 price-int units per tick.

    Returns:
        DataFrame with all original columns plus `label` (Int8 ∈ {0, 1, 2}).
        Row count = tob.height - k_events.
    """
    if k_events < 1:
        raise ValueError(f"k_events must be >= 1, got {k_events}")

    mid_x2 = pl.col("bid_price_l1") + pl.col("ask_price_l1")  # 2*mid (no float)
    return (
        tob.with_columns(
            _mid_now_x2=mid_x2,
            _mid_future_x2=mid_x2.shift(-k_events),
        )
        .with_columns(
            _delta_ticks=(
                (pl.col("_mid_future_x2") - pl.col("_mid_now_x2")) / (2 * tick_size)
            )
        )
        .with_columns(
            label=pl.when(pl.col("_delta_ticks") <= -deadband_ticks)
            .then(0)
            .when(pl.col("_delta_ticks") >= deadband_ticks)
            .then(2)
            .otherwise(1)
            .cast(pl.Int8)
        )
        .drop_nulls("_mid_future_x2")
        .drop("_mid_now_x2", "_mid_future_x2", "_delta_ticks")
    )


def class_share(labels: pl.Series) -> dict[int, float]:
    """Returns the share of each class label in [0, 1].

    Used in train.py for sanity logging and ADR rationale (e.g., "INTC at K=10
    is 70% Stable" justifies balanced under-sampling)."""
    counts = labels.value_counts().sort("label")
    total = labels.len()
    out: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
    for row in counts.iter_rows(named=True):
        out[int(row["label"])] = int(row["count"]) / total
    return out

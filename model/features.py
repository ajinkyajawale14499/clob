"""Canonical feature computation pipeline.

`build_feature_matrix(joined_tob, trades, ticker)` -> DataFrame whose columns
are exactly `["ts_ns"] + FEATURE_NAMES`, with all features cast to float32.

This is the SINGLE Python entry point for training-time feature construction.
The C++ FeatureState in W10 mirrors this same pipeline at inference time;
parity is enforced by `tests/python/test_onnx_cpp_parity.py` (W10).
"""

from pathlib import Path

import polars as pl

from data.features.imbalance import imbalance
from data.features.ofi import multi_level_ofi, order_flow_imbalance_event
from data.features.queue import queue_depletion_ewma
from data.features.spread import spread_ticks, spread_zscore
from data.features.trades import signed_trade_flow, trade_flow_imbalance
from data.features.vol import realized_vol
from model.microprice_g import MicropriceLut
from model.schema import (
    ALL_TICKERS,
    FEATURE_NAMES,
    LABEL_TICK_SIZE,
)


def build_feature_matrix(
    joined: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    ticker: str,
    lut: MicropriceLut,
) -> pl.DataFrame:
    """Compose the 19-feature matrix from a joined TOB + trades frame.

    Args:
        joined: from `data.ingestion.lobster_orderbook.join_messages_orderbook(msg, book)`.
            Must contain `ts_ns`, `bid_price_l1..l10`, `bid_size_l1..l10`,
            `ask_price_l1..l10`, `ask_size_l1..l10`.
        trades: from `data.ingestion.lobster_trades.extract_trades(msg)`.
            Must contain `ts_ns`, `aggressor_side`, `size`.
        ticker: one of ALL_TICKERS; sets the ticker one-hot.
        lut: pre-fitted Stoikov G(I, S) lookup table.

    Returns:
        DataFrame with columns `["ts_ns"] + FEATURE_NAMES`, all features cast
        to float32. Row count == joined.height. The first few hundred rows have
        nulls in rolling features; the warm flags signal when each window is ready.
    """
    if ticker not in ALL_TICKERS:
        raise ValueError(f"ticker {ticker!r} not in {ALL_TICKERS}")

    # 1. Apply all book-shape + flow features in one pipe chain.
    # `imbalance()` from W3 outputs column "imbalance"; schema expects "imbalance_l1"
    # (the L1 qualifier is meaningful — future depth-N variants would differ).
    df = (
        joined
        .pipe(imbalance)
        .rename({"imbalance": "imbalance_l1"})
        .pipe(spread_ticks, tick_size=LABEL_TICK_SIZE)
        .pipe(spread_zscore, window=200)
        .pipe(order_flow_imbalance_event, window=50)
        .pipe(order_flow_imbalance_event, window=200)
        .pipe(multi_level_ofi, levels=[2, 3, 4, 5], window=50)
        .pipe(realized_vol, window=200)
        .pipe(queue_depletion_ewma, alpha=0.05)
    )

    # 2. Stoikov microprice deviation (vectorized LUT lookup).
    imb_vals = df["imbalance_l1"].to_numpy()
    sp_vals = df["spread_ticks"].to_numpy().astype(int)
    g_adj = lut.lookup_vec(imb_vals, sp_vals)
    df = df.with_columns(pl.Series("microprice_g_dev", g_adj))

    # 3. Compute trade-flow features ON THE TRADES frame (separate cadence),
    #    then asof-join back to the TOB frame on ts_ns (backward strategy:
    #    each TOB row gets the LAST seen trade-flow value).
    tf = trades
    if tf.height > 0:
        tf = (
            tf.pipe(signed_trade_flow, window=50)
            .pipe(trade_flow_imbalance, window=50)
            .select("ts_ns", "signed_trade_flow_50", "tfi_50")
            .sort("ts_ns")
        )
        df = (
            df.sort("ts_ns")
            .join_asof(tf, on="ts_ns", strategy="backward")
            .with_columns(
                pl.col("signed_trade_flow_50").fill_null(0).cast(pl.Float64),
                pl.col("tfi_50").fill_null(0.0).cast(pl.Float64),
            )
        )
    else:
        # No trades at all (shouldn't happen for LOBSTER but defensive).
        df = df.with_columns(
            pl.lit(0.0).alias("signed_trade_flow_50"),
            pl.lit(0.0).alias("tfi_50"),
        )

    # 4. Ticker one-hots.
    df = df.with_columns([
        pl.lit(1.0 if t == ticker else 0.0).alias(f"ticker_{t}")
        for t in ALL_TICKERS
    ])

    # 5. Warm flags — 1.0 once the longest-window rolling features are populated.
    n = df.height
    df = df.with_columns(
        (pl.int_range(0, n) >= 50).cast(pl.Float32).alias("is_warm_50"),
        (pl.int_range(0, n) >= 200).cast(pl.Float32).alias("is_warm_200"),
    )

    # 6. Fill nulls in rolling features with 0.0 (warmflags signal validity).
    for col in FEATURE_NAMES:
        if col in df.columns and df[col].null_count() > 0:
            df = df.with_columns(pl.col(col).fill_null(0.0))

    # 7. Rename mlofi column to canonical name (the function emits with window suffix).
    if "mlofi_l2_l5_w50" not in df.columns:
        for c in df.columns:
            if c.startswith("mlofi_"):
                df = df.rename({c: "mlofi_l2_l5_w50"})
                break

    # 8. Final select + dtype cast (Float32 matches C++ FeatureState output).
    return df.select(
        ["ts_ns"] + [pl.col(name).cast(pl.Float32) for name in FEATURE_NAMES]
    )


def load_stock(
    ticker: str,
    *,
    sample_dir: Path = Path("data/raw"),
    lut: MicropriceLut | None = None,
) -> pl.DataFrame:
    """Convenience: load one LOBSTER stock and build its feature matrix.

    Returns DataFrame with columns ["ts_ns"] + FEATURE_NAMES.
    """
    from data.ingestion.lobster_message import load_lobster_messages
    from data.ingestion.lobster_orderbook import (
        join_messages_orderbook,
        load_lobster_orderbook,
    )
    from data.ingestion.lobster_trades import extract_trades

    msg_path = next(sample_dir.glob(f"{ticker}_*_message_*.csv"))
    book_path = next(sample_dir.glob(f"{ticker}_*_orderbook_*.csv"))

    msg = load_lobster_messages(msg_path)
    book = load_lobster_orderbook(book_path, n_levels=10)
    joined = join_messages_orderbook(msg, book)
    trades = extract_trades(msg)

    if lut is None:
        lut_path = sample_dir.parent.parent / "model" / "artifacts" / "microprice_g.json"
        if not lut_path.exists():
            raise FileNotFoundError(
                f"Microprice LUT not found at {lut_path}. "
                "Regenerate via `uv run python -m model.microprice_g` or pass `lut=`."
            )
        lut = MicropriceLut.load(lut_path)

    return build_feature_matrix(joined, trades, ticker=ticker, lut=lut)

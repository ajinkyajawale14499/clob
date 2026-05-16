"""Single source of truth for the v1.0 feature contract.

This module is imported by:
  - model/features.py   — Python feature computation pipeline (training time)
  - model/labels.py     — label generator
  - model/train.py      — training entry point
  - tests/python/test_feature_schema.py — schema invariants
  - bindings/pyclob.cpp — pybind11 marshalls these names into ScoredFeatures
  - core/scoring/feature_state.hpp — C++ struct field order MUST match
                                     FEATURE_NAMES below. Enforced via the
                                     train/serve skew test in W10.

If you change FEATURE_NAMES (add / remove / reorder), bump SCHEMA_VERSION,
update the C++ ScoredFeatures struct in the same PR, and re-generate
model.onnx — the input shape gates downstream code.
"""

from typing import Final

# 19 features total. Order is the on-the-wire contract.
FEATURE_NAMES: Final[list[str]] = [
    # microprice (Stoikov G(I,S) — adjustment relative to mid, in tick units)
    "microprice_g_dev",
    # book-shape (L1 only)
    "imbalance_l1",
    "spread_ticks",
    "spread_zscore_200",
    # order flow (L1 + multi-level)
    "ofi_50",
    "ofi_200",
    "mlofi_l2_l5_w50",
    # trade flow (LOBSTER event_type ∈ {4,5}, aggressor-signed)
    "signed_trade_flow_50",
    "tfi_50",
    # volatility + queue dynamics
    "realized_vol_200",
    "queue_depletion_bid",
    "queue_depletion_ask",
    # ticker one-hot — pooled training across all 5 LOBSTER stocks
    "ticker_AAPL",
    "ticker_AMZN",
    "ticker_GOOG",
    "ticker_INTC",
    "ticker_MSFT",
    # warm flags — 1.0 when the 50/200-event rolling buffers have ≥ N obs
    "is_warm_50",
    "is_warm_200",
]
assert len(FEATURE_NAMES) == 19, "FEATURE_NAMES must be 19 (see ADR 0008)"

# Stocks in the pooled training set. Order MUST match ticker one-hot order above.
ALL_TICKERS: Final[list[str]] = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]

# Label horizon grid — train.py grid-searches K, picks best val multi_logloss.
# Range from literature: DeepLOB k ∈ {10,20,30,50,100}; Briola 2024 {10,50,100}.
# K>100 drifts into macro and is empirically not useful per agent-3 research.
LABEL_K_GRID: Final[list[int]] = [10, 50, 100]

# Label encoding: 3-class with 1-tick deadband per ADR 0008.
# (mid[t+K] - mid[t]) in ticks: <= -θ -> 0 (Down), |x| < θ -> 1 (Stable), >= θ -> 2 (Up).
LABEL_DEADBAND_TICKS: Final[int] = 1
LABEL_CLASSES: Final[tuple[int, ...]] = (0, 1, 2)
LABEL_CLASS_NAMES: Final[dict[int, str]] = {0: "Down", 1: "Stable", 2: "Up"}
LABEL_TICK_SIZE: Final[int] = 100  # LOBSTER native: 1 tick = 100 price-int units

# LightGBM hyperparameters — research-anchored (Yu 2023, Kaggle Optiver, ADR 0006).
LIGHTGBM_PARAMS: Final[dict] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    # Cap at 300 to stay below ONNX float32 drift knee (ADR 0006).
    "n_estimators": 300,
}
LIGHTGBM_EARLY_STOPPING_ROUNDS: Final[int] = 50

# Float dtype throughout — must match C++ FeatureState's float32 output.
FEATURE_DTYPE: Final[str] = "float32"

# Schema version — bump if FEATURE_NAMES changes; gates pickle/ONNX compat.
SCHEMA_VERSION: Final[int] = 1


def class_probs_to_score(probs):
    """Inference-time score derived from 3-class softmax.

    score = P(Up) - P(Down) ∈ [-1, +1]; positive => model expects upward move.
    The matcher's ScoreSink emits this value; backtest's MLAwareMaker thresholds
    on |score|.

    Args:
        probs: numpy array of shape (N, 3) with columns [Down, Stable, Up].

    Returns:
        numpy array of shape (N,).
    """
    return probs[:, 2] - probs[:, 0]

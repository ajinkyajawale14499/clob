"""Schema invariants for model/schema.py.

These tests are NOT data-marked — they only check static contracts. They run
in CI and fail loudly if anyone accidentally drops, reorders, or duplicates a
feature name without bumping SCHEMA_VERSION.
"""

import numpy as np
import pytest

from model.schema import (
    ALL_TICKERS,
    FEATURE_DTYPE,
    FEATURE_NAMES,
    LABEL_CLASS_NAMES,
    LABEL_CLASSES,
    LABEL_DEADBAND_TICKS,
    LABEL_K_GRID,
    LIGHTGBM_PARAMS,
    SCHEMA_VERSION,
    class_probs_to_score,
)


def test_feature_count_is_19():
    assert len(FEATURE_NAMES) == 19


def test_no_duplicate_feature_names():
    assert len(set(FEATURE_NAMES)) == 19


def test_ticker_one_hots_complete_and_ordered():
    onehots = [n for n in FEATURE_NAMES if n.startswith("ticker_")]
    assert onehots == [f"ticker_{t}" for t in ALL_TICKERS]


def test_warm_flags_present():
    assert "is_warm_50" in FEATURE_NAMES
    assert "is_warm_200" in FEATURE_NAMES


def test_schema_version_pinned():
    # Bump only when FEATURE_NAMES changes — gates pickle/ONNX compat.
    assert SCHEMA_VERSION == 1


def test_label_k_grid_in_literature_range():
    """K-grid must match the literature-anchored {10, 50, 100} range."""
    assert LABEL_K_GRID == [10, 50, 100]


def test_label_deadband_is_one_tick():
    """Per Briola 2024 / ADR 0008. Tunable, but the v1 default is 1."""
    assert LABEL_DEADBAND_TICKS == 1


def test_label_classes_are_3_class():
    assert LABEL_CLASSES == (0, 1, 2)
    assert LABEL_CLASS_NAMES == {0: "Down", 1: "Stable", 2: "Up"}


def test_lightgbm_objective_is_multiclass():
    assert LIGHTGBM_PARAMS["objective"] == "multiclass"
    assert LIGHTGBM_PARAMS["num_class"] == 3


def test_lightgbm_n_estimators_capped_per_adr_0006():
    """Hard cap to stay below ONNX float32 drift knee. Don't raise without ADR update."""
    assert LIGHTGBM_PARAMS["n_estimators"] <= 300


def test_feature_dtype_is_float32():
    """C++ FeatureState emits float32; Python must match."""
    assert FEATURE_DTYPE == "float32"


def test_class_probs_to_score_basics():
    # P(Up)=1, P(Down)=0 -> score=+1
    probs = np.array([[0.0, 0.0, 1.0]])
    assert class_probs_to_score(probs)[0] == 1.0
    # P(Down)=1 -> score=-1
    probs = np.array([[1.0, 0.0, 0.0]])
    assert class_probs_to_score(probs)[0] == -1.0
    # All-Stable -> score=0
    probs = np.array([[0.0, 1.0, 0.0]])
    assert class_probs_to_score(probs)[0] == 0.0


def test_class_probs_to_score_batched():
    probs = np.array([
        [0.1, 0.2, 0.7],   # mild up
        [0.5, 0.3, 0.2],   # down lean
        [0.33, 0.34, 0.33],  # neutral
    ])
    scores = class_probs_to_score(probs)
    assert scores.shape == (3,)
    assert scores[0] == pytest.approx(0.6)
    assert scores[1] == pytest.approx(-0.3)
    assert scores[2] == pytest.approx(0.0, abs=1e-9)


# ----- data-marked: requires LOBSTER + a pre-fitted microprice LUT -----

from pathlib import Path  # noqa: E402

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"
LUT_PATH = Path(__file__).parents[2] / "model" / "artifacts" / "microprice_g.json"


def _have_lobster_and_lut() -> bool:
    return (
        any(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))
        and LUT_PATH.exists()
    )


@pytest.mark.data
@pytest.mark.skipif(not _have_lobster_and_lut(),
                    reason="LOBSTER files or microprice_g.json missing")
def test_build_feature_matrix_real_lobster_smoke():
    """Smoke: build_feature_matrix runs end-to-end on real AAPL, no NaN/Inf."""
    from model.features import load_stock
    feats = load_stock("AAPL", sample_dir=SAMPLE_DIR)
    assert feats.columns == ["ts_ns", *FEATURE_NAMES]
    assert feats.height > 100_000
    # Every feature column is float32 and fully finite (no NaN/Inf).
    for name in FEATURE_NAMES:
        s = feats[name]
        assert str(s.dtype) == "Float32", f"{name}: dtype={s.dtype}"
        assert s.is_finite().all(), f"{name}: has non-finite values"

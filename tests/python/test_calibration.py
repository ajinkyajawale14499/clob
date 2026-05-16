"""Calibration smoke test for the trained model (data-marked).

Brier score gates:
    - Uniform 1/3 baseline: 0.667
    - Random binary: 0.25 (irrelevant here — we're multiclass)
    - 'Decent' signal on 3-class: < 0.55
    - 'Strong' signal: < 0.45
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).parents[2] / "model" / "artifacts"


def _have_lobster_and_model() -> bool:
    return (
        len(list(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))) > 0
        and (ARTIFACTS_DIR / "model.lgb").exists()
    )


if not _have_lobster_and_model():
    pytest.skip("LOBSTER or trained model.lgb missing", allow_module_level=True)


@pytest.fixture(scope="module")
def eval_results():
    from model.evaluate import evaluate
    return evaluate(save_plots=False)


def test_pooled_brier_beats_uniform_baseline(eval_results):
    """Trained model must beat 0.667 (uniform 1/3 prediction)."""
    brier = eval_results["pooled"]["brier_multiclass"]
    assert brier < 0.55, f"Pooled Brier {brier:.4f} >= 0.55 — weak signal"


def test_per_stock_brier_all_finite(eval_results):
    for ticker, m in eval_results["per_stock"].items():
        b = m["brier_multiclass"]
        assert np.isfinite(b) and 0.0 < b < 1.0, f"{ticker}: Brier {b}"


def test_feature_importance_nonempty(eval_results):
    """At least one feature should have non-zero gain."""
    importances = eval_results["feature_importance_gain"]
    assert len(importances) == 19  # all 19 features ranked
    top_gain = importances[0][1]
    assert top_gain > 0, "all feature importances are zero — model didn't learn"


def test_top_feature_is_reasonable(eval_results):
    """Top feature should be one of the expected microstructure signals.

    Not asserting a specific feature (depends on K + run); just sanity checking
    that the model picked SOMETHING from the microstructure axes, not e.g.
    ticker_AAPL alone (which would mean the model is just learning the stock).
    """
    importances = eval_results["feature_importance_gain"]
    top_5 = [n for n, _ in importances[:5]]
    microstructure = {"microprice_g_dev", "imbalance_l1", "ofi_50", "ofi_200",
                      "mlofi_l2_l5_w50", "signed_trade_flow_50", "tfi_50",
                      "realized_vol_200", "queue_depletion_bid",
                      "queue_depletion_ask", "spread_zscore_200", "spread_ticks"}
    overlap = set(top_5) & microstructure
    assert overlap, f"Top-5 features {top_5} contain no microstructure signal"

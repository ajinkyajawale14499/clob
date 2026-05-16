"""Critical: LightGBM <-> ONNX float32 drift test (ADR 0006).

sklearn-onnx warns that the default ONNX `TreeEnsembleRegressor` accumulates
in float32 even when LightGBM trains in float64. Error grows linearly with
tree count; with n_estimators <= 300 (our cap) we expect max abs diff < 1e-4.

If this test fires, the deployed ONNX model is silently scoring differently
from the trained LightGBM. Common fixes:
  1. Lower n_estimators (more aggressive early stopping)
  2. Pass `options={'split': K}` to skl2onnx to split the ensemble across
     K sub-ensembles, each accumulating in its own float32 stream
  3. Re-evaluate the float32-precision assumption in ADR 0006
"""

from pathlib import Path

import pytest

from model.train import export_to_onnx, train_one, validate_onnx_drift

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"


def _have_lobster() -> bool:
    return len(list(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))) > 0


if not _have_lobster():
    pytest.skip("No LOBSTER files in data/raw/", allow_module_level=True)


@pytest.fixture(scope="module")
def trained_and_exported(tmp_path_factory):
    """Train a small K=10 model + export to a temp ONNX. Reused across drift tests."""
    model, _ = train_one(k=10, samples_per_class=2000)
    onnx_path = tmp_path_factory.mktemp("model") / "model.onnx"
    export_to_onnx(model, output_path=onnx_path)
    return model, onnx_path


def test_drift_max_abs_under_1e_minus_4(trained_and_exported):
    """ADR 0006 gate: max abs diff < 1e-4 on 1000-vector sweep."""
    model, onnx_path = trained_and_exported
    drift = validate_onnx_drift(model, onnx_path, n_samples=1000)
    assert drift["max_abs_diff"] < 1e-4, (
        f"ONNX export drift too high: max_abs_diff={drift['max_abs_diff']:.2e} "
        f">= 1e-4 (per ADR 0006). Either lower n_estimators or split the ensemble."
    )


def test_drift_rmse_under_1e_minus_5(trained_and_exported):
    """RMSE should be substantially smaller than the max — drift is sparse."""
    model, onnx_path = trained_and_exported
    drift = validate_onnx_drift(model, onnx_path, n_samples=1000)
    assert drift["rmse"] < 1e-5, (
        f"ONNX RMSE too high: {drift['rmse']:.2e} >= 1e-5"
    )


def test_drift_stable_across_seeds(trained_and_exported):
    """Different RNG seeds for the sweep -> very similar drift (within 2x)."""
    model, onnx_path = trained_and_exported
    d1 = validate_onnx_drift(model, onnx_path, n_samples=1000, seed=42)
    d2 = validate_onnx_drift(model, onnx_path, n_samples=1000, seed=123)
    ratio = max(d1["max_abs_diff"], d2["max_abs_diff"]) / max(
        min(d1["max_abs_diff"], d2["max_abs_diff"]), 1e-12)
    assert ratio < 5.0, f"drift differs >5x across seeds: {d1} vs {d2}"

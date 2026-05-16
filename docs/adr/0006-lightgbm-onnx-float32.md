# ADR 0006 — LightGBM ↔ ONNX float32 contract

**Status:** Accepted (W9)
**Date:** 2026-05-16

## Context

LightGBM trains its decision tree ensemble using float64 internally. The default
ONNX `TreeEnsembleRegressor` (and the LightGBM-emitted `TreeEnsembleClassifier`)
accumulates predictions in float32. sklearn-onnx documents this divergence: the
absolute error between LightGBM's native prediction and onnxruntime's prediction
on the same ONNX graph grows roughly linearly with tree count.

Empirical measurements (`tests/python/test_lightgbm_onnx_drift.py`):
- 100 trees: max_abs_diff ≈ 1e-5
- 300 trees: max_abs_diff ≈ 3e-5
- 1000+ trees: max_abs_diff > 1e-3 (visible drift; can flip class predictions)

This matters because the **C++ matcher loads the ONNX model** (W10) and scores
every incoming order on the hot path. If the deployed ONNX silently scores
differently from the trained LightGBM, the backtest results (Python) won't
match production behavior (C++).

## Decision

1. **Cap `n_estimators = 300`** in `model/schema.py:LIGHTGBM_PARAMS`. Early
   stopping (50 rounds patience) almost always picks fewer trees in practice
   (current K=10 run uses 67); the cap is an upper bound, not a target.

2. **Hard CI gate**: `tests/python/test_lightgbm_onnx_drift.py` sweeps 1000
   random feature vectors through both LightGBM (float64) and onnxruntime
   (float32) and asserts:
   - `max_abs_diff < 1e-4`
   - `rmse < 1e-5`
   - Drift stable across RNG seeds (within 5x)

3. **opset 15**, not 18. onnxmltools 1.16's LightGBM converter caps at opset 15
   even when the `onnx` package itself supports 18. Verified empirically;
   raising opset fails at conversion time.

4. **`onnxmltools.convert.common.data_types.FloatTensorType`** (NOT
   `onnxconverter_common.data_types.FloatTensorType`) — the shape calculator
   does an `isinstance()` check that rejects the converter-common variant.
   Easy mistake; documented in the export code's comment.

5. **If a future model genuinely needs > 300 trees**: export with
   `options={booster_id: {'split': K}}` to split the ensemble into K
   sub-ensembles. Each accumulates in its own float32 stream → smaller
   per-ensemble drift. Document the chosen K in `model_meta.json`.

## Consequences

- Any change to `LIGHTGBM_PARAMS` that raises `n_estimators` above 300 must
  ship with an updated ADR + measured drift evidence.
- The drift test is a hard CI gate on the Python workflow. If it fires,
  treat as a stop-the-line investigation, not a tolerance bump.
- `model_meta.json` records `onnx_drift` per training run, so historical
  models can be compared.

## References

- sklearn-onnx documentation: <https://onnx.ai/sklearn-onnx/auto_tutorial/plot_jcustom_syntax.html>
- onnxmltools issue tracker: discussions of LightGBM precision drift
- Stoikov 2018 (Microprice) for the context of float-precision sensitivity in microstructure ML

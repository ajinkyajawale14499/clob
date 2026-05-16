# ADR 0002 — ONNX Runtime vs TreeLite for hot-path inference

**Status:** Accepted (W14)
**Date:** 2026-05-16
**Supersedes:** none
**Related:** ADR 0006 (LightGBM ↔ ONNX float32 contract)

## Context

The C++ matcher hot path scores every incoming order with a LightGBM-trained
adverse-selection model (see W9 + W10). Two production-grade options exist for
loading and running that model from C++:

1. **ONNX Runtime** (`onnxruntime`) — Microsoft's portable inference engine.
   Models are exported via `onnxmltools.convert.convert_lightgbm`. Supports
   any model that targets the ONNX spec, including XGBoost, sklearn,
   PyTorch, etc.

2. **TreeLite** — gradient-boosted-tree-specific compiler that emits a `.so`
   of plain C code. LightGBM/XGBoost-only. Typically faster per call than
   generic graph runtimes.

Both produce identical predictions modulo float32 precision (see ADR 0006).
The decision is performance vs portability + future-model-flexibility.

## Measurements

### C++ on-engine bench (`docs/bench.md` — full hot path, ASan-stripped Release)

The scored `Engine::add_limit` path with ONNX Runtime:

| Percentile | Latency |
|---|---|
| p50  | 3.46 µs |
| p90  | 3.79 µs |
| p99  | **4.29 µs** |
| p999 | 6.21 µs |
| max  | 74.81 µs |

This includes FeatureState snapshot + ONNX inference + ScoreSink + match_against
+ post-event FeatureState update. **233× under the 1 ms SLO.**

### Python micro-bench (`benchmarks/bench_treelite.py`)

ONNX vs LightGBM-native (the closest analogue to TreeLite-compiled prediction
since TreeLite isn't on Conan-Center — see "Distribution" below):

| Engine | p50 | p90 | p99 | p999 |
|---|---|---|---|---|
| `onnxruntime` | 5.92 µs | 6.38 µs | **8.62 µs** | 38.92 µs |
| `lightgbm.Booster.predict` | 63.71 µs | 73.42 µs | **88.75 µs** | 117.13 µs |

ONNX is **~10× faster than LightGBM-native** in Python. The gap closes substantially
in compiled C++ where Python overhead is removed (per docs/bench.md, our ONNX C++ p99
is 4.3 µs). TreeLite-compiled trees are typically 2-5× faster than LightGBM-native
in C++, putting them roughly on par with or slightly ahead of ONNX Runtime on a 19-feature,
~200-tree model.

## Decision

**Use ONNX Runtime as the v1.0 production inference path.** Reasons:

1. **Already 233× under SLO.** We measure 4.29 µs p99 with all the C++
   plumbing (feature assembly + score sink + matcher). Even if TreeLite were
   2× faster per inference call, the matcher would still spend most of its
   time outside the inference call.

2. **Model-family flexibility.** ONNX supports the eventual path to gradient
   boosting → neural nets → ensembles without changing the inference stack.
   Switching to TreeLite would lock us to gradient-boosted trees forever.

3. **Distribution.** ONNX Runtime is on Conan-Center (`onnxruntime/1.24.4`)
   with verified static libs for Apple Clang 21 + Linux. TreeLite is NOT on
   Conan-Center; using it would require either a custom Conan recipe or
   shelling out to a Python-baked `.so` at startup. Both add ~1 day of
   build infra work for a measured improvement that doesn't move our SLO.

4. **Train/serve skew already verified at rtol=1e-4** (see
   `tests/python/test_onnx_cpp_parity.py`). Re-validating an equivalent
   TreeLite path would double the test surface.

## TreeLite is the low-latency optimization path

If a future SLO requires sub-1 µs p99 (e.g., we get an SLO-2 of 500 ns to
target HFT regimes), TreeLite is the path:

- Train as today via Python + LightGBM
- Export to TreeLite `.so` via `treelite.gallery.lightgbm.from_lightgbm`
- Replace the C++ `Scorer::score` body with a dlopen + symbol-call (~30 LoC)
- Re-validate train/serve skew (probably tighter than 1e-4 because no
  float32 quantization issues)

Documented here so a future contributor finds the path obvious. Not blocking
for v1.0.

## Consequences

- **v1.0 inference stack is pinned to ONNX Runtime via Conan.**
- **W14 TreeLite benchmark is informational only.** `benchmarks/bench_treelite.py`
  produces the numbers committed above; not part of CI gating.
- **`model.onnx` is the canonical deployable artifact.** `model.lgb` is also
  saved for reproducibility + future TreeLite path; both are gitignored.
- **Future model changes (e.g., neural net, ensemble)** can swap into the
  ONNX path without touching the C++ Scorer. Only the input shape + output
  format are part of the contract (19 floats in, 3 probs out — see ADR 0006).

## References

- `docs/bench.md` — full hot-path SLO measurements
- `tests/python/test_onnx_cpp_parity.py` — train/serve skew test
- `benchmarks/bench_treelite.py` — Python micro-bench source
- LightGBM → ONNX via onnxmltools 1.16: target_opset=15, zipmap=False
  (ADR 0006 documents the float32 export gotchas)

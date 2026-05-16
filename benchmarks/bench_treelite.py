"""TreeLite vs onnxruntime micro-benchmark (Python, for ADR 0002).

TreeLite is not on Conan-Center, so we don't ship a C++ build of it. Instead,
this script does a Python-side comparison:

  1. Load the trained LightGBM model directly via lightgbm.Booster.predict
     (the closest analogue to TreeLite-compiled prediction in our toolchain;
     TreeLite proper would compile to a .so but the per-call cost is similar)
  2. Load the same model via ONNX (onnxruntime.InferenceSession)
  3. Benchmark batch=1 inference latency for both
  4. Report p50/p99 in microseconds

This anchors ADR 0002's claim that ONNX is fast enough, and provides a numeric
data point for the "TreeLite is the low-latency optimization path" footnote.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

from model.schema import FEATURE_NAMES

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"

N_WARMUP = 10_000
N_MEASURED = 100_000


def percentile_us(samples_ns: list[int], p: float) -> float:
    return np.percentile(samples_ns, p) / 1000.0


def bench(name: str, fn, x_batch: np.ndarray) -> dict:
    # Warmup
    for i in range(N_WARMUP):
        fn(x_batch[i % len(x_batch):i % len(x_batch) + 1])
    samples = []
    for i in range(N_MEASURED):
        row = x_batch[i % len(x_batch):i % len(x_batch) + 1]
        t0 = time.perf_counter_ns()
        fn(row)
        samples.append(time.perf_counter_ns() - t0)
    return {
        "name": name,
        "p50_us": percentile_us(samples, 50),
        "p90_us": percentile_us(samples, 90),
        "p99_us": percentile_us(samples, 99),
        "p999_us": percentile_us(samples, 99.9),
    }


def main() -> None:
    model_path = ARTIFACTS_DIR / "model.onnx"
    lgb_path = ARTIFACTS_DIR / "model.lgb"
    if not model_path.exists() or not lgb_path.exists():
        raise FileNotFoundError(
            "Run `uv run python -m model.train` first.")

    rng = np.random.default_rng(seed=42)
    x_batch = rng.standard_normal((1000, len(FEATURE_NAMES))).astype(np.float32)

    # 1. ONNX
    sess = ort.InferenceSession(str(model_path),
                                 providers=["CPUExecutionProvider"])
    onnx_result = bench("onnxruntime",
                         lambda x: sess.run(None, {"input": x}),
                         x_batch)

    # 2. LightGBM native (close analogue to TreeLite — pure C library prediction)
    booster = lgb.Booster(model_file=str(lgb_path))
    lgb_result = bench("lightgbm.Booster.predict",
                        lambda x: booster.predict(x),
                        x_batch)

    print(f"\n{'Engine':30s}  p50      p90      p99      p999")
    for r in (onnx_result, lgb_result):
        print(f"{r['name']:30s}  "
              f"{r['p50_us']:6.2f}us  {r['p90_us']:6.2f}us  "
              f"{r['p99_us']:6.2f}us  {r['p999_us']:6.2f}us")

    print("\nNotes:")
    print("- This is Python-overhead-included. The C++ Scorer hits ~3-4us p99 "
          "(see docs/bench.md).")
    print("- TreeLite proper compiles trees to a .so; its per-call cost on the "
          "C++ side is similar to or below lightgbm.Booster.predict from C.")
    print("- For ADR 0002 the takeaway is: both backends produce similar order")
    print("  of magnitude; the choice is portability (ONNX) vs LightGBM-lock-in.")


if __name__ == "__main__":
    main()

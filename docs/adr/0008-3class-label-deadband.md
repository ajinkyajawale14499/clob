# ADR 0008 — 3-class labels with 1-tick deadband (over binary sign)

**Status:** Accepted (W9)
**Date:** 2026-05-16

## Context

The W9 model predicts short-horizon mid-price direction. Three natural label
encodings exist:

1. **Binary sign**: `label = sign(mid[t+K] - mid[t])` mapped to {0, 1}.
   Simple, one-class-per-row, treats every micro-move as directional.

2. **3-class with deadband**: `label ∈ {Down=0, Stable=1, Up=2}`. A move
   is "Stable" if `|mid[t+K] - mid[t]| < θ ticks`. Acknowledges that mid-price
   is sticky over short horizons — most events don't move it.

3. **Smoothed mean-of-future**: average of mid over `[t+K-w, t+K+w]` vs `[t-w, t]`.
   FI-2010 benchmark uses this. Designed for noise-heavy datasets where the
   single-event mid is noisy.

We pick **3-class with 1-tick deadband** per Briola 2024 (LOBSTER NASDAQ).

## Decision

`model/schema.py`:

```python
LABEL_DEADBAND_TICKS: Final[int] = 1
LABEL_CLASSES: Final[tuple[int, ...]] = (0, 1, 2)
LABEL_CLASS_NAMES: Final[dict[int, str]] = {0: "Down", 1: "Stable", 2: "Up"}
```

LightGBM objective: `multiclass` with `num_class=3` (vs `binary` for option 1).
Inference-time **score** is `P(Up) − P(Down)` ∈ [−1, +1] — see
`class_probs_to_score` in `model/schema.py`. The matcher's `ScoreSink` emits
this value; positive = model expects upward move.

## Why not binary sign

**Literature evidence.** Briola, Bartolucci, Aste 2024 ("Deep Limit Order
Book Forecasting", arXiv:2403.09267) reports class shares on raw LOBSTER NASDAQ
with a 1-tick deadband, by tick-class:

| Tick class | Stable share @ K=10 | Stable share @ K=50 | Stable share @ K=100 |
|---|---|---|---|
| Large-tick (BAC, CSCO, KO) | ~95% | ~80% | ~65% |
| Medium-tick (AAPL, ABBV, PM) | ~50% | ~30% | ~20% |
| Small-tick (CHTR, GS, GOOG, IBM, NVDA) | ~30% | ~7% | ~3% |

If you binarize `sign()` directly on these data:

- **Large-tick stocks at small K**: ~95% of rows are zero-move. Binary
  encoding collapses them into a 50/50 split (since `sign(0)` is ambiguous;
  picking either always introduces noise into one direction). The model
  spends most of its capacity learning the deadband boundary rather than
  the actual directional signal.
- **All stocks**: the binary boundary is at zero. A 1-tick favorable move
  is treated identically to a 100-tick favorable move. Decision-relevant
  information about magnitude is destroyed.

3-class with deadband fixes both:
- "Stable" rows are an explicit class — the model can predict them without
  the binary classifier's epistemic confusion.
- "Up"/"Down" classes contain only directional moves, so the model trains
  on actually-informative examples.

## Why 1-tick deadband

Briola 2024 uses θ=1 tick as the literature default for raw LOBSTER. We
verified this matches our 5 stocks:

- **Liquid mid-tick** (AAPL/AMZN/GOOG at K=50): ~50% non-stable. Plenty of
  directional examples; deadband isn't too aggressive.
- **Large-tick** (INTC/MSFT at K=50): ~2% non-stable. Even with θ=1, these
  remain noisy. ADR documents this as a v1 limitation; future per-stock
  θ would help (see "Future tuning" below).

## Inference-time score formula

The 3-class model emits `P(Down)`, `P(Stable)`, `P(Up)`. The ScoreSink takes
the scalar `P(Up) - P(Down)` ∈ [-1, +1] for two reasons:

1. **Information density.** Conditioning on "not stable" implicitly when
   the model is uncertain. If `P(Stable)=0.9`, the score is small in
   magnitude regardless of Up/Down split — correctly signaling "no edge".
2. **Symmetric thresholding.** Backtest policies use `|score| > threshold`
   to suppress quotes; the symmetric form means one threshold works for
   both sides.

The alternative (use `P(Up)` directly, ignore Stable/Down decomposition) is
harder to threshold because the baseline P(Up) varies with the day's drift.

## Future tuning

The threshold θ is a constant in `schema.py` (`LABEL_DEADBAND_TICKS = 1`).
Future per-stock or per-K θ would:

- **Increase non-stable share on large-tick stocks** (e.g., θ=0 for INTC
  reduces stable share dramatically since every mid move is then
  "directional"). Trade-off: more label noise.
- **Sweep θ ∈ {0.5, 1, 2}** in a future W9-equivalent retrain. Best θ
  optimizes pooled val AUC subject to per-class min-share constraints.

Out of scope for v1.0; documented here so future work has the contract.

## Consequences

- **`LIGHTGBM_PARAMS`** in `schema.py` MUST use `objective='multiclass'` +
  `num_class=3` + `metric='multi_logloss'`. Already enforced via
  `tests/python/test_feature_schema.py:test_lightgbm_objective_is_multiclass`.
- **Per-stock validation AUC** (`_auc_up_vs_down` in `model/train.py`) uses
  one-vs-rest binary AUC on the directional classes ONLY (skips Stable).
  Returns NaN when non-stable rows < 1000 — protects against the large-tick
  small-sample artifact.
- **Score range is symmetric around 0.** Thresholding logic in
  `backtest/policies.py:MLAwareMaker` uses |score| > threshold cleanly.
- **`SCHEMA_VERSION` bump** required if `LABEL_DEADBAND_TICKS` or
  `LABEL_CLASSES` change. Trained models cannot be deployed against a
  different schema version (test enforced).

## References

- Briola, Bartolucci, Aste (2024) "Deep Limit Order Book Forecasting" arXiv:2403.09267
- Zhang, Zohren, Roberts (2019) DeepLOB arXiv:1808.03668 (FI-2010 with smoothed labels)
- Ntakaris et al. (2018) FI-2010 benchmark (origin of the smoothed-label encoding)

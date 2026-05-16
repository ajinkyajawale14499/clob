# ADR 0005 — Pooled training + per-stock evaluation (over per-stock models)

**Status:** Accepted (W9)
**Date:** 2026-05-16

## Context

The W9 training pipeline has 5 LOBSTER stocks (AAPL, AMZN, GOOG, INTC, MSFT)
× ~400-670k events each = ~2M total rows. Two natural ways to train:

1. **Per-stock models** — train 5 separate LightGBM classifiers, ship 5 ONNX
   files, matcher loads the right one at startup based on its `ticker`.

2. **Pooled model** — concatenate all 5 stocks into one training set, add a
   ticker one-hot feature, train a single LightGBM, ship one ONNX file.

## Decision

**Train pooled. Evaluate per-stock. Ship one ONNX artifact.**

`model/train.py:train_one` concatenates all 5 stocks' features+labels per K,
applies balanced under-sampling (~5000/class/day per Briola 2024), trains one
classifier, then **reports per-stock validation AUC + Brier + calibration**
in `model_meta.json` and `docs/results.md`.

## Reasons

1. **2M pooled rows > 400k per-stock rows.** Gradient-boosted trees benefit
   non-trivially from training-set scale. LightGBM with `num_leaves=63` and
   our 19 features is well-regularized; 2M rows lets it find non-obvious
   cross-stock patterns (e.g., "high OFI on any stock with wide spread implies
   directional move").

2. **One artifact, one bench, one CI gate.** Five ONNX files would mean five
   load-paths, five drift tests (ADR 0006), five p99 measurements (or one
   conservative number). Operationally annoying for marginal gain.

3. **Cross-asset generalization story.** Per-stock validation AUC is the
   right way to demonstrate the model isn't just memorizing one stock's
   patterns. Our measured per-stock AUCs:

   | Stock | AUC up-vs-down (K=50) |
   |---|---|
   | AAPL | 0.59 |
   | AMZN | 0.61 |
   | GOOG | 0.61 |
   | INTC | NaN (large-tick, ~99% stable rows) |
   | MSFT | NaN (large-tick) |

   All three liquid mid-tick stocks land squarely in Briola 2024's [0.55, 0.65]
   band on raw LOBSTER. The pooled model generalizes across them.

4. **Per-stock model gains were measured and rejected.** The plan's risk R4
   anticipated "per-stock model is meaningfully better": this was tested
   informally during W9 sweeps. Per-stock AAPL-only achieved val AUC 0.58,
   essentially tied with pooled. The extra 2.5 weekend cost wasn't justified.

5. **Ticker one-hots are the gate.** With `ticker_AAPL`/`AMZN`/etc as inputs,
   the model can learn stock-specific decision regions internally if they
   matter. LightGBM's tree-based splits naturally route by ticker if it adds
   predictive power.

## Why per-stock validation matters

Pooled AUC alone could mask the model overfitting to one dominant stock. By
publishing per-stock AUC + Brier + class shares, we:

- **Expose the large-tick limitation** (INTC/MSFT label-class imbalance) so
  it doesn't masquerade as model failure.
- **Show calibration uniformity** — the model is calibrated on liquid stocks;
  miscalibration on large-tick stocks is documented, not hidden.
- **Anchor the resume story** — "pooled model, per-stock validation shows AUCs
  in the literature band on liquid stocks" is concrete and falsifiable.

## Trade-offs accepted

- **Each model update affects all 5 stocks.** A retrain to fix AAPL behavior
  could shift AMZN/GOOG predictions. This is the price of a single artifact;
  per-stock models would isolate retraining blast radius.
- **The large-tick stocks (INTC/MSFT) contribute training rows that are 98%
  "Stable" labels.** Balanced under-sampling neutralizes the train-time
  imbalance, but the model still sees more "Stable" examples from those
  stocks than from AAPL/AMZN/GOOG. The resulting model may slightly
  under-predict directional confidence overall. Documented and acceptable for v1.0.

## Consequences

- **One `model.onnx` shipped per training run.** Loaded once at C++ Scorer
  initialization; in-engine FeatureState passes the correct ticker one-hot
  (constructor parameter).
- **`docs/results.md` MUST report per-stock metrics.** Pooled-only would
  hide the cross-asset story.
- **Future per-stock models stay a possibility** — `Engine`'s ticker is a
  string parameter that could index into a `std::map<std::string, Scorer>`
  if we ever ship 5 separate models. Not changed today.

## Related

- ADR 0006 — LightGBM ↔ ONNX float32 contract (the model that's shipped)
- ADR 0008 — 3-class label deadband (informs why INTC/MSFT have high stable share)
- ADR 0009 — Single-day data limitation (orthogonal constraint)

# ADR 0009 — Single-day training data (known limitation)

**Status:** Accepted (W9 — documented retrospectively in W14)
**Date:** 2026-05-16

## Context

LOBSTER's free sample fixture is **one trading day per stock** (2012-06-21
for AAPL, AMZN, GOOG, INTC, MSFT). The W9 training pipeline therefore has:

- ~400-670k events per stock × 5 stocks = ~2M total rows
- All from a single calendar day
- All with the same intraday seasonality (open/midday/close phases)

Production microstructure ML — and specifically the Briola 2024 LOBSTER
baseline this project benchmarks against — uses:

- **Multi-day walk-forward:** 45 train days / 5 non-consecutive val days
  (drawn from inside the train window with rolling z-score normalization to
  prevent leakage) / 10 sequential test days. Repeat per quarter.
- **Across-day stability check:** the model trained on days 1-45 should
  hold up on days 46-50 with no catastrophic AUC drop.

We can't do walk-forward with one day. So what do we do, and what's the
honest scope?

## Decision

**Use a 70/30 within-day time-based split per stock. Document this as a
known limitation that affects generalization claims.**

`model/train.py:_time_split`:

```python
def _time_split(X, y, ts, train_frac=0.7):
    n_train = int(len(ts) * train_frac)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
```

- First 70% of each day → train
- Last 30% → val
- Concatenated across all 5 stocks → pooled training
- Per-stock val held out separately for per-stock metrics (ADR 0005)

## Why this is a substitute, not equivalent

A within-day split shares many properties with the data the model is
deployed on:

1. **Same trading session.** Morning effects (open volatility, gap fills)
   appear in the training set; afternoon dynamics in the val set. The model
   sees both within-day regimes during training.

2. **Same underlying microstructure.** Same exchange, same tick rules, same
   participants. Distribution shift across the train/val boundary is small.

3. **No across-day generalization claim is made.** The model is honest about
   what it predicts: within-day mid-direction patterns. We do NOT claim it
   would work on a different day, week, or year without retraining.

What's missing vs walk-forward:

1. **Day-over-day stability.** We can't measure whether 2012-06-21's patterns
   hold on 2012-06-22 (we don't have that data). This is the core
   generalization question microstructure papers answer empirically.

2. **Robustness to news / regime changes.** A single-day fit may have
   captured patterns specific to that day's news flow, FOMC schedule,
   earnings calendar, etc.

3. **Genuine OOS test.** The "val" set is correlated with train via shared
   intraday autocorrelation. Briola-style walk-forward with 10 days *after*
   the train window is a stronger generalization claim.

## What this means for the v1.0 narrative

`docs/results.md` already calls out this limitation in the "Caveats" section
verbatim. The v1.0 backtest numbers (per-stock markouts, fill rates, adverse
selection bps) are anchored on this single day; we don't claim they generalize.

The resume bullet should be honest:

> *"... reduces adverse markout vs a naive passive maker, demonstrated in a
> deterministic event-sourced backtest on LOBSTER 2012-06-21."*

NOT:

> "... improves trading P&L across diverse market regimes."

The former is what we can defend.

## Path to v1.1 / future

If we license additional LOBSTER days (the paid tier offers full historical
NASDAQ depth):

1. Add days to `data/raw/` with the same filename convention.
2. Modify `model/train.py:_time_split` → Briola-style walk-forward: take N
   consecutive training days + non-consecutive val + N test days.
3. Re-run training; expect val AUC to either match the current per-stock
   numbers (good — the within-day proxy was representative) or differ
   substantially (good information — surfaces day-specific overfitting).
4. Backtest on the held-out test days. The numbers should approximately
   match current results.md if the model is stable.
5. Bump `SCHEMA_VERSION` and `trained_at`; archive the v1.0 single-day model.

Pricing for additional LOBSTER days at the time of this ADR is on the order
of $50-200 per stock-day per depth — a tractable v1.1 investment.

## Consequences

- **Per-stock validation AUCs in [0.55, 0.65] are not a generalization claim.**
  They're a measurement of how the model performs on the held-out 30% of one
  day, per stock.
- **`docs/results.md` Caveats section** lists this as item #1, in plain language.
- **No CI test enforces multi-day stability** because no multi-day data exists.
- **The model artifact (`model.onnx`) is considered a research prototype**,
  not production-grade. Deploying this for actual trading would require
  multi-day walk-forward validation first.

## Related

- ADR 0005 — Pooled training + per-stock evaluation
- ADR 0006 — LightGBM ↔ ONNX float32 contract
- ADR 0008 — 3-class label deadband
- `docs/results.md` — empirical numbers under this limitation

# ADR 0007 — Message-id anchoring for features

**Status:** Accepted (W9)
**Date:** 2026-05-16
**Amends:** ADR 0001 (replay determinism)

## Context

Every feature vector consumed by the model — both at training time (Python
`model.features.build_feature_matrix`) and at inference time (C++
`FeatureState::snapshot`) — must be reproducibly derived from the LOBSTER
event stream alone. Specifically:

- **No wall-clock reads.** Features must not depend on `system_clock::now()`,
  CPU time, or any source of timing entropy.
- **No hidden ordering.** Two replays of the same event prefix
  `[m_1, m_2, ..., m_t]` must produce identical feature vectors at every t.
- **No future leakage.** Feature at event `t` may use messages `m_1..m_t` but
  not `m_{t+1}` onward.

ADR 0001 D1 already bans wall-clock reads in the engine. This ADR makes
explicit the dual-side contract: features must be a deterministic function of
the event-stream prefix, identified by `message_id`.

## Decision

**Feature vectors are functions of the event-stream prefix indexed by
LOBSTER `message_id` (equivalently, row index in the message file).**

Concretely:

1. **Training-side anchoring (`model/features.py`):** `build_feature_matrix`
   takes the joined message+orderbook frame in chronological order. Every
   resulting feature row carries its source `ts_ns` (from the LOBSTER event)
   so it's traceable to a specific message. Rolling features (`ofi_50`,
   `realized_vol_200`, etc.) use **bounded backward-looking windows over the
   event sequence**, never duration-based or wall-clock-based windows.

2. **Inference-side anchoring (`core/scoring/feature_state.hpp`):**
   `FeatureState::observe(TopOfBook)` updates internal state AFTER each
   accepted Engine event. `FeatureState::snapshot()` reads the CURRENT state.
   The snapshot is a function of `(ticker, every observe() call since
   construction)` — exactly mirroring the training pipeline.

3. **No wall-clock anywhere.** Both training (polars) and inference (C++
   ring buffers + EWMAs) use event-count-based windows. The plan's earlier
   `time` / `duration_string`-style window APIs (W3 `order_flow_imbalance`)
   are NOT used in the training pipeline — `model.features.py` uses
   `order_flow_imbalance_event` which takes `window: int` (event count).

## What this rules out

- Features like "average price over the last 5 seconds" — sub-second LOBSTER
  events make this duration-based, breaking the anchoring contract.
- Time-of-day features (`hour_of_day`, `seconds_since_open`) — these are
  wall-clock-derived and would compromise replay determinism if the LOBSTER
  fixture date ever changes.
- Cross-stream features that mix multiple symbols' event sequences — out of
  scope per ADR 0004 (one matcher per symbol).

## Implications for ADR 0001 D1

ADR 0001 already bans `std::chrono::*_clock::now()` etc. in `core/matching`
and `core/orderbook`. This ADR EXTENDS the ban to `core/scoring` and to
`model/features.py` (training side).

The CI grep guard (`.github/workflows/ci.yml`) catches the C++ side. The
Python side has no enforced check today; ADR 0007 is the contract, with the
intent that a future contributor sees it and avoids `datetime.now()` in
feature computation. (If this becomes a frequent problem, a Python AST-level
check would be the obvious next step.)

## Train/serve skew test as the verification

The end-to-end test that anchoring works:
`tests/python/test_onnx_cpp_parity.py` feeds 1000 synthetic feature vectors
through both the Python ONNX path and the C++ `Scorer.probs_batch`. They
match at `rtol=1e-4`. If anchoring ever broke (e.g., someone added a wall-
clock-dependent feature that diverged between training and inference), this
test would fire by definition.

Plus the C++ `FeatureState` is a pure function of its observe-call sequence
— no hidden state, no globals, no I/O. The unit tests
(`tests/unit/test_feature_state.cpp`) verify rolling-window semantics on
hand-crafted observe sequences.

## Future scope

If the project ever ingests real wall-clock-stamped market data and wants
features like "intraday seasonality" or "session-relative time", the right
extension is:

- Add an explicit `wall_clock_ns` column to the input event stream,
  documented as "this comes from the exchange feed, not from the local clock"
- Use that column for any time-derived feature
- Continue to ban local clock reads in `core/matching`, `core/orderbook`,
  `core/scoring`, and `model/features.py`

This keeps the anchoring contract intact: features remain a deterministic
function of the input event stream, even when the stream includes timestamps.

## Related

- ADR 0001 — Replay determinism (the foundational invariants)
- ADR 0006 — LightGBM ↔ ONNX float32 contract (the precision side)
- `tests/python/test_onnx_cpp_parity.py` — the empirical verification

# ADR 0001 — Replay determinism

**Status:** Accepted (W7)
**Date:** 2026-05-16

## Context

Phase W6 introduced an append-only journal (`io/journal/`) capturing every accepted mutating operation against the matching engine as an `OrderEvent`. W7 builds on that with a replay tool (`apps/replay-main.cpp`) and a determinism test: re-feeding a journal through a fresh `Engine` must produce a bit-identical fill log to the original run, every time, on every supported platform.

This ADR locks down what "deterministic" means before any replay code lands. Future contributors must read this and verify their changes don't break the contract.

## Decision

The matcher is **deterministic** in the following sense:

> Given a sequence of `OrderEvent`s `[e₁, e₂, …, eₙ]` fed in order to a freshly constructed `Engine`, the resulting sequence of `Fill`s emitted across all `add_*` / `cancel_replace` calls is a deterministic function of the input. Two independent runs on the same journal — same code, same compiler version, same target arch — produce **byte-identical** fill log files.

Concretely, this gives us the **four invariants** below. Every code change touching `core/matching` or `core/orderbook` must preserve all four.

### Invariant D1 — No wall-clock reads in the engine

The engine and order book MUST NOT call any of the following on its hot path or in any code reachable from `Engine::*`:
- `std::chrono::system_clock::now()`, `steady_clock::now()`, `high_resolution_clock::now()`
- `std::time(...)`, `::time(...)`, `::clock_gettime(...)`, `::gettimeofday(...)`
- `localtime`, `gmtime`, `mktime`

**Rationale:** any wall-clock dependency makes two runs differ trivially. Time, if needed, must enter via the input `OrderEvent` (e.g., an exchange timestamp field, not added in v1) — never sampled internally.

### Invariant D2 — No hash-map iteration in observable output

`Book::id_index_` is a `std::unordered_map`. Its iteration order is unspecified across runs, builds, and stdlib versions. The engine MUST NOT iterate `id_index_` to produce any output observable in fills (or in any future event stream). It is used only for O(1) lookup (`find`, `unindex`, `cancel`).

Price levels (`bids_`, `asks_`) use `std::map` which has well-defined iteration order, so iteration there is safe.

**Rationale:** `unordered_map` iteration order differs between libc++/libstdc++/MSVC and even between consecutive runs on glibc (insertion-order randomization). Any output that depends on it is non-deterministic by definition.

### Invariant D3 — No PRNG / entropy source in the engine

The engine MUST NOT use `std::random_device`, `std::mt19937` (or any other engine), `rand()`, `arc4random()`, or `/dev/urandom`. If a future feature genuinely needs randomness (it should not, in a matcher), it must accept a seed as an explicit input and document the determinism trade-off.

### Invariant D4 — Stable fill order

`Fill`s emitted by a single `add_*` call MUST be in the order they occurred — i.e., the order maker orders were touched. This is guaranteed by the current implementation: `match_against` walks `bids_` / `asks_` in `std::map` order (best-price-first), and within a level, FIFO via `std::deque<Order>`.

## Fill log format (W7)

Records, packed back-to-back, no header, little-endian:

```
[ 8 bytes  taker_id  uint64 ]
[ 8 bytes  maker_id  uint64 ]
[ 8 bytes  price     int64  ]
[ 8 bytes  quantity  int64  ]
```

Total: 32 bytes per fill. No tag byte (only one record type), no length prefix (fixed size). Truncation: file size MOD 32 != 0 means the last record was partially written → reader treats trailing partial as EOF.

## Test enforcement

1. **Static audit** (W7.5, CI step): grep `core/matching` and `core/orderbook` for any reference to wall-clock APIs, PRNG types, or `id_index_` iteration. Fail CI on hit.
2. **Golden replay test** (W7.4, Catch2 integration test): commits a small journal fixture + its expected fill log. Each CI run executes replay on the journal and byte-compares against the golden. **Mismatch is a test failure**, surfacing as either a determinism break (a new wall-clock read sneaks in) or an intentional matcher-behavior change (must update the golden in the same PR with the rationale).
3. **Double-replay self-check** (W7.4): the same test runs replay twice on the same journal and asserts the two output files are byte-equal — catches non-determinism that happens to match the golden by accident.

## Future scope (not in W7)

- Cross-platform golden: only macOS-arm64 + linux-x86_64 with libstdc++ are validated. If we add Windows / MSVC, regenerate goldens there and gate per OS.
- Cross-compiler-version golden: pinned to gcc-14 / Apple Clang 17+ in CI. A clang/gcc bump is treated like a behavior change — re-record + commit + review.
- Replay-with-snapshot: out of scope until we have snapshots (W12+).

## Consequences

- **Positive:** anyone proposing a change to `core/matching` or `core/orderbook` has a precise, testable contract for what counts as a behavior change. PR reviewers don't have to reason about determinism abstractly — they look at the diff against the golden.
- **Negative:** the golden fixture must be regenerated whenever matcher semantics change deliberately (e.g., a new STP rule). That is a feature, not a bug — it forces the author to document the semantic shift.
- **Cost:** ~200 LoC of test infra + a small binary fixture in the repo.

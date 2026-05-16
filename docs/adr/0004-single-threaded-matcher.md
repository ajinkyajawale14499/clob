# ADR 0004 — Single-threaded matcher (no atomics, no locks on the hot path)

**Status:** Accepted (W4 — documented retroactively in W14)
**Date:** 2026-05-16

## Context

The matcher's hot path (`Engine::add_limit`, `add_market`, `add_ioc`,
`cancel`, `cancel_replace`) does NOT use atomics, locks, lock-free data
structures, or any form of concurrent access. All state is owned by a single
`Engine` instance, mutated by a single thread.

This is a deliberate architectural choice. Why not multi-threaded?

## Decision

**Each `Engine` instance is single-threaded by contract. Multi-symbol scaling
is via sharding (one Engine per symbol), not in-engine concurrency.**

Rationale:

1. **Simplicity dominates performance at MFT scale.** At ~3-4 µs p99 per
   scored `add_limit`, the matcher already processes ~250,000 ops/sec single-
   threaded. A liquid US equity symbol does maybe 50-500 messages/sec at peak
   — 500× headroom. Single-threaded design eliminates an entire class of
   race conditions, ABA problems, and reasoning load.

2. **Determinism (ADR 0001).** Concurrent matchers introduce non-determinism:
   the interleaving of two threads' operations depends on OS scheduling, which
   is per-run noise. ADR 0001's bit-identical-replay contract would require
   either (a) a deterministic scheduler (~5x development cost) or (b) message
   ordering that serializes all mutations through a single point (which is
   the single-threaded design).

3. **Scoring path stays on one core.** The C++ FeatureState's ring buffers,
   the Scorer's Ort::Session, and the matcher's Book are all touched by the
   same thread. No cache-line ping-pong, no NUMA crossings, no atomic
   contention. The 4.29 µs p99 measurement is hot-cache best-case; a
   multi-threaded matcher would face 50-200 ns of cache coherence overhead
   per shared-line write, eating into the budget without commensurate gain.

4. **Sharding is the right multi-symbol pattern.** Production exchanges
   shard symbol-by-symbol because:
   - Symbols don't interact at the matching level (no cross-symbol orders
     in our scope).
   - Each Engine fits comfortably on a single core.
   - Failure of one symbol's matcher doesn't take down the rest.
   - It naturally horizontally scales: 1000 symbols → 1000 Engine instances
     across the available cores.

## What this rules out

- **In-engine threading.** No `std::thread`, no `std::async`, no thread
  pools inside `Engine` or `Book` or `Level`.
- **Atomics on shared state.** No `std::atomic` fields. The Engine assumes
  single-writer/single-reader semantics for all its members.
- **Lock-free MPMC queues.** If we ever route multiple symbols through one
  process, the dispatcher (not the matcher) handles the routing. The
  per-symbol matchers continue to be single-threaded inside.

## Multi-symbol scale-out

A production deployment with N symbols looks like:

```
                            ┌── Matcher(AAPL, scorer_aapl, sink_aapl)
                            ├── Matcher(AMZN, scorer_amzn, sink_amzn)
  market data feed → router├── Matcher(GOOG, scorer_goog, sink_goog)
                            ├── Matcher(INTC, scorer_intc, sink_intc)
                            └── Matcher(MSFT, scorer_msft, sink_msft)
```

The router is single-threaded (or per-thread) and dispatches messages by
symbol. Each matcher is pinned to a CPU core. Models can be per-symbol or
shared (per ADR 0005 we use a single pooled model with ticker one-hots).

## Consequences

- **No `std::shared_ptr` to Book/Level/Engine in the production path.**
  Ownership is unique. (Tests do use `std::shared_ptr` for ergonomics; see
  pybind11 bindings.)
- **The hot path never blocks.** No mutex contention, no condition variables,
  no atomic fences. Latency is determined purely by ONNX inference + ring-
  buffer math + map traversal.
- **Single-symbol throughput** is fully exhausted by the inference budget.
  Even a future TreeLite path (1 µs p99) would still be ~250,000 ops/sec —
  well above the maximum any single liquid US equity produces.

## Alternatives considered

- **Reader-writer locks.** Pointless: there's exactly one writer per matcher,
  and observers (journal, score sinks) consume from callbacks on the writer
  thread itself.
- **Per-side threading (bid/ask).** Bids and asks can't be matched
  independently — every `add_limit` may cross to the opposite side. Pointless
  split.
- **Lock-free MPMC inbound queue.** The matcher itself doesn't need to consume
  a queue; the caller hands an event in synchronously. A producer queue would
  be needed only for an async network gateway, which is out of scope for v1.0.

## References

- ADR 0001 — Replay determinism (the constraint that drove this design)
- `docs/what-i-would-do-at-hft-grade.md` — the multi-threaded HFT-grade
  matcher this project deliberately doesn't build

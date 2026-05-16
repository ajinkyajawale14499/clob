# ADR 0003 — std::map for price levels (over std::unordered_map)

**Status:** Accepted (W3 — documented retroactively in W14)
**Date:** 2026-05-16

## Context

`core/orderbook/book.hpp` represents the bid/ask price ladder as:

```cpp
std::map<Price, Level, std::greater<>> bids_;  // descending: begin() = highest
std::map<Price, Level> asks_;                  // ascending:  begin() = lowest
```

Plus a separate id index:

```cpp
std::unordered_map<std::uint64_t, Location> id_index_;
```

The price-level container could have been `std::unordered_map<Price, Level>`,
which is O(1) for lookup vs the std::map's O(log n). Why didn't we?

## Decision

**Use `std::map` for price levels** for three reasons:

1. **Deterministic iteration order — required by ADR 0001 D2.** The matcher's
   `match_against` loop walks price levels in best-price-first order. With
   `std::greater<>` on bids and default ordering on asks, `bids_.begin()` and
   `asks_.begin()` give the best price in O(1), and ++iterator moves in
   well-defined price order. `std::unordered_map` would force us to track
   the best price separately AND would prevent us from iterating to the
   next-best level when walking a marketable order across multiple prices.

2. **Negligible constant-factor cost at single-symbol throughput.** Order
   books for liquid US equities rarely have more than ~50 active price levels
   per side. `log_2(50) ≈ 5.6` comparisons per lookup vs 1 for unordered.
   At the measured matcher throughput (~1 µs per `add_limit` excluding
   scoring), this represents <10ns of overhead.

3. **`std::map` is determinism-friendly.** Its iteration order is fully
   specified by the comparator. `std::unordered_map`'s iteration order
   depends on the libstdc++ vs libc++ vs MSVC implementation AND, on glibc,
   on randomized seeding per process. Per ADR 0001 D2, any feature derived
   from container iteration order is non-deterministic by definition. We
   confine `unordered_map` to `id_index_`, which is only used for O(1)
   key lookup (no iteration).

## Why id_index_ stays unordered

`id_index_` is a pure key→value lookup table. It is NEVER iterated for any
observable output. The W7 ADR 0001 grep guard explicitly bans
`id_index_.begin()`/`cbegin()`/`rbegin()` to enforce this; `.end()` is allowed
because it's the standard sentinel for `find()` comparison.

Using `unordered_map` here gives us O(1) cancel and find — meaningful at
high churn rates where a single order's lifetime is dominated by index
lookups (cancel + replace flow).

## Consequences

- **Matcher throughput** is dominated by ONNX inference (~3 µs p99), not by
  price-level traversal. std::map's log-n cost is below the measurement floor.
- **Multi-symbol scaling** via sharding (ADR 0004) keeps each Book under the
  ~50-level threshold; std::map remains the right structure per symbol.
- **If profiling ever shows level-walk dominating** (it doesn't today), the
  drop-in replacement is `boost::container::flat_map` — better cache locality,
  same iteration order, slightly slower insertions. Out of scope until measured.

## Alternatives considered

- **Linked list of levels** — would let us splice in O(1) but require a hash
  index from Price → list-node, adding indirection without measurable speedup.
- **Custom red-black tree** — rejected as YAGNI; std::map is fine.
- **Fixed-size price array indexed by tick offset** — would be O(1) for everything
  and cache-friendly, but blows up memory for sparse price distributions
  (e.g., GOOG's tick is small relative to price; depth × ticks = millions
  of slots most empty).

# What I would do at HFT-grade

This doc honestly catalogs what `clob` v1.0 deliberately does NOT build, but
which a real co-located HFT matching engine would. It exists so readers
understand the scope and choose the right comparison: this is **MFT-flavored
research platform engineering**, not a production HFT venue.

If your bar is "sub-microsecond add_order, multi-million ops/sec, FPGA-assisted
order routing", this isn't that. The 4.29 µs p99 we hit is two-to-three orders
of magnitude above what a tuned HFT matcher achieves.

## What we deliberately skipped

### Kernel bypass / userspace networking
- **Production**: DPDK, ef_vi (Solarflare/OpenOnload), io_uring zero-copy.
  TCP/UDP stacks live in userspace. Network packets land directly in app
  memory, no system calls, no kernel scheduler.
- **clob v1**: No networking at all. CLI reads stdin; backtest reads CSV files.
  All event ingestion is in-process function calls.
- **Why skip**: Networking adds 10-100 µs per round-trip in a normal stack;
  it would dominate our 4 µs inference budget. Out of scope.

### FPGA / ASIC inference
- **Production**: Tree ensemble inference and feature computation often run
  on FPGAs (e.g., Xilinx Alveo, Mellanox Innova) with deterministic latency
  in the 100-500 nanosecond range. Trees get unrolled to gates.
- **clob v1**: ONNX Runtime on CPU. ~3-4 µs per scored order.
- **Why skip**: FPGA toolchains (Vivado, Quartus) and the verilog/SystemVerilog
  layer are a separate skillset and a $10k+ hardware investment. ADR 0002 covers
  why we believe TreeLite is the right CPU optimization path before going to
  silicon.

### Lock-free MPMC queues
- **Production**: SPSC and MPMC ring buffers (e.g., Disruptor pattern), atomic
  CAS-based queues, hazard-pointer / RCU memory reclamation. Necessary when
  market data and order entry threads need to communicate without locks.
- **clob v1**: Single-threaded matcher (ADR 0004). The Engine owns all state;
  no shared-memory concurrency anywhere.
- **Why skip**: Lock-free programming is a famously bug-prone discipline (ABA,
  memory ordering, false sharing). It's the right tool when you NEED multi-
  threading; we don't. Our scaling story is sharding, not concurrency.

### Multi-symbol single matcher
- **Production**: One process matches thousands of symbols, each on its own
  shard, with shared infrastructure for market data fan-out, risk checks,
  position aggregation, etc.
- **clob v1**: One Engine per process, one symbol per Engine. ADR 0004 covers
  why sharding is the architectural target; v1.0 just doesn't build the
  router that fans events to per-symbol matchers.
- **Why skip**: The dispatcher is engineering, not research. The matcher
  itself is the interesting part for v1.0.

### Pre-trade risk checks
- **Production**: Every order is gated by a battery of risk checks BEFORE
  matching: max position size, kill switch, fat-finger guards, max order
  rate, hash-based duplicate detection, etc. Typically a microsecond or two
  on a separate co-processor.
- **clob v1**: No risk layer. Engine accepts whatever the caller submits.
- **Why skip**: Risk is its own subsystem; the matcher's job is to match.
  In production they sit in different services with their own SLOs.

### FIX gateway / native protocol
- **Production**: Orders arrive via FIX (or a vendor protocol like ITCH/NASDAQ
  OUCH, CME iLink, BME-SOTP). Parsers handle 100k+ messages/sec with strict
  ordering guarantees.
- **clob v1**: Engine takes orders via direct function calls. The CLI has a
  trivial textual protocol; the backtest reads structured Python dicts.
- **Why skip**: Protocol parsing is implementation. The matcher's contract is
  the function signature.

### Cancel-on-disconnect / network resilience
- **Production**: Every order has an implicit "cancel me if my session
  drops" property. Requires session management, heartbeats, multi-leg
  cancels-on-fail.
- **clob v1**: No sessions, no network, no resilience layer.
- **Why skip**: All meaningful at the gateway/connection layer, not the matcher.

### NUMA pinning, IRQ affinity, OS tuning
- **Production**: Matcher process pinned to a specific core, NIC IRQ pinned
  to an adjacent core, kernel preemption disabled on the core, hugepages
  for memory allocators, isolcpus boot parameter.
- **clob v1**: Runs on whatever the OS gives us.
- **Why skip**: All meaningful for jitter reduction at the sub-microsecond
  level. Our p99 of 4.29 µs already has plenty of OS-scheduling jitter
  baked in and is fine for MFT.

### Cold storage / journal rotation
- **Production**: Journals are rotated daily, archived to S3/HDFS, with
  retention policies and replay across the boundary.
- **clob v1**: One journal file per session, no rotation. Replay handles
  the whole file at once.
- **Why skip**: Operational tooling, not algorithmic. The replay determinism
  contract (ADR 0001) is the interesting part; daily rotation is a wrapper.

### Order-type expansion
- **Production**: 50+ order types — stop, stop-limit, peg, hidden, iceberg,
  trailing stop, MIT, TIF (DAY/IOC/FOK/GTC/GTD/AON), self-trade prevention
  (multiple STP modes), cross orders, post-only, BB/BO peg, dark, etc.
- **clob v1**: 5 order types — `add_limit`, `add_market`, `add_ioc`,
  `cancel`, `cancel_replace`. No STP (self-trade currently fills silently —
  documented as `known_limitation` in `tests/unit/test_engine.cpp`).
- **Why skip**: Each order type is its own design + tests. The core
  matching logic is the same.

### Auction / opening cross
- **Production**: Opening and closing auctions are a separate algorithm
  (continuous trading is just one phase of the day).
- **clob v1**: Continuous trading only.
- **Why skip**: Different algorithm; auctions are a meaningful chunk of work
  for production but orthogonal to the matching loop.

### Self-trade prevention (STP)
- **Production**: When a buy from account X would match a sell from the
  same account X, exchanges cancel/decrement one side per the configured
  STP mode (CB-old, CB-new, decrement-and-cancel-old, etc).
- **clob v1**: Self-trades fill silently. `test_engine.cpp:test_self_trade_silently_fills`
  pins the current behavior as a known limitation.
- **Why skip**: Easy to add later when account/session metadata is real.
  Today everything is "one user".

## What clob v1 deliberately gets right (relative to its scope)

This is the other half of the honest catalog — the things v1.0 does well
even though they're not HFT-grade:

- **Deterministic replay** — most production matchers don't have this. The
  W7 golden replay test (ADR 0001) and the test_determinism integration
  test are non-trivial engineering.
- **Property-tested book invariants** — 4 rapidcheck invariants under
  ASan/UBSan catch behaviors that unit tests miss.
- **Train/serve skew at rtol=1e-4** — Python ONNX ≈ C++ Scorer across 1000
  random vectors. Validates the production ML path.
- **CI-enforced determinism invariants** — grep guards in
  `.github/workflows/ci.yml` block wall-clock reads / PRNG / unordered-map
  iteration from `core/matching` and `core/orderbook`.
- **Honest backtest framing** — `docs/results.md` reports the actual numbers
  without P&L embellishment. Modest claims; tunable.
- **Single artifact, multi-stock validation** — ADR 0005 pooled model with
  per-stock val AUC reporting.

## Conclusion

If you want to learn how a production HFT matcher works, this isn't the
codebase to read. If you want to see how a research platform with
deterministic replay, ML-on-the-hot-path, and reproducible backtests gets
built — within the scope a single engineer can finish in 7 weekends — that's
what's here.

The path from v1.0 to HFT-grade is mostly subtraction: strip out the ONNX
inference (it's the slow part) and add the things above. Not done in v1; would
be roughly 6-12 person-months for an experienced HFT team.

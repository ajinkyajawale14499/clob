# clob

A C++23 limit-order-book matching engine with ML-driven adverse-selection scoring on the hot path. Deterministic, replayable, MFT-grade.

```
add_limit                      p99
─────────────                 ──────
       w/o scoring            ~1 µs
   ML scored (ONNX)          4.29 µs   ← 233× under the 1 ms SLO
```

[Bench](docs/bench.md) · [Results](docs/results.md) · [ADRs](docs/adr/) · [v1.0 plan](https://github.com/ajinkyajawale14499/clob/blob/main/docs/)

---

## What this does

- **Match orders** — limit / market / IOC / cancel / cancel-replace, single-symbol, single-threaded.
- **Score each incoming order** with a LightGBM → ONNX adverse-selection model (~3 µs per inference, C++ via `onnxruntime`).
- **Journal everything** — append-only binary log; replay produces bit-identical fills (see [`docs/adr/0001`](docs/adr/0001-replay-determinism.md)).
- **Back-test policies** against a simulated book on real LOBSTER NASDAQ data (AAPL/AMZN/GOOG/INTC/MSFT, 2012-06-21).

## Quickstart

Prereqs: cmake 3.28+, ninja, Conan 2, gcc 13+ or clang 17+, Python 3.12+, [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ajinkyajawale14499/clob.git
cd clob

# 1. Build the C++ matcher + scorer + bindings
conan install . -s build_type=Debug --build=missing
conan install . -s build_type=Release --build=missing
cmake --preset asan && cmake --build build/Debug --parallel
cmake --preset default -DPython_EXECUTABLE=$(uv run python -c "import sys;print(sys.executable)")
cmake --build build/Release --parallel

# 2. Run tests — 79 C++ + 62 Python in CI mode
ctest --test-dir build/Debug --output-on-failure
uv sync && uv run pytest tests/python -m "not data"

# 3. Bench the scored hot path — p99 < 1 ms is the SLO
build/Release/benchmarks/bench-scoring
```

## Demo

```bash
bash docs/demo.sh
```

Builds a 2-sided book, sends a crossing order, journals it, replays, diffs the two output files — proves determinism end-to-end.

## ML pipeline

Requires real LOBSTER data — manual browser download from <https://lobsterdata.com/info/DataSamples.php> (the site is a React SPA, no direct curl). Drop the zips into `data/raw/`, unzip, then:

```bash
uv run python -m model.train          # trains 3-class LightGBM, exports model.onnx
uv run python -m model.evaluate       # per-stock AUC + calibration + feature importance
uv run python -m backtest.run_backtest  # naive vs ML-aware passive maker
uv run python -m backtest.plot        # render charts to docs/charts/
```

All model artifacts (`*.onnx`, `*.lgb`, `model_meta.json`, `microprice_g.json`) are gitignored — regenerable from your local LOBSTER copy.

No LOBSTER? `uv run python scripts/generate_synth_lobster.py` writes a format-correct synthetic AAPL fixture so the parsers + pipeline run end-to-end.

## Architecture

```
core/             # no IO; deterministic; ASan/UBSan-clean
├── types/         Price, Quantity, OrderId, Side — int64 strong types
├── events/        OrderEvent variant — NewLimit/NewMarket/NewIoc/Cancel/Replace
├── orderbook/     Book = std::map<Price,Level,…> + unordered_map id_index_
│                  Level = std::deque<Order> with FIFO consume + erase_by_id
├── matching/      Engine — observer-only JournalSink + ScoreSink injection
└── scoring/       FeatureState (ring-buffer rolling stats) + Scorer (ONNX)
                   + MicropriceLut (Stoikov G(I,S) JSON lookup)

io/               # system-facing; core/ may NOT link this (CI grep guard)
└── journal/       JournalWriter/Reader (length-prefixed binary)
                   FillLogWriter/Reader (32-byte fixed records)

bindings/         pybind11 — Engine, Scorer, MicropriceLut, Book, Fill
apps/             matcher-cli, replay-cli
benchmarks/       bench-scoring (HdrHistogram p50/p90/p99/p999)
model/            train.py, evaluate.py, features.py, labels.py, schema.py
backtest/         driver.py, policies.py, metrics.py, plot.py, run_backtest.py
data/             ingestion/ (LOBSTER + Binance parsers), features/, tob/

tests/
├── unit/          51 Catch2 tests
├── property/      4 rapidcheck book invariants
├── integration/   Determinism replay + bench SLO gate
└── python/        62 pytest tests (CI mode) + data-marked tests for real LOBSTER
```

## Design notes

- **int64 ticks throughout** — no floats for prices. LOBSTER: 1 tick = $0.0001. Binance: 1 tick = $1e-8.
- **std::map for price levels** — required for deterministic iteration (see [ADR 0003](docs/adr/0003-stdmap-levels.md)).
- **std::format, not std::print** — Apple Clang's libc++ doesn't ship `<print>` yet.
- **Single-threaded matcher** — no atomics, no locks ([ADR 0004](docs/adr/0004-single-threaded-matcher.md)). Multi-symbol scaling is via sharding.
- **ONNX over TreeLite** — at 4.29 µs p99 we're 233× under SLO with no portability lock-in ([ADR 0002](docs/adr/0002-onnx-vs-treelite.md)).
- **Train/serve skew tested at rtol=1e-4** — Python `onnxruntime.InferenceSession` ≈ C++ `Scorer.probs_batch` across 1000 random vectors ([test](tests/python/test_onnx_cpp_parity.py)).
- **Score is observer-only** — matcher fills don't depend on the model output; the ScoreSink emits scores to whatever Python policy or logger you attach.

## ADRs

| | Title |
|---|---|
| [0001](docs/adr/0001-replay-determinism.md) | Replay determinism (4 invariants, CI-enforced) |
| [0002](docs/adr/0002-onnx-vs-treelite.md) | ONNX Runtime vs TreeLite for hot-path inference |
| [0003](docs/adr/0003-stdmap-levels.md) | `std::map` for price levels |
| [0004](docs/adr/0004-single-threaded-matcher.md) | Single-threaded matcher |
| [0005](docs/adr/0005-pooled-vs-perstock.md) | Pooled training + per-stock evaluation |
| [0006](docs/adr/0006-lightgbm-onnx-float32.md) | LightGBM ↔ ONNX float32 contract |
| [0007](docs/adr/0007-message-id-anchoring.md) | Message-id anchoring for features |
| [0008](docs/adr/0008-3class-label-deadband.md) | 3-class labels with 1-tick deadband |
| [0009](docs/adr/0009-single-day-train-limit.md) | Single-day training data — known limitation |

## Repo conventions

- Every commit is green. CI gates ASan/UBSan build + tests, the [4 determinism invariants](docs/adr/0001-replay-determinism.md), and the p99 < 1 ms SLO.
- `core/` does not link `io/`. CI grep guard enforces.
- No wall-clock reads or PRNG calls in `core/matching` or `core/orderbook` (ADR 0001 D1/D3). CI grep guard.
- LightGBM `n_estimators ≤ 300` to stay under ONNX's float32 drift knee (ADR 0006).
- 19-feature schema lives in `model/schema.py:FEATURE_NAMES` — single source of truth; the C++ `ScoredFeatures` struct mirrors it field-for-field. Bump `SCHEMA_VERSION` to change the contract.

## Contributing

Standard fork → branch → PR. Locally:

```bash
# C++ tests under ASan/UBSan
ctest --test-dir build/Debug --output-on-failure

# Python tests + ruff
uv run pytest tests/python -m "not data"
uv run ruff check .
```

If your change touches `core/matching` or `core/orderbook`, the determinism CI gate will run the [golden replay test](tests/integration/test_determinism.cpp) automatically. Don't bypass it; if it fires, your change has a real behavior delta you should document in a commit message.

## License

MIT.

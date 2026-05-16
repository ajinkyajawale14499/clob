# clob

End-to-end ML execution stack on order-book data: C++23 matching engine + ONNX adverse-selection model on the hot path + Python feature pipeline (LOBSTER equities + Binance crypto) + deterministic event-sourced replay + reproducible backtest.

**Status — `v0.5` MVP shippable:** C++23 matching engine with limit / market / IOC / cancel / cancel-replace, property-tested book invariants, append-only journal, and deterministic event-sourced replay. Python feature pipeline ingests LOBSTER + Binance into a unified BBO schema.

| Phase | Deliverable | Status |
|---|---|---|
| W1 | C++ toolchain + CI + first strong type | ✅ |
| W2 | Python ingestion: LOBSTER + Binance → unified BBO | ✅ |
| W3 | Feature library: microprice, OFI, imbalance | ✅ |
| W4 | Matching engine: limit + market + cancel | ✅ |
| W5 | Matcher completion: IOC + replace + property tests + CLI | ✅ `v0.1` |
| W6 | Append-only journal + `JournalSink` Engine integration | ✅ |
| W7 | Deterministic replay (ADR 0001 — 4 invariants, CI-enforced) | ✅ |
| W8 | MVP polish: demo + docs | ✅ `v0.5` |
| W9+ | ML stack (pybind11, LightGBM, ONNX hot-path, backtest, Streamlit) | — |

---

## Quickstart (<5 min from clone to passing tests)

```bash
# Prereqs (one-time): cmake 3.28+, ninja, conan 2.x, gcc 14+ or clang 17+.
# macOS:  brew install cmake ninja
#         pipx install conan && conan profile detect --force
# Linux:  apt install cmake ninja-build g++-14 && pipx install conan && conan profile detect --force

git clone https://github.com/ajinkyajawale14499/clob.git
cd clob

# Conan resolves Catch2 + rapidcheck; CMake configures with ASan/UBSan presets.
conan install . -s build_type=Debug --build=missing
cmake --preset asan
cmake --build build/Debug --parallel
ctest --test-dir build/Debug --output-on-failure
# Expected: 60+ tests pass under ASan/UBSan.
```

Two binaries land in `build/Debug/apps/`:
- `matcher-cli` — interactive REPL (`--journal=PATH` to record).
- `replay-cli` — feed a journal back through a fresh engine, write a fill log.

## Demo

```bash
bash docs/demo.sh
```

Drives both binaries end-to-end: builds a 2-sided book, sends a crossing order, journals the session, replays it through a fresh engine, then runs the replay a second time and `cmp`s the two output files. The final `✓ byte-identical` line proves the determinism contract (see ADR 0001).

To record an asciinema cast of the demo:

```bash
pipx install asciinema    # or: brew install asciinema
asciinema rec -c "bash docs/demo.sh" docs/demo.cast
```

## Python pipeline

```bash
uv sync                          # one-time
uv run pytest tests/python -v    # 11 active tests (data-marked tests skip without LOBSTER)
```

To exercise the data-dependent paths, download a LOBSTER sample (manual browser step — direct curl returns a React SPA shell) and a Binance bookTicker daily file into `data/raw/`:

```bash
# Manual: https://lobsterdata.com/info/DataSamples.php → AAPL depth-10 → unzip into data/raw/
curl -o data/raw/binance.zip \
  "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-01-15.zip"
unzip -d data/raw data/raw/binance.zip && rm data/raw/binance.zip

uv run pytest tests/python -v    # now 24 tests run (13 previously skipped come alive)
```

## Architecture

```
core/         (no IO)
  types/        Price, Quantity, OrderId, Side — int64 strong types
  events/       OrderEvent variant (NewLimit, NewMarket, NewIoc, Cancel, Replace)
  orderbook/    Book = std::map<Price, Level, …> + unordered_map id_index_
                Level = std::deque<Order> with consume_front/erase_by_id
  matching/    Engine = match_against helper + JournalSink injection

io/           (system-facing — core/ may NOT link this; enforced in CI)
  journal/      JournalWriter / JournalReader (length-prefixed binary)
                FillLogWriter / FillLogReader (32-byte fixed records)

apps/         matcher-cli, replay-cli

tests/
  unit/         44 Catch2 tests across types/level/book/engine/journal
  property/     4 rapidcheck invariants on the Engine
  integration/  Determinism replay test (ADR 0001)
```

Key design choices:
- **`int64` ticks throughout, never floats for prices.** LOBSTER: 1 tick = $0.0001. Binance: 1 tick = $1e-8.
- **`std::map` price levels.** Predictable iteration order is required for determinism (ADR 0001).
- **Single-threaded matcher.** No locks, no atomics on the hot path.
- **`std::format`, not `std::print`/`std::println`.** Apple Clang's libc++ doesn't ship `<print>` yet.
- **Every commit is green.** No "broken intermediate" commits.

## Reading order

1. [`docs/adr/0001-replay-determinism.md`](docs/adr/0001-replay-determinism.md) — the determinism contract: 4 invariants, all CI-enforced.
2. `core/matching/engine.cpp` — `match_against` is the heart of the matcher (~30 LoC).
3. `core/orderbook/book.cpp` — `Book` with two `std::map`s + unordered id index.
4. `tests/integration/test_determinism.cpp` — end-to-end determinism check.

## What this is NOT

- Not HFT-grade. MFT-flavored (millisecond-to-tens-of-millisecond inference budget).
- Single-symbol per matcher instance, single-threaded core, no kernel bypass.
- See `docs/what-i-would-do-at-hft-grade.md` (W14) for what was deliberately skipped.

## License

MIT. See `LICENSE`.

# clob

End-to-end ML execution stack on order-book data: C++23 matching engine + ONNX adverse-selection model on the hot path + Python feature pipeline (LOBSTER equities + Binance crypto) + deterministic event-sourced replay + reproducible backtest.

**Status:** W1 — toolchain ramp.

## Quickstart (C++ only, until W2)

```bash
conan install . -s build_type=Debug --build=missing
cmake --preset asan
cmake --build build/Debug --parallel
ctest --preset asan --output-on-failure
```

## What this is NOT

- Not HFT-grade. MFT-flavored (millisecond-to-tens-of-millisecond inference budget).
- Single-symbol per matcher instance, single-threaded core, no kernel bypass.
- See `docs/what-i-would-do-at-hft-grade.md` (W14) for what was deliberately skipped.

## License

MIT. See `LICENSE`.

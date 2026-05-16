# clob — scoring latency benchmark

Measured: scored `Engine::add_limit` (FeatureState snapshot + Scorer + ScoreSink + match_against + FeatureState observe)
over **100000** ops after **10000** warmup ops on a 1000-order seeded book.

Includes: feature assembly + ONNX inference + matcher hot path. Excludes: IO, model load, process startup.

**SLO** (ADR 0001 amendment): p99 < 1ms. **Stretch** (ADR 0002 W14): p99 < 200us via TreeLite.

| Percentile | Latency (ns) | Latency (us) |
|---|---|---|
| p50 | 70847 | 70.847 |
| p90 | 73727 | 73.727 |
| p99 | 83135 | 83.135 |
| p999 | 126271 | 126.271 |
| max | 5570559 | 5570.56 |

Gate: :white_check_mark: PASS (p99 = 83.135 us vs SLO 1000 us)

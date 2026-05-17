# clob — scoring latency benchmark

Measured: scored `Engine::add_limit` (FeatureState snapshot + Scorer + ScoreSink + match_against + FeatureState observe)
over **100000** ops after **10000** warmup ops on a 1000-order seeded book.

Includes: feature assembly + ONNX inference + matcher hot path. Excludes: IO, model load, process startup.

**SLO** (ADR 0001 amendment): p99 < 1ms. **Stretch** (ADR 0002 W14): p99 < 200us via TreeLite.

| Percentile | Latency (ns) | Latency (us) |
|---|---|---|
| p50 | 70911 | 70.911 |
| p90 | 75391 | 75.391 |
| p99 | 88255 | 88.255 |
| p999 | 149247 | 149.247 |
| max | 6549503 | 6549.5 |

Gate: :white_check_mark: PASS (p99 = 88.255 us vs SLO 1000 us)

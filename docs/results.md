# clob v1.0 — Backtest results

**Policy A (naive maker)**: always quotes 1 contract at L1 ± 1 tick on every event.  
**Policy B (ML-aware maker)**: same baseline, but suppresses the bid quote when the model's `P(Up) - P(Down)` score > +threshold (model expects upward move; resting bid would be adversely picked off) and symmetrically suppresses the ask when score < -threshold.

Both policies use the same C++ Engine + ONNX scorer on the matcher's hot path (p99 < 5µs — see `docs/bench.md`). Backtest replays the full 5-stock LOBSTER 2012-06-21 day.

## Per-stock metrics

| ticker | policy | fills | fill rate | markout mean ticks | adverse selection bps | gross pnl ticks | quotes posted |
|---|---|---|---|---|---|---|---|
| AAPL | naive | 87498 | 0.8750 | +12.9355 | -2.2182 | 1131828.5000 | 99994 |
| AAPL | ml_aware | 78537 | 0.8932 | +12.4418 | -2.1335 | 977140.5000 | 87927 |
| AMZN | naive | 84693 | 0.8469 | +3.2944 | -1.4792 | 279013.5000 | 99998 |
| AMZN | ml_aware | 76752 | 0.8531 | +3.0655 | -1.3764 | 235286.5000 | 89965 |
| GOOG | naive | 82316 | 0.8234 | +7.5845 | -1.3288 | 624322.5000 | 99976 |
| GOOG | ml_aware | 73586 | 0.8417 | +7.8705 | -1.3789 | 579161.0000 | 87423 |
| INTC | naive | 90565 | 0.9057 | +1.3461 | -4.9758 | 121910.0000 | 99990 |
| INTC | ml_aware | 77342 | 0.9074 | +1.2242 | -4.5253 | 94685.0000 | 85231 |
| MSFT | naive | 86746 | 0.8678 | +1.0935 | -3.5795 | 94856.0000 | 99964 |
| MSFT | ml_aware | 72521 | 0.8806 | +0.9810 | -3.2113 | 71144.0000 | 82358 |

## Reading the metrics

- `markout_mean_ticks` = mean of (mid[t+K] − fill_price) × side_sign over all filled policy quotes (K = 100 events). **Positive = favorable to the policy.**
- `adverse_selection_bps` = mean adverse markout in bps of mid price. **Negative = policy gains; positive = policy loses.**
- `fill_rate` = filled_quotes / posted_quotes.
- `gross_pnl_ticks` = sum of signed markouts (no inventory model — no Sharpe).

## Caveats (ADR 0009 — single-day data)

This is a 1-day backtest with 70/30 within-day train/val split. Production-grade ML would use multi-day walk-forward per Briola 2024 (45 train / 5 val / 10 test). The model's stability across days is not validated; ADR 0009 documents this. Future work: license additional LOBSTER days.

LOBSTER partial cancels (event_type=2) are approximated as full cancels — ~5% of events. ADR 0009 caveat.

INTC/MSFT have >65% "Stable" labels at all K values (large-tick stocks with $0.01 ticks and ~$25 prices). Per-stock AUC for these is degenerate; the pooled model still extracts useful signal from AAPL/AMZN/GOOG which dominate the validation set.

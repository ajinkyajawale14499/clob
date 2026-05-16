"""run_backtest — CLI entry point for the W12 results pipeline.

Usage:
    uv run python -m backtest.run_backtest [--stock STOCK] [--policy {both,naive,ml_aware}]
                                            [--max-events N] [--threshold T]
                                            [--output PATH] [--markdown PATH]

If --stock is "all" (default), runs all 5 LOBSTER stocks. For each, runs both
policies and writes:
    - <output>: JSON of all (stock, policy) metric dicts
    - <markdown>: docs/results.md with per-stock table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backtest.driver import run_backtest
from backtest.metrics import summarise

ALL_TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]


def _md_table(rows: list[dict[str, Any]]) -> str:
    headers = ["ticker", "policy", "fills", "fill_rate", "markout_mean_ticks",
               "adverse_selection_bps", "gross_pnl_ticks", "quotes_posted"]
    out = ["| " + " | ".join(h.replace("_", " ") for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for h in headers:
            v = r[h]
            if isinstance(v, float):
                cells.append(f"{v:+.4f}" if "markout" in h or "selection" in h
                              else f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _write_results_md(rows: list[dict], path: Path) -> None:
    """Build the v1.0 docs/results.md with headline metrics + per-stock table."""
    text = [
        "# clob v1.0 — Backtest results",
        "",
        ("**Policy A (naive maker)**: always quotes 1 contract at L1 ± 1 tick on "
         "every event.  "),
        ("**Policy B (ML-aware maker)**: same baseline, but suppresses the "
         "bid quote when the model's `P(Up) - P(Down)` score > +threshold "
         "(model expects upward move; resting bid would be adversely picked off) "
         "and symmetrically suppresses the ask when score < -threshold."),
        "",
        ("Both policies use the same C++ Engine + ONNX scorer on the matcher's "
         "hot path (p99 < 5µs — see `docs/bench.md`). Backtest replays the full "
         "5-stock LOBSTER 2012-06-21 day."),
        "",
        "## Per-stock metrics",
        "",
        _md_table(rows),
        "",
        "## Reading the metrics",
        "",
        "- `markout_mean_ticks` = mean of (mid[t+K] − fill_price) × side_sign over all "
        "filled policy quotes (K = 100 events). **Positive = favorable to the policy.**",
        "- `adverse_selection_bps` = mean adverse markout in bps of mid price. "
        "**Negative = policy gains; positive = policy loses.**",
        "- `fill_rate` = filled_quotes / posted_quotes.",
        "- `gross_pnl_ticks` = sum of signed markouts (no inventory model — no Sharpe).",
        "",
        "## Caveats (ADR 0009 — single-day data)",
        "",
        ("This is a 1-day backtest with 70/30 within-day train/val split. Production-"
         "grade ML would use multi-day walk-forward per Briola 2024 (45 train / 5 val / "
         "10 test). The model's stability across days is not validated; ADR 0009 "
         "documents this. Future work: license additional LOBSTER days."),
        "",
        ("LOBSTER partial cancels (event_type=2) are approximated as full cancels — "
         "~5% of events. ADR 0009 caveat."),
        "",
        ("INTC/MSFT have >65% \"Stable\" labels at all K values (large-tick stocks "
         "with $0.01 ticks and ~$25 prices). Per-stock AUC for these is degenerate; "
         "the pooled model still extracts useful signal from AAPL/AMZN/GOOG which "
         "dominate the validation set."),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="all",
                         choices=["all", *ALL_TICKERS])
    parser.add_argument("--policy", default="both",
                         choices=["both", "naive", "ml_aware"])
    parser.add_argument("--max-events", type=int, default=None,
                         help="Limit events per stock (smoke testing).")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", default="model/artifacts/results.json")
    parser.add_argument("--markdown", default="docs/results.md")
    args = parser.parse_args()

    stocks = ALL_TICKERS if args.stock == "all" else [args.stock]
    policies = ["naive", "ml_aware"] if args.policy == "both" else [args.policy]

    rows: list[dict] = []
    for ticker in stocks:
        for policy in policies:
            print(f"--- {ticker} / {policy} ---")
            r = run_backtest(ticker, policy,
                              threshold=args.threshold,
                              max_events=args.max_events)
            m = summarise(r)
            rows.append(m)
            print(f"  fills={m['fills']:>6}  fill_rate={m['fill_rate']:.3f}  "
                  f"markout_mean={m['markout_mean_ticks']:+.2f}t  "
                  f"adverse_bps={m['adverse_selection_bps']:+.3f}  "
                  f"pnl={m['gross_pnl_ticks']:+.0f}t")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {output_path}")

    if args.stock == "all" and args.policy == "both":
        _write_results_md(rows, Path(args.markdown))


if __name__ == "__main__":
    main()

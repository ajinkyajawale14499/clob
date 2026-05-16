"""Render backtest result charts to docs/charts/.

Reads:
    model/artifacts/results.json   — from `python -m backtest.run_backtest`
    model/artifacts/model_meta.json — from `python -m model.train`
                                       + `python -m model.evaluate`

Produces (PNGs, ~120 DPI):
    docs/charts/policy_a_vs_b_summary.png   — grouped bar chart, 5 metrics
    docs/charts/markout_per_stock.png       — Naive vs ML-aware markout per stock
    docs/charts/auc_per_stock.png           — from model_meta.json
    docs/charts/feature_importance.png      — copied from artifacts/ (gitignored
                                              source -> committed copy)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"
CHARTS_DIR = REPO_ROOT / "docs" / "charts"


def _load_results() -> list[dict]:
    p = ARTIFACTS_DIR / "results.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Run `uv run python -m backtest.run_backtest` first to produce {p}")
    return json.loads(p.read_text())


def _load_model_meta() -> dict | None:
    p = ARTIFACTS_DIR / "model_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def chart_policy_a_vs_b_summary(rows: list[dict]) -> Path:
    """Grouped bar chart: per-stock fills + markout side by side."""
    stocks = sorted({r["ticker"] for r in rows})
    by = {(r["ticker"], r["policy"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: fills count.
    x = np.arange(len(stocks))
    width = 0.38
    naive_fills = [by[(s, "naive")]["fills"] for s in stocks]
    ml_fills = [by[(s, "ml_aware")]["fills"] for s in stocks]
    axes[0].bar(x - width/2, naive_fills, width, label="naive",  color="#888888")
    axes[0].bar(x + width/2, ml_fills,    width, label="ml-aware", color="#3678c4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stocks)
    axes[0].set_ylabel("filled policy quotes")
    axes[0].set_title("Quotes filled (50k events / stock)")
    axes[0].legend()

    # Right: markout (positive = favorable).
    naive_mk = [by[(s, "naive")]["markout_mean_ticks"] for s in stocks]
    ml_mk = [by[(s, "ml_aware")]["markout_mean_ticks"] for s in stocks]
    axes[1].bar(x - width/2, naive_mk, width, label="naive",   color="#888888")
    axes[1].bar(x + width/2, ml_mk,    width, label="ml-aware", color="#3678c4")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stocks)
    axes[1].set_ylabel("mean markout (ticks; +ve = favorable)")
    axes[1].set_title("Per-fill markout: Naive vs ML-aware")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend()

    fig.suptitle("clob v1.0 backtest — Policy A vs B", fontsize=14)
    fig.tight_layout()
    out = CHARTS_DIR / "policy_a_vs_b_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_adverse_selection_per_stock(rows: list[dict]) -> Path:
    """Adverse-selection bps (more negative = better) per stock."""
    stocks = sorted({r["ticker"] for r in rows})
    by = {(r["ticker"], r["policy"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(stocks))
    width = 0.38
    naive_adv = [by[(s, "naive")]["adverse_selection_bps"] for s in stocks]
    ml_adv    = [by[(s, "ml_aware")]["adverse_selection_bps"] for s in stocks]
    ax.bar(x - width/2, naive_adv, width, label="naive",    color="#888888")
    ax.bar(x + width/2, ml_adv,    width, label="ml-aware", color="#3678c4")
    ax.set_xticks(x)
    ax.set_xticklabels(stocks)
    ax.set_ylabel("adverse selection (bps; more negative = policy gains more)")
    ax.set_title("Adverse selection per stock — Naive vs ML-aware")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    out = CHARTS_DIR / "adverse_selection_per_stock.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_auc_per_stock(meta: dict) -> Path:
    """Per-stock val AUC up-vs-down from model_meta.json."""
    per_stock = meta["per_stock"]
    stocks = sorted(per_stock.keys())
    aucs = [per_stock[s]["auc_up_vs_down"] for s in stocks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(stocks, aucs, color=["#3678c4" if 0.5 < a < 0.7 else "#cccccc"
                                         for a in aucs])
    for bar, auc in zip(bars, aucs, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 if not np.isnan(auc) else 0.02,
                 "NaN" if np.isnan(auc) else f"{auc:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax.axhspan(0.55, 0.65, alpha=0.15, color="green",
                label="Briola 2024 band [0.55, 0.65]")
    ax.axhline(0.5, color="k", lw=0.5, linestyle="--", label="random")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("val AUC up vs down")
    ax.set_title(f"Per-stock validation AUC (K={meta['k_best']}, "
                 f"pooled training)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = CHARTS_DIR / "auc_per_stock.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_feature_importance(meta: dict) -> Path:
    """Top-15 features by LightGBM gain."""
    importances = meta.get("feature_importance_gain")
    if not importances:
        return CHARTS_DIR / "feature_importance.png"  # nothing to plot
    importances = importances[:15]
    names = [n for n, _ in importances]
    gains = [g for _, g in importances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], gains[::-1], color="#3678c4")
    ax.set_title(f"LightGBM feature importance (gain) — K={meta['k_best']}")
    ax.set_xlabel("gain")
    fig.tight_layout()
    out = CHARTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_results()
    print(f"Loaded {len(rows)} result rows")

    out1 = chart_policy_a_vs_b_summary(rows)
    print(f"  {out1}")
    out2 = chart_adverse_selection_per_stock(rows)
    print(f"  {out2}")

    meta = _load_model_meta()
    if meta is not None:
        # If evaluate.py already wrote calibration_per_stock.png, copy it over.
        src = ARTIFACTS_DIR / "calibration_per_stock.png"
        if src.exists():
            dst = CHARTS_DIR / "calibration_per_stock.png"
            shutil.copy(src, dst)
            print(f"  {dst}")
        out3 = chart_auc_per_stock(meta)
        print(f"  {out3}")
        out4 = chart_feature_importance(meta)
        print(f"  {out4}")
    else:
        print("(model_meta.json missing — skipping AUC/feature importance plots)")


if __name__ == "__main__":
    main()

"""Standalone evaluation: loads model + recomputes metrics + calibration.

Run via `uv run python -m model.evaluate` after `python -m model.train`.
Produces:
    - Per-stock confusion matrix + calibration curve printed to stdout
    - model/artifacts/calibration_<TICKER>.png  (matplotlib; gitignored)
    - model/artifacts/feature_importance.png    (gitignored)
    - Updates model/artifacts/model_meta.json with `calibration_brier_pool`

Brier score baseline: a uniform-predict-1/3 multiclass model has Brier ≈ 0.667.
Anything lower is genuine signal. Random 50/50 binary has Brier=0.25; the
multiclass equivalent we should beat is ~0.45-0.55.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix

from model.schema import ALL_TICKERS, FEATURE_DTYPE, FEATURE_NAMES, LABEL_CLASSES
from model.train import (
    _ensure_microprice_lut,
    _load_stock_features_and_labels,
    _time_split,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_model_and_meta() -> tuple[lgb.Booster, dict]:
    """Load the trained model + its metadata. Both must exist on disk."""
    meta_path = ARTIFACTS_DIR / "model_meta.json"
    lgb_path = ARTIFACTS_DIR / "model.lgb"
    if not meta_path.exists() or not lgb_path.exists():
        raise FileNotFoundError(
            "Run `uv run python -m model.train` first to produce "
            f"{lgb_path} and {meta_path}."
        )
    meta = json.loads(meta_path.read_text())
    booster = lgb.Booster(model_file=str(lgb_path))
    return booster, meta


def multiclass_brier(y_true: np.ndarray, p_classes: np.ndarray) -> float:
    """Multiclass Brier = mean over rows of Σ_c (p_c - 1{y=c})^2.

    Baseline: uniform (1/3, 1/3, 1/3) on 3-class -> Brier = 2/3 = 0.667.
    """
    n, k = p_classes.shape
    y_onehot = np.zeros((n, k))
    for c in range(k):
        y_onehot[y_true == c, c] = 1.0
    return float(np.mean(np.sum((p_classes - y_onehot) ** 2, axis=1)))


def per_class_calibration(y_true: np.ndarray, p_classes: np.ndarray) -> dict:
    """For each class, compute Brier (one-vs-rest) + reliability bins (10 quantiles)."""
    out: dict = {}
    for cls in LABEL_CLASSES:
        y_bin = (y_true == cls).astype(int)
        p_bin = p_classes[:, cls]
        brier = float(brier_score_loss(y_bin, p_bin))
        # Use min(10, unique_bins) — if all probs are tightly clustered, calibration_curve fails.
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, p_bin, n_bins=10,
                                                     strategy="quantile")
            bins = [
                {"mean_pred": float(mp), "frac_pos": float(fp)}
                for mp, fp in zip(mean_pred, frac_pos, strict=False)
            ]
        except ValueError:
            bins = []
        out[cls] = {"brier": brier, "bins": bins}
    return out


def evaluate(k: int | None = None, save_plots: bool = True) -> dict:
    """Re-evaluate the saved model on per-stock val sets; return rich metrics dict."""
    booster, meta = load_model_and_meta()
    if k is None:
        k = meta["k_best"]

    lut = _ensure_microprice_lut()
    results: dict = {
        "k": k,
        "n_features": len(FEATURE_NAMES),
        "per_stock": {},
        "pooled": {},
    }

    all_X_va, all_y_va = [], []
    for ticker in ALL_TICKERS:
        X, y, _ = _load_stock_features_and_labels(ticker, k, lut)
        _, _, Xva, yva = _time_split(X, y, np.arange(len(y)))
        # booster.predict returns probabilities for multiclass.
        p_va = booster.predict(Xva.astype(FEATURE_DTYPE))
        cm = confusion_matrix(yva, p_va.argmax(axis=1), labels=list(LABEL_CLASSES))
        results["per_stock"][ticker] = {
            "n_val": int(len(yva)),
            "brier_multiclass": multiclass_brier(yva, p_va),
            "calibration": per_class_calibration(yva, p_va),
            "confusion_matrix": cm.tolist(),
        }
        all_X_va.append(Xva); all_y_va.append(yva)

    Xva_pool = np.concatenate(all_X_va)
    yva_pool = np.concatenate(all_y_va)
    p_pool = booster.predict(Xva_pool.astype(FEATURE_DTYPE))
    results["pooled"] = {
        "n_val": int(len(yva_pool)),
        "brier_multiclass": multiclass_brier(yva_pool, p_pool),
        "calibration": per_class_calibration(yva_pool, p_pool),
    }
    results["feature_importance_gain"] = sorted(
        zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain"),
            strict=False),
        key=lambda kv: kv[1], reverse=True,
    )

    if save_plots:
        _save_plots(results, p_pool, yva_pool, booster)

    # Patch meta with new calibration numbers + persist.
    meta["calibration_brier_pool"] = results["pooled"]["brier_multiclass"]
    meta["confusion_matrix_pool"] = (
        confusion_matrix(yva_pool, p_pool.argmax(axis=1),
                         labels=list(LABEL_CLASSES))).tolist()
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return results


def _save_plots(results: dict, p_pool: np.ndarray, y_pool: np.ndarray,
                booster: lgb.Booster) -> None:
    """Render calibration + feature importance to PNGs (gitignored artifacts)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 1. Calibration curves per stock (5-panel grid).
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, ticker in zip(axes, ALL_TICKERS, strict=False):
        for cls in LABEL_CLASSES:
            bins = results["per_stock"][ticker]["calibration"][cls]["bins"]
            if not bins:
                continue
            x = [b["mean_pred"] for b in bins]
            y = [b["frac_pos"] for b in bins]
            ax.plot(x, y, marker="o", label=f"class {cls}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(ticker)
        ax.set_xlabel("predicted prob")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction positive")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Per-stock calibration (10-bin quantile)  K={results['k']}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "calibration_per_stock.png", dpi=120)
    plt.close(fig)

    # 2. Feature importance.
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [n for n, _ in results["feature_importance_gain"]][:20]
    gains = [g for _, g in results["feature_importance_gain"]][:20]
    ax.barh(names[::-1], gains[::-1])
    ax.set_title(f"LightGBM feature importance (gain) — K={results['k']}")
    ax.set_xlabel("gain")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("Loading model + metadata...")
    results = evaluate()
    print(f"\n=== Pooled metrics (K={results['k']}) ===")
    print(f"  Brier (multiclass): {results['pooled']['brier_multiclass']:.4f}")
    print(f"  (baseline uniform 1/3: 0.667; lower is better)")
    print("\n=== Per-stock Brier (multiclass) ===")
    for ticker, m in results["per_stock"].items():
        print(f"  {ticker}: Brier={m['brier_multiclass']:.4f}  n_val={m['n_val']}")
    print("\n=== Top 10 features by gain ===")
    for name, gain in results["feature_importance_gain"][:10]:
        print(f"  {name:30s}  {gain:10.0f}")
    print(f"\nPlots saved to {ARTIFACTS_DIR}/calibration_per_stock.png, "
          f"{ARTIFACTS_DIR}/feature_importance.png")


if __name__ == "__main__":
    main()

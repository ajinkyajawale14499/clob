"""Training reproducibility + sanity gates (data-marked).

These tests run the actual LightGBM pipeline on real LOBSTER, so they require:
  1. data/raw/*.csv (5 stocks)
  2. model/artifacts/microprice_g.json (regenerable via train.py)

Gates enforced (per ADR 0008, risk R4):
  - val_auc_pool_up_vs_down ∈ [0.52, 0.70]
    < 0.52: features/labels broken
    > 0.70: leakage suspected (Briola 2024 ceiling on real LOBSTER is ~0.65)
  - Re-running with same seed produces identical val_logloss (within FP noise)
  - Balanced under-sampling produces ~33%/33%/33% train shares
"""

from pathlib import Path

import pytest

from model.train import train_one

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"


def _have_lobster() -> bool:
    return len(list(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))) > 0


if not _have_lobster():
    pytest.skip("No LOBSTER files in data/raw/", allow_module_level=True)


@pytest.fixture(scope="module")
def trained_k10():
    """Train one model at K=10 (fastest horizon, smallest data drop)."""
    _, result = train_one(k=10, seed=42, samples_per_class=5000)
    return result


def test_val_auc_in_realistic_band(trained_k10):
    """Briola 2024 anchor: 0.55-0.65 on raw LOBSTER. Floor 0.52, ceiling 0.70.

    Below 0.52: features/labels broken — feature pipeline order, leakage in
    labels, or stale microprice LUT.
    Above 0.70: leakage almost certain — model is seeing the future.
    """
    auc = trained_k10.val_auc_pool_up_vs_down
    assert auc > 0.52, f"val AUC {auc:.4f} <= 0.52 — features/labels broken"
    assert auc < 0.70, f"val AUC {auc:.4f} >= 0.70 — leakage suspected"


def test_training_deterministic_under_same_seed():
    """LightGBM + same seed + same data -> identical val_logloss within FP noise."""
    _, r1 = train_one(k=10, seed=42, samples_per_class=2000)
    _, r2 = train_one(k=10, seed=42, samples_per_class=2000)
    assert abs(r1.val_logloss_pool - r2.val_logloss_pool) < 1e-6, (
        f"non-deterministic: {r1.val_logloss_pool} vs {r2.val_logloss_pool}"
    )


def test_balanced_undersampling_produces_roughly_equal_train_shares(trained_k10):
    """After balanced under-sampling, train shares should be near 1/3 each."""
    shares = trained_k10.train_class_shares
    for cls in (0, 1, 2):
        assert 0.25 < shares[cls] < 0.45, (
            f"class {cls} share {shares[cls]:.3f} out of [0.25, 0.45] — "
            f"balanced under-sampling broken?"
        )


def test_per_stock_metrics_all_present(trained_k10):
    """All 5 stocks should have per-stock metrics in the result.

    AUC may be NaN for large-tick stocks at small K (INTC/MSFT at K=10 have
    ~99.9% stable rows; _auc_up_vs_down returns NaN when the non-stable
    sample is too small to be reliable). NaN here is a feature — it signals
    'no signal at this K for this stock; use K=50/100 instead'.
    """
    import math
    assert set(trained_k10.per_stock.keys()) == {
        "AAPL", "AMZN", "GOOG", "INTC", "MSFT"
    }
    for ticker, metrics in trained_k10.per_stock.items():
        assert metrics.n_val > 1000
        if math.isnan(metrics.auc_up_vs_down):
            non_stable = metrics.class_shares_val[0] + metrics.class_shares_val[2]
            assert non_stable < 0.01, (
                f"{ticker}: AUC is NaN but non-stable share {non_stable:.4f} >= 1%"
            )
        else:
            assert 0.0 < metrics.auc_up_vs_down < 1.0


def test_n_trees_used_under_adr_0006_cap(trained_k10):
    """Early stopping should usually pick < 300 trees, but the cap is enforced."""
    assert trained_k10.n_trees_used <= 300

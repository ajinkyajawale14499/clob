"""Backtest determinism: same LOBSTER + same model -> same metrics dict."""

from pathlib import Path

import pytest

from backtest.driver import run_backtest
from backtest.metrics import summarise

pytestmark = pytest.mark.data

SAMPLE_DIR = Path(__file__).parents[2] / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).parents[2] / "model" / "artifacts"


def _have_everything() -> bool:
    return (
        len(list(SAMPLE_DIR.glob("AAPL_*_message_*.csv"))) > 0
        and (ARTIFACTS_DIR / "model.onnx").exists()
        and (ARTIFACTS_DIR / "microprice_g.json").exists()
    )


if not _have_everything():
    pytest.skip("LOBSTER + model.onnx + microprice_g.json required",
                allow_module_level=True)


def test_naive_backtest_deterministic():
    """Two runs of the naive policy on the same data -> identical metrics."""
    r1 = summarise(run_backtest("AAPL", "naive", max_events=5_000))
    r2 = summarise(run_backtest("AAPL", "naive", max_events=5_000))
    assert r1 == r2, f"naive backtest non-deterministic: {r1} vs {r2}"


def test_ml_aware_backtest_deterministic():
    """ML-aware also deterministic (model is loaded once per run; ONNX inference is pure)."""
    r1 = summarise(run_backtest("AAPL", "ml_aware", max_events=5_000))
    r2 = summarise(run_backtest("AAPL", "ml_aware", max_events=5_000))
    assert r1 == r2, f"ml_aware backtest non-deterministic: {r1} vs {r2}"


def test_naive_and_ml_aware_produce_different_results():
    """If both policies produced identical metrics, something is wrong (probably
    the model isn't affecting quote decisions). A 5k-event slice should show
    at least a single-fill difference."""
    r_naive = summarise(run_backtest("AAPL", "naive", max_events=5_000))
    r_ml = summarise(run_backtest("AAPL", "ml_aware", max_events=5_000))
    assert r_naive["quotes_posted"] != r_ml["quotes_posted"], (
        "naive and ml_aware posted same quote count — ml_aware threshold may be off"
    )

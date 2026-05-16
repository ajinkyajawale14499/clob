"""CheckList-style behavioral tests for the trained model.

Test categories (Ribeiro et al. 2020 — Beyond Accuracy):
  - MFT (Minimum Functionality): obvious cases the model must get right
  - INV (Invariance): irrelevant transformations shouldn't change predictions
  - DIR (Directional): monotone responses to monotone inputs

These run through clob_py.Scorer (the deployed C++ inference path) so any
regression in either training or inference fires here.
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import clob_py
from tests.fixtures.book import make_book

pytestmark = pytest.mark.data

MODEL_PATH = Path(__file__).parents[2] / "model" / "artifacts" / "model.onnx"


def _have_model() -> bool:
    return MODEL_PATH.exists()


if not _have_model():
    pytest.skip("model/artifacts/model.onnx missing", allow_module_level=True)


@pytest.fixture(scope="module")
def scorer():
    return clob_py.Scorer.load(str(MODEL_PATH))


# ----- MFT: Minimum Functionality -----------------------------------------

def test_mft_heavy_bid_imbalance_predicts_up(scorer):
    """Strong bid imbalance + positive OFI/TFI -> score > 0 (P(Up) > P(Down))."""
    score = scorer.score(make_book("heavy_bid_imbalance").features())
    assert score > 0.05, f"heavy bid imbalance -> score {score:.4f}, expected > 0.05"


def test_mft_heavy_ask_imbalance_predicts_down(scorer):
    """Strong ask imbalance -> score < 0."""
    score = scorer.score(make_book("heavy_ask_imbalance").features())
    assert score < -0.05, f"heavy ask imbalance -> score {score:.4f}, expected < -0.05"


def test_mft_balanced_book_near_neutral(scorer):
    """Balanced book with zeros across the board -> score near 0 (no strong signal).

    'Near zero' allows |score| < 0.30 since the trained model may have a baseline
    Down/Up asymmetry from the LOBSTER day's drift.
    """
    score = scorer.score(make_book("balanced_book").features())
    assert abs(score) < 0.30, f"balanced book -> score {score:.4f}, expected |score| < 0.30"


@pytest.mark.parametrize("ticker", ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"])
def test_mft_per_stock_score_in_range(scorer, ticker):
    """Every ticker can produce scores in [-1, +1] without crashing."""
    score = scorer.score(make_book("balanced_book", ticker=ticker).features())
    assert -1.0 <= score <= 1.0


# ----- INV: Invariance ----------------------------------------------------

def test_inv_repeated_call_is_deterministic(scorer):
    """Same features twice -> same score (down to floating point)."""
    f = make_book("heavy_bid_imbalance").features()
    s1 = scorer.score(f)
    s2 = scorer.score(f)
    assert abs(s1 - s2) < 1e-12


def test_inv_batch_one_equals_single(scorer):
    """score_batch([x])[0] == score(x) (verified at task 6.5 — re-checked here)."""
    f = make_book("heavy_bid_imbalance").features()
    single = scorer.score(f)
    batched = scorer.score_batch(np.array([f]))[0]
    assert abs(single - batched) < 1e-12


# ----- DIR: Directional ---------------------------------------------------

def test_dir_score_monotone_in_imbalance(scorer):
    """As imbalance sweeps from -1 to +1, score should generally increase.

    We don't require strict monotonicity (tree splits can produce flat regions)
    but the trend must be unambiguous: score at +0.95 > score at -0.95.
    """
    high = scorer.score(make_book("balanced_book").with_(imbalance_l1=0.95).features())
    low = scorer.score(make_book("balanced_book").with_(imbalance_l1=-0.95).features())
    assert high > low, f"high={high:.4f}, low={low:.4f}"


def test_dir_score_responds_to_ofi(scorer):
    """Positive OFI raises score; negative lowers it (holding imbalance constant)."""
    base = make_book("balanced_book").features()
    pos = make_book("balanced_book").with_(ofi_50=1000.0, ofi_200=3000.0).features()
    neg = make_book("balanced_book").with_(ofi_50=-1000.0, ofi_200=-3000.0).features()
    assert scorer.score(pos) > scorer.score(base)
    assert scorer.score(neg) < scorer.score(base)


# ----- Property-based: continuity in imbalance ----------------------------

@settings(max_examples=20, deadline=None)
@given(imbalance=st.floats(min_value=-0.5, max_value=0.5))
def test_score_continuous_in_small_imbalance_shifts(imbalance):
    """Small (≤0.01) changes in imbalance shouldn't swing score by > 0.30.

    Tree models can have step-like responses but a 1% input perturbation
    should never produce a 30%+ score swing — that'd indicate the model is
    sitting right on a decision boundary, which is rare for well-trained
    LightGBM ensembles.
    """
    if not _have_model():
        return
    scorer = clob_py.Scorer.load(str(MODEL_PATH))
    f1 = make_book("balanced_book").with_(imbalance_l1=float(imbalance)).features()
    f2 = make_book("balanced_book").with_(imbalance_l1=float(imbalance) + 0.01).features()
    s1 = scorer.score(f1)
    s2 = scorer.score(f2)
    assert abs(s2 - s1) < 0.30

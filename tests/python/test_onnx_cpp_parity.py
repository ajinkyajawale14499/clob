"""Train/serve skew test — THE critical W10 gate (ADR 0006).

Loads the same ONNX model in Python (onnxruntime.InferenceSession) and C++
(clob_py.Scorer); scores 1000 random feature vectors through both; asserts
numpy.testing.assert_allclose(rtol=1e-4, atol=1e-5).

If this test fires, the deployed C++ scorer is silently scoring differently
from the Python training-time predictions. That breaks the entire ML stack —
backtest results (Python) don't match production (C++).
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from numpy.testing import assert_allclose

# Skip module entirely if the C++ pybind11 bindings haven't been built.
clob_py = pytest.importorskip("clob_py")

from model.schema import FEATURE_NAMES  # noqa: E402

pytestmark = pytest.mark.data

ARTIFACTS_DIR = Path(__file__).parents[2] / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.onnx"


def _have_model() -> bool:
    return MODEL_PATH.exists()


if not _have_model():
    pytest.skip("model/artifacts/model.onnx missing — run `uv run python -m model.train`",
                allow_module_level=True)


@pytest.fixture(scope="module")
def random_feature_vectors():
    """1000 synthetic vectors covering reasonable ranges per feature class."""
    rng = np.random.default_rng(seed=42)
    n = 1000
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    # microprice_g_dev: -ish hundreds-of-ticks
    X[:, 0] = rng.uniform(-200, 200, n)
    # imbalance_l1: [-1, 1]
    X[:, 1] = rng.uniform(-1, 1, n)
    # spread_ticks: [1, 30]
    X[:, 2] = rng.uniform(1, 30, n)
    # spread_zscore_200: standard normal
    X[:, 3] = rng.standard_normal(n)
    # ofi_50, ofi_200, mlofi: -500..500
    X[:, 4:7] = rng.uniform(-500, 500, (n, 3))
    # signed_trade_flow_50, tfi_50: bounded
    X[:, 7] = rng.uniform(-500, 500, n)
    X[:, 8] = rng.uniform(-1, 1, n)
    # realized_vol_200: positive
    X[:, 9] = rng.uniform(0, 100, n)
    # queue_depletion_bid/ask: standard normal
    X[:, 10:12] = rng.standard_normal((n, 2))
    # Ticker one-hots: pick one randomly per row
    X[:, 12:17] = 0.0
    chosen = rng.integers(0, 5, n)
    for i in range(n):
        X[i, 12 + chosen[i]] = 1.0
    # warm flags: 1
    X[:, 17:19] = 1.0
    return X


@pytest.fixture(scope="module")
def py_session():
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def cpp_scorer():
    return clob_py.Scorer.load(str(MODEL_PATH))


def test_python_onnx_probs_match_cpp_scorer_probs(random_feature_vectors,
                                                    py_session, cpp_scorer):
    """probs_batch (P(Down), P(Stable), P(Up)) match within rtol=1e-4, atol=1e-5."""
    X = random_feature_vectors
    # Python via raw onnxruntime — same library the C++ Scorer wraps.
    py_probs = np.asarray(py_session.run(None, {"input": X})[1], dtype=np.float64)
    # C++ via pybind11.
    cpp_probs = cpp_scorer.probs_batch(X)
    assert py_probs.shape == cpp_probs.shape == (1000, 3)
    # ADR 0006 tolerance.
    assert_allclose(py_probs, cpp_probs, rtol=1e-4, atol=1e-5)


def test_python_onnx_score_matches_cpp_scorer_score(random_feature_vectors,
                                                      py_session, cpp_scorer):
    """score = P(Up) - P(Down), checked end-to-end via the Scorer API."""
    X = random_feature_vectors
    py_probs = np.asarray(py_session.run(None, {"input": X})[1], dtype=np.float64)
    py_score = py_probs[:, 2] - py_probs[:, 0]
    cpp_score = np.asarray(cpp_scorer.score_batch(X), dtype=np.float64)
    assert_allclose(py_score, cpp_score, rtol=1e-4, atol=1e-5)


def test_single_score_matches_batch_score(random_feature_vectors, cpp_scorer):
    """Scorer.score(single_vec) == Scorer.score_batch([single_vec])[0]."""
    X = random_feature_vectors[:50]
    batched = np.asarray(cpp_scorer.score_batch(X), dtype=np.float64)
    individual = np.array([cpp_scorer.score(X[i]) for i in range(50)])
    assert_allclose(batched, individual, rtol=1e-9, atol=1e-9)

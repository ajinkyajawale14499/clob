"""BookFixture — hand-crafted feature scenarios for behavioral tests.

Drop-in helper for `tests/python/test_model_behavior.py`. Provides a small
DSL for building specific microstructure regimes (heavy bid, balanced,
widened spread, stale book) without typing 19 features by hand.

Scaling: to add a new scenario, append to SCENARIOS dict. To add a feature,
extend `_features()` mapping in BookFixture.
"""

from dataclasses import dataclass, field

import numpy as np

from model.schema import ALL_TICKERS, FEATURE_NAMES

# Predefined microstructure regimes — each overrides only the features that
# matter for that scenario; everything else stays at the BookFixture default.
SCENARIOS: dict[str, dict[str, float]] = {
    "heavy_bid_imbalance": {
        "imbalance_l1": 0.95,
        "ofi_50": 500.0,
        "ofi_200": 1500.0,
        "tfi_50": 0.8,
        "signed_trade_flow_50": 400.0,
    },
    "heavy_ask_imbalance": {
        "imbalance_l1": -0.95,
        "ofi_50": -500.0,
        "ofi_200": -1500.0,
        "tfi_50": -0.8,
        "signed_trade_flow_50": -400.0,
    },
    "balanced_book": {},
    "widened_spread_only": {
        "spread_ticks": 20.0,
        "spread_zscore_200": 3.5,
    },
    "stale_book": {
        # Cold start — no warm flags.
        "realized_vol_200": 0.0,
        "queue_depletion_bid": 0.0,
        "queue_depletion_ask": 0.0,
        "is_warm_50": 0.0,
        "is_warm_200": 0.0,
    },
}


@dataclass
class BookFixture:
    overrides: dict[str, float] = field(default_factory=dict)
    ticker: str = "AAPL"

    def features(self) -> np.ndarray:
        """Build the 19-feature vector. Defaults: all zeros, except spread_ticks=1
        (one tick), warm_50=warm_200=1, and the chosen ticker one-hot set."""
        f: dict[str, float] = {n: 0.0 for n in FEATURE_NAMES}
        f["spread_ticks"] = 1.0
        f["is_warm_50"] = 1.0
        f["is_warm_200"] = 1.0
        f[f"ticker_{self.ticker}"] = 1.0
        f.update(self.overrides)
        return np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32)

    def with_(self, **overrides: float) -> "BookFixture":
        """Return a new fixture with additional overrides applied."""
        merged = {**self.overrides, **overrides}
        return BookFixture(overrides=merged, ticker=self.ticker)


def make_book(scenario: str, ticker: str = "AAPL") -> BookFixture:
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; choices: {list(SCENARIOS.keys())}")
    return BookFixture(overrides=dict(SCENARIOS[scenario]), ticker=ticker)


# Sanity at import time.
assert all(t in ALL_TICKERS for t in (["AAPL"])), "BookFixture default ticker missing"

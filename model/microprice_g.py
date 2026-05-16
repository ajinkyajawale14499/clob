"""Stoikov G(I, S) microprice — offline fixed-point fit + JSON dump for C++ runtime.

Algorithm (Stoikov 2018, "The Micro-Price"):
    State: discretized (imbalance, spread) pair.
    Transition matrix P[(I, S) -> (I', S')] estimated from observed event sequences.
    Payoff R[(I, S)] = E[Δmid | starting from (I, S)].
    Fixed point: G = R + P · G (iterate to convergence).

At inference, the C++ matcher loads the JSON LUT and does an O(1) lookup keyed
on (imbalance_bucket, spread_bucket). No model inference needed — pure table.

The output `G(I, S)` is the price ADJUSTMENT relative to mid, in LOBSTER native
price-int units. Add to mid to get the martingale-fair price.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class MicropriceLut:
    """Stoikov G(I, S) table.

    table[i, j] = G adjustment for imbalance bucket i, spread bucket j.
    imbalance buckets span [-1, +1] uniformly; spread buckets are 1-tick steps.
    """
    table: np.ndarray           # shape (n_imb, n_sp), dtype float64
    n_imb: int
    n_sp: int
    tick_size: int              # LOBSTER native = 100

    @classmethod
    def load(cls, path: str | Path) -> "MicropriceLut":
        data = json.loads(Path(path).read_text())
        table = np.array(data["table"], dtype=np.float64)
        return cls(table=table,
                   n_imb=data["n_imbalance_buckets"],
                   n_sp=data["n_spread_buckets"],
                   tick_size=data["tick_size"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "n_imbalance_buckets": self.n_imb,
            "n_spread_buckets": self.n_sp,
            "tick_size": self.tick_size,
            "table": self.table.tolist(),
        }, indent=2))

    def _imbalance_bucket(self, imbalance: float) -> int:
        # imbalance ∈ [-1, +1] -> bucket ∈ [0, n_imb-1]
        i = int((imbalance + 1.0) / 2.0 * self.n_imb)
        return max(0, min(self.n_imb - 1, i))

    def _spread_bucket(self, spread_ticks: int) -> int:
        # spread_ticks: 1 -> bucket 0, 2 -> bucket 1, ..., >n_sp -> last
        b = int(spread_ticks) - 1
        return max(0, min(self.n_sp - 1, b))

    def lookup(self, imbalance: float, spread_ticks: int) -> float:
        i = self._imbalance_bucket(imbalance)
        s = self._spread_bucket(spread_ticks)
        return float(self.table[i, s])

    def lookup_vec(self, imbalances, spread_ticks_list) -> np.ndarray:
        imb_arr = np.asarray(imbalances)
        sp_arr = np.asarray(spread_ticks_list)
        i_idx = np.clip(((imb_arr + 1.0) / 2.0 * self.n_imb).astype(int),
                        0, self.n_imb - 1)
        s_idx = np.clip(sp_arr.astype(int) - 1, 0, self.n_sp - 1)
        return self.table[i_idx, s_idx]


def fit_microprice_g(tob: pl.DataFrame, *, n_imbalance_buckets: int,
                     n_spread_buckets: int, tick_size: int,
                     max_iter: int = 200, tol: float = 1e-6) -> MicropriceLut:
    """Fit Stoikov G(I, S) by fixed-point iteration on the discretized state space.

    Args:
        tob: DataFrame with bid_price_l1, ask_price_l1, bid_size_l1, ask_size_l1.
        n_imbalance_buckets: number of buckets across imbalance ∈ [-1, +1].
        n_spread_buckets: number of buckets across spread (1-tick steps).
        tick_size: LOBSTER native price-int units per tick (typically 100).
        max_iter: cap on fixed-point iterations.
        tol: convergence tolerance on G's max abs change per iteration.
    """
    bid_px = tob["bid_price_l1"].to_numpy()
    ask_px = tob["ask_price_l1"].to_numpy()
    bid_sz = tob["bid_size_l1"].to_numpy()
    ask_sz = tob["ask_size_l1"].to_numpy()

    mid = (bid_px + ask_px) / 2.0
    spread = (ask_px - bid_px) // tick_size  # spread in ticks (integer)
    total = bid_sz + ask_sz
    # Avoid div-by-zero: when total==0, imbalance treated as 0 (rare edge).
    safe_total = np.where(total == 0, 1, total)
    imbalance = (bid_sz - ask_sz) / safe_total

    # Discretize.
    i_buckets = np.clip(((imbalance + 1.0) / 2.0 * n_imbalance_buckets).astype(int),
                        0, n_imbalance_buckets - 1)
    s_buckets = np.clip(spread.astype(int) - 1, 0, n_spread_buckets - 1)

    state = i_buckets * n_spread_buckets + s_buckets  # flat state index
    n_states = n_imbalance_buckets * n_spread_buckets

    # Per-state transition counts + payoff sums.
    # We use successive state pairs (t -> t+1).
    next_state = np.roll(state, -1)
    next_mid = np.roll(mid, -1)
    delta_mid = next_mid - mid

    P = np.zeros((n_states, n_states), dtype=np.float64)
    R = np.zeros(n_states, dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)

    # Skip last row (no t+1 sample).
    for k in range(len(state) - 1):
        s_t = state[k]
        s_next = next_state[k]
        P[s_t, s_next] += 1
        R[s_t] += delta_mid[k]
        counts[s_t] += 1

    # Normalize.
    P_norm = np.zeros_like(P)
    for s in range(n_states):
        if counts[s] > 0:
            P_norm[s] = P[s] / counts[s]
            R[s] = R[s] / counts[s]
    # Unseen states stay zero (G = 0 by default; rare).

    # Fixed-point iteration: G_{k+1} = R + P · G_k.
    G = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iter):
        G_next = R + P_norm @ G
        if np.max(np.abs(G_next - G)) < tol:
            G = G_next
            break
        G = G_next

    # Reshape back to (n_imb, n_sp).
    table = G.reshape(n_imbalance_buckets, n_spread_buckets)
    return MicropriceLut(
        table=table,
        n_imb=n_imbalance_buckets,
        n_sp=n_spread_buckets,
        tick_size=tick_size,
    )

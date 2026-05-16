"""Backtest driver: loops LOBSTER events through Engine + Policy."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import clob_py
import numpy as np

from backtest.metrics import BacktestResult
from backtest.policies import LOBSTER_MAX_ORDER_ID, BasePolicy, MLAwareMaker, NaiveMaker
from backtest.stream import stream_lobster_events
from data.ingestion.lobster_message import load_lobster_messages
from data.ingestion.lobster_orderbook import (
    join_messages_orderbook,
    load_lobster_orderbook,
)
from data.tob.unified import lobster_to_tob

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).parent.parent / "model" / "artifacts"


def _mid_array(ticker: str) -> np.ndarray:
    """Pre-compute the mid-price per LOBSTER event row, for markout lookups."""
    msg_path = next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv"))
    book_path = next(SAMPLE_DIR.glob(f"{ticker}_*_orderbook_*.csv"))
    joined = join_messages_orderbook(
        load_lobster_messages(msg_path),
        load_lobster_orderbook(book_path, n_levels=10),
    )
    tob = lobster_to_tob(joined)
    return ((tob["bid_price_l1"] + tob["ask_price_l1"]) // 2).to_numpy()


def run_backtest(
    ticker: str,
    policy_name: Literal["naive", "ml_aware"],
    *,
    model_path: Path | None = ARTIFACTS_DIR / "model.onnx",
    lut_path: Path | None = ARTIFACTS_DIR / "microprice_g.json",
    threshold: float = 0.15,
    max_events: int | None = None,
) -> BacktestResult:
    """Replay LOBSTER for `ticker` through the matcher + policy; return metrics input."""
    msg_path = next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv"))

    # Engine + Scorer wiring: naive policy doesn't need a scorer; ml_aware does.
    captured_scores: list[tuple[int, float]] = []
    if policy_name == "ml_aware":
        if model_path is None or lut_path is None:
            raise ValueError("ml_aware policy requires model_path + lut_path")
        scorer = clob_py.Scorer.load(str(model_path))
        lut = clob_py.MicropriceLut.load(str(lut_path))
        engine = clob_py.Engine(
            scorer=scorer,
            score_sink=lambda oid, s: captured_scores.append((oid, s)),
            ticker=ticker,
            lut=lut,
        )
    else:
        engine = clob_py.Engine()  # no scoring path

    # Construct policy AFTER engine — both share the same Engine.
    policy: BasePolicy
    if policy_name == "naive":
        policy = NaiveMaker(engine)
    elif policy_name == "ml_aware":
        policy = MLAwareMaker(engine, threshold=threshold)
    else:
        raise ValueError(f"unknown policy {policy_name!r}")

    mids = _mid_array(ticker)
    last_score = 0.0  # default for naive path (no scoring)

    # Replay loop.
    for event_index, ev in enumerate(stream_lobster_events(msg_path)):
        if max_events is not None and event_index >= max_events:
            break
        _ts, op, *args = ev
        fills = []
        if op == "add_limit":
            order_id, side, price, size = args
            # Skip if the order_id collides with our policy's reserved range
            # (extremely unlikely — LOBSTER IDs cap around 32-bit).
            if order_id >= LOBSTER_MAX_ORDER_ID:
                continue
            fills = engine.add_limit(order_id, side, price, size)
        elif op == "cancel":
            (order_id,) = args
            if order_id >= LOBSTER_MAX_ORDER_ID:
                continue
            engine.cancel(order_id)

        # If a score was captured for this event, use it; else keep last (or 0).
        if captured_scores:
            _, last_score = captured_scores[-1]

        # Policy reacts: may post bid/ask quotes.
        policy.on_event(ev, last_score, fills, event_index)

    return BacktestResult(
        ticker=ticker,
        policy_name=policy.state.name,
        quotes_posted=policy.state.quotes_posted,
        fills=list(policy.state.recently_filled),
        mids=mids,
    )


def run_backtest_lazy_mids(
    ticker: str,
    policy_name: Literal["naive", "ml_aware"],
    *,
    model_path: Path | None = ARTIFACTS_DIR / "model.onnx",
    lut_path: Path | None = ARTIFACTS_DIR / "microprice_g.json",
    threshold: float = 0.15,
    max_events: int | None = None,
) -> BacktestResult:
    """Same as run_backtest but pre-loads mids inside — convenience for tests."""
    return run_backtest(
        ticker, policy_name,
        model_path=model_path, lut_path=lut_path,
        threshold=threshold, max_events=max_events,
    )

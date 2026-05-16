"""Backtest policies — naive vs ML-aware passive market making.

Both policies post bid+ask at L1 (1 contract each) and re-quote on every
incoming event. They differ only in WHEN to suppress a side:

    NaiveMaker:     always posts both bid + ask
    MLAwareMaker:   suppresses bid if score > +threshold (model expects up)
                    suppresses ask if score < -threshold (model expects down)

Order IDs >= POLICY_ORDER_ID_START so they never collide with LOBSTER IDs
(LOBSTER's are int32 max). Records markouts when policy quotes fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import clob_py

POLICY_ORDER_ID_START = 10_000_000_000
LOBSTER_MAX_ORDER_ID = 9_000_000_000  # LOBSTER int32 ids; way below POLICY base


@dataclass
class PolicyFill:
    """Records a fill of one of the policy's own quotes."""
    order_id: int
    fill_price: int       # in LOBSTER price-int units
    quantity: int
    side_sign: int        # +1 if policy was on bid side, -1 if ask
    event_index: int      # absolute index into the LOBSTER event stream


@dataclass
class PolicyState:
    """Accumulates a policy's per-run trades + posted quote count."""
    name: str
    open_quotes: dict[int, dict[str, Any]] = field(default_factory=dict)
    quotes_posted: int = 0
    recently_filled: list[PolicyFill] = field(default_factory=list)
    _next_id: int = POLICY_ORDER_ID_START


class BasePolicy:
    """Common state + book-keeping for fills."""

    def __init__(self, engine: clob_py.Engine, name: str):
        self.engine = engine
        self.state = PolicyState(name=name)

    def _next_oid(self) -> int:
        self.state._next_id += 1
        return self.state._next_id

    def _record_fills(self, fills: list, current_index: int) -> None:
        """If any fill involved one of our open quotes, record it as a PolicyFill."""
        for f in fills:
            for oid in (f.taker_id, f.maker_id):
                if oid in self.state.open_quotes:
                    q = self.state.open_quotes.pop(oid)
                    self.state.recently_filled.append(PolicyFill(
                        order_id=oid,
                        fill_price=f.price,
                        quantity=f.quantity,
                        side_sign=(1 if q["side"] == clob_py.Side.Bid else -1),
                        event_index=current_index,
                    ))

    def _quote(self, side: clob_py.Side, price: int, qty: int = 1) -> int:
        """Place a 1-contract policy quote; return its order id."""
        oid = self._next_oid()
        fills = self.engine.add_limit(oid, side, price, qty)
        # If our quote crossed (immediate match), record now.
        # The engine returns fills with taker_id = our oid.
        # Note: this is rare for L1 ± 1 tick quotes but possible.
        for f in fills:
            if f.taker_id == oid:
                self.state.recently_filled.append(PolicyFill(
                    order_id=oid,
                    fill_price=f.price,
                    quantity=f.quantity,
                    side_sign=(1 if side == clob_py.Side.Bid else -1),
                    event_index=-1,  # immediate
                ))
        # If it rested (no fill or partial), track it as open.
        # Total qty posted is `qty`; subtract filled portion.
        filled_qty = sum(f.quantity for f in fills if f.taker_id == oid)
        if filled_qty < qty:
            self.state.open_quotes[oid] = {"side": side, "price": price,
                                            "qty_resting": qty - filled_qty}
        self.state.quotes_posted += 1
        return oid

    def on_event(self, ev: tuple, score: float, fills: list, event_index: int) -> None:
        """Override in subclasses."""
        raise NotImplementedError


class NaiveMaker(BasePolicy):
    """Always quote bid + ask at current L1."""

    def __init__(self, engine: clob_py.Engine):
        super().__init__(engine, "naive")

    def on_event(self, ev: tuple, score: float, fills: list, event_index: int) -> None:
        # First, record any of our quotes that just filled on this event.
        self._record_fills(fills, event_index)

        bb = self.engine.book().best_bid()
        ba = self.engine.book().best_ask()
        if bb is None or ba is None:
            return  # nothing to quote against

        # Repost — cancel old quotes is intentionally skipped to avoid
        # explosive cancel volume. The simple v1 model just keeps adding
        # 1-contract quotes; quotes_posted captures total flow.
        self._quote(clob_py.Side.Bid, bb)
        self._quote(clob_py.Side.Ask, ba)


class MLAwareMaker(BasePolicy):
    """Same as Naive, but suppresses a side when the model says adverse."""

    def __init__(self, engine: clob_py.Engine, threshold: float = 0.15):
        super().__init__(engine, "ml_aware")
        # Score = P(Up) - P(Down) ∈ [-1, +1].
        # If score > +threshold: model thinks mid is heading UP -> suppress BID
        #   (we'd be picked off; our resting bid sells low to an informed buyer).
        # If score < -threshold: suppress ASK.
        self.threshold = threshold

    def on_event(self, ev: tuple, score: float, fills: list, event_index: int) -> None:
        self._record_fills(fills, event_index)
        bb = self.engine.book().best_bid()
        ba = self.engine.book().best_ask()
        if bb is None or ba is None:
            return
        # Score above threshold means likely up move -> suppress bid.
        # Score below -threshold means likely down move -> suppress ask.
        post_bid = score <= self.threshold
        post_ask = score >= -self.threshold
        if post_bid:
            self._quote(clob_py.Side.Bid, bb)
        if post_ask:
            self._quote(clob_py.Side.Ask, ba)

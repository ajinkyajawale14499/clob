"""Generate format-correct synthetic LOBSTER fixtures.

LOBSTER's sample download moved behind a React SPA — direct curl returns HTML.
Until the real samples land in data/raw/ (manual browser step), this script
populates data/raw/ with synthetic files that follow the LOBSTER on-disk format
exactly. They exercise every parser code path:

  - message file: 6 columns, no header (time, event_type, order_id, size, price, direction)
  - orderbook file: 4*N columns, no header (ask_price_i, ask_size_i, bid_price_i, bid_size_i)
  - 1:1 row correspondence between the two files
  - never-crossed book (bid_l1 < ask_l1 for every row)
  - monotonically non-decreasing timestamps
  - prices in $0.0001 ticks, sides in {-1, +1}

Seed is fixed -> bit-identical output every run.

Usage:
    uv run python scripts/generate_synth_lobster.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Match LOBSTER's real naming: <TICKER>_<DATE>_<START_NS>_<END_NS>_message_<LEVELS>.csv
TICKER = "AAPL"
DATE = "2012-06-21"
START_S = 34200  # 09:30:00 — market open
END_S = 34800    # 09:40:00 — 10 min slice
N_LEVELS = 10
N_EVENTS = 250   # need >100 for test_load_lobster_messages_returns_polars_frame
SEED = 42

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MSG_PATH = OUT_DIR / f"{TICKER}_{DATE}_{START_S}000_{END_S}000_message_{N_LEVELS}.csv"
BOOK_PATH = OUT_DIR / f"{TICKER}_{DATE}_{START_S}000_{END_S}000_orderbook_{N_LEVELS}.csv"

# Event types we emit. LOBSTER uses 1=new limit, 2=partial cancel, 3=full cancel,
# 4=visible exec, 5=hidden exec, 6=cross, 7=halt. We emit a realistic mix of 1/3/4.
EVENT_KINDS = (1, 1, 1, 1, 1, 3, 3, 4)  # weight new-limits, mix in cancels + execs

TICK = 100  # $0.01 minimum price increment in $0.0001 units


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Seed a 10-level book around mid = $565.00.
    # ask levels go UP from mid; bid levels go DOWN.
    mid = 5_650_000
    ask = [mid + (i + 1) * TICK for i in range(N_LEVELS)]
    bid = [mid - (i + 1) * TICK for i in range(N_LEVELS)]
    ask_sz = [rng.randint(100, 500) for _ in range(N_LEVELS)]
    bid_sz = [rng.randint(100, 500) for _ in range(N_LEVELS)]

    t = float(START_S)
    next_order_id = 1
    live_orders: list[int] = []  # ids we've seen, for cancel/exec targeting

    msg_rows: list[tuple] = []
    book_rows: list[list[int]] = []

    for _ in range(N_EVENTS):
        # Advance time by 1-300ms.
        t += rng.uniform(0.001, 0.3)
        kind = rng.choice(EVENT_KINDS)

        if kind == 1 or not live_orders:
            # New limit on a random side at L1 (most realistic — new liquidity at the touch).
            side = rng.choice((-1, 1))
            order_id = next_order_id
            next_order_id += 1
            size = rng.randint(10, 200)
            if side == 1:  # bid
                price = bid[0] + (TICK if rng.random() < 0.3 else 0)
                if price >= ask[0]:
                    price = ask[0] - TICK  # never cross
                bid_sz[0] += size
                bid[0] = price
            else:  # ask
                price = ask[0] - (TICK if rng.random() < 0.3 else 0)
                if price <= bid[0]:
                    price = bid[0] + TICK
                ask_sz[0] += size
                ask[0] = price
            msg_rows.append((round(t, 9), 1, order_id, size, price, side))
            live_orders.append(order_id)

        elif kind == 3 and live_orders:
            # Full cancel of a previously live order.
            order_id = rng.choice(live_orders)
            live_orders.remove(order_id)
            side = rng.choice((-1, 1))
            size = rng.randint(10, 100)
            price = bid[0] if side == 1 else ask[0]
            if side == 1:
                bid_sz[0] = max(50, bid_sz[0] - size)
            else:
                ask_sz[0] = max(50, ask_sz[0] - size)
            msg_rows.append((round(t, 9), 3, order_id, size, price, side))

        elif kind == 4 and live_orders:
            # Visible execution against the best opposite — taker hits the L1.
            taker_side = rng.choice((-1, 1))
            maker_id = rng.choice(live_orders)
            size = rng.randint(10, 80)
            if taker_side == 1:
                price = ask[0]
                ask_sz[0] = max(50, ask_sz[0] - size)
            else:
                price = bid[0]
                bid_sz[0] = max(50, bid_sz[0] - size)
            # LOBSTER convention: direction is the OPPOSITE side's sign for execs.
            msg_rows.append((round(t, 9), 4, maker_id, size, price, -taker_side))

        # Inviolable: never cross the touch.
        if bid[0] >= ask[0]:
            ask[0] = bid[0] + TICK

        # Snapshot the full 10-level book at this event.
        snapshot: list[int] = []
        for i in range(N_LEVELS):
            snapshot.extend([ask[i], ask_sz[i], bid[i], bid_sz[i]])
        book_rows.append(snapshot)

    with open(MSG_PATH, "w", newline="") as f:
        w = csv.writer(f)
        for row in msg_rows:
            w.writerow(row)

    with open(BOOK_PATH, "w", newline="") as f:
        w = csv.writer(f)
        for row in book_rows:
            w.writerow(row)

    print(f"Wrote {len(msg_rows):>4} message rows -> {MSG_PATH.name}")
    print(f"Wrote {len(book_rows):>4} orderbook rows -> {BOOK_PATH.name}")


if __name__ == "__main__":
    main()

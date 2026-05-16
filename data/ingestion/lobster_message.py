"""LOBSTER message-file ingestion.

LOBSTER doc: https://lobsterdata.com/info/DataStructure.php

Output schema:
    ts_ns      Datetime("ns")  -- LOBSTER seconds-since-midnight -> ns since unix epoch
    event_type Int8            -- 1=new limit, 2=partial cancel, 3=full cancel,
                               --  4=visible exec, 5=hidden exec, 6=cross, 7=halt
    order_id   Int64
    side       Int8            -- LOBSTER: 1=bid, -1=ask
    size       Int64
    price      Int64           -- LOBSTER native: 1 tick = $0.0001
"""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

LOBSTER_DATE = date(2012, 6, 21)  # the sample's trading date
LOBSTER_MESSAGE_SCHEMA = {
    "time": pl.Float64,
    "event_type": pl.Int8,
    "order_id": pl.Int64,
    "size": pl.Int64,
    "price": pl.Int64,
    "direction": pl.Int8,
}


def load_lobster_messages(path: Path) -> pl.DataFrame:
    """Parse a LOBSTER `*_message_*.csv` (plain CSV) into a normalized frame."""
    raw = pl.read_csv(
        path,
        has_header=False,
        new_columns=list(LOBSTER_MESSAGE_SCHEMA.keys()),
        schema_overrides=LOBSTER_MESSAGE_SCHEMA,
    )
    midnight_ns = int(
        datetime.combine(LOBSTER_DATE, datetime.min.time(), tzinfo=UTC).timestamp()
        * 1e9
    )
    return raw.select(
        (midnight_ns + (pl.col("time") * 1_000_000_000).cast(pl.Int64))
        .cast(pl.Datetime("ns"))
        .alias("ts_ns"),
        pl.col("event_type"),
        pl.col("order_id"),
        pl.col("direction").alias("side"),
        pl.col("size"),
        pl.col("price"),
    )

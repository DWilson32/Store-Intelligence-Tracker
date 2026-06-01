"""
load_pos.py — Load POS transactions into the Store Intelligence database.

Supports two input formats:
  1. Pipeline format (simple):  store_id, transaction_id, timestamp, basket_value_inr
  2. Purplle raw format:        order_id, order_date, order_time, store_id, NMV, ...

Usage:
    python scripts/load_pos.py --csv data/pos_transactions.csv
    python scripts/load_pos.py --csv data/Brigade_Bangalore_10_April_26.csv --raw
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////data/store_intelligence.db")

import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import POSTransaction, Base


def parse_raw_purplle(csv_path: str) -> list[dict]:
    """Parse Purplle raw sales CSV into pipeline-format records."""
    df = pd.read_csv(csv_path)

    orders = df.groupby('order_id').agg(
        store_id=('store_id', 'first'),
        order_date=('order_date', 'first'),
        order_time=('order_time', 'first'),
        NMV=('NMV', 'sum'),
    ).reset_index()

    records = []
    for _, row in orders.iterrows():
        # Convert DD-MM-YYYY HH:MM:SS → ISO-8601
        d = str(row['order_date'])   # 10-04-2026
        t = str(row['order_time'])   # 16:55:36
        try:
            day, month, year = d.split('-')
            timestamp = f"{year}-{month}-{day}T{t}Z"
        except Exception:
            timestamp = f"2026-01-01T00:00:00Z"

        records.append({
            "store_id":       str(row['store_id']),
            "transaction_id": f"TXN_{row['order_id']}",
            "timestamp":      timestamp,
            "basket_value":   round(float(row['NMV']), 2),
        })
    return records


def parse_pipeline_format(csv_path: str) -> list[dict]:
    """Parse simple pipeline-format CSV."""
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "store_id":       str(row['store_id']),
            "transaction_id": str(row['transaction_id']),
            "timestamp":      str(row['timestamp']),
            "basket_value":   round(float(row['basket_value_inr']), 2),
        })
    return records


async def load(csv_path: str, raw: bool):
    db_url = os.environ["DATABASE_URL"]
    engine = create_async_engine(db_url, connect_args={"check_same_thread": False})
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    records = parse_raw_purplle(csv_path) if raw else parse_pipeline_format(csv_path)

    async with Session() as session:
        await session.execute(
            sqlite_insert(POSTransaction).prefix_with("OR IGNORE"),
            records,
        )
        await session.commit()

    print(f"✓ Loaded {len(records)} POS transactions from {csv_path}")
    for r in records[:5]:
        print(f"  {r['transaction_id']}  {r['timestamp']}  ₹{r['basket_value']}")
    if len(records) > 5:
        print(f"  ... and {len(records)-5} more")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--raw", action="store_true",
                        help="Input is Purplle raw sales format (not pipeline format)")
    args = parser.parse_args()
    asyncio.run(load(args.csv, args.raw))

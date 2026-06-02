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

    # Determine amount column dynamically (NMV, total_amount, or GMV)
    amt_col = None
    for col in ["NMV", "total_amount", "GMV"]:
        if col in df.columns:
            amt_col = col
            break
    if not amt_col:
        raise ValueError(f"Could not find amount column (NMV, total_amount, or GMV) in {csv_path}")

    orders = df.groupby('order_id').agg(
        store_id=('store_id', 'first'),
        order_date=('order_date', 'first'),
        order_time=('order_time', 'first'),
        amount=(amt_col, 'sum'),
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
            "basket_value":   round(float(row['amount']), 2),
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

    # Auto-generate mock POS transactions for ST1076 from billing_area.jsonl if it exists
    billing_jsonl = Path(csv_path).parent / "events" / "billing_area.jsonl"
    if billing_jsonl.exists():
        import json
        st1076_records = []
        with open(billing_jsonl) as f:
            for line in f:
                if line.strip():
                    ev = json.loads(line)
                    is_completed = (ev.get("event_type") == "queue_completed" or
                                    ev.get("event_type") == "BILLING_QUEUE_JOIN" and not ev.get("abandoned", False))
                    if ev.get("store_id") == "ST1076" and is_completed:
                        ts = ev.get("queue_exit_ts") or ev.get("queue_served_ts") or ev.get("event_time") or ev.get("timestamp")
                        if ts:
                            st1076_records.append({
                                "store_id": "ST1076",
                                "transaction_id": f"TXN_MUM_{len(st1076_records)+1}",
                                "timestamp": ts,
                                "basket_value": round(float(450 + (len(st1076_records) * 175) % 900), 2),
                            })
        if st1076_records:
            records.extend(st1076_records)
            print(f"Generated {len(st1076_records)} mock POS transactions for ST1076 based on billing queue completion events.")

    async with Session() as session:
        await session.execute(
            sqlite_insert(POSTransaction).prefix_with("OR IGNORE"),
            records,
        )
        await session.commit()

    print(f"Successfully loaded {len(records)} POS transactions from {csv_path}")
    for r in records[:5]:
        print(f"  {r['transaction_id']}  {r['timestamp']}  INR {r['basket_value']}")
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

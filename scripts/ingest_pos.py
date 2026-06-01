"""Ingest POS transaction CSV into the database."""
import csv
import os
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "pos_transactions.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "store_intelligence.db")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Ensure table exists
c.execute("""
    CREATE TABLE IF NOT EXISTS pos_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_id TEXT NOT NULL,
        transaction_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        basket_value REAL NOT NULL
    )
""")

inserted = 0
skipped = 0

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            c.execute(
                "INSERT OR IGNORE INTO pos_transactions (store_id, transaction_id, timestamp, basket_value) VALUES (?, ?, ?, ?)",
                (row["store_id"], row["transaction_id"], row["timestamp"], float(row["basket_value_inr"])),
            )
            if c.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  Error: {e} — row: {row}")
            skipped += 1

conn.commit()
conn.close()

print(f"POS transactions: {inserted} inserted, {skipped} skipped (duplicates)")

# Verify
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM pos_transactions")
print(f"Total POS records in DB: {c.fetchone()[0]}")
c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM pos_transactions")
row = c.fetchone()
print(f"Time range: {row[0]} → {row[1]}")
c.execute("SELECT SUM(basket_value), AVG(basket_value) FROM pos_transactions")
row = c.fetchone()
print(f"Total revenue: ₹{row[0]:,.2f} | Avg basket: ₹{row[1]:,.2f}")
conn.close()

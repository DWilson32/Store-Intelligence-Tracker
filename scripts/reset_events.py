import sqlite3
import os
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")

# Try to parse DB path from DATABASE_URL if set
db_url = os.environ.get("DATABASE_URL", "")
if db_url.startswith("sqlite+aiosqlite:///"):
    DB_PATH = db_url.replace("sqlite+aiosqlite:///", "")
elif db_url.startswith("sqlite:///"):
    DB_PATH = db_url.replace("sqlite:///", "")
else:
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "store_intelligence.db")

# Clear the SQLite table
if os.path.exists(DB_PATH):
    print(f"Connecting to database {DB_PATH} to clear events...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    print(f"Deleted all records from 'events' table. Total count: {cursor.execute('SELECT COUNT(*) FROM events').fetchone()[0]}")
    conn.close()
else:
    print("Database file not found.")

# Clear any jsonl and pkl files in data/events/
for ext in ("*.jsonl", "*.pkl"):
    files = glob.glob(os.path.join(EVENTS_DIR, ext))
    for f in files:
        try:
            os.remove(f)
            print(f"Removed old file: {os.path.basename(f)}")
        except Exception as e:
            print(f"Error removing {f}: {e}")

print("Reset complete!")

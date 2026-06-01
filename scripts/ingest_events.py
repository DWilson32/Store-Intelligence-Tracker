"""Ingest all event JSONL files into the API."""
import json, os, httpx

API = "http://localhost:8000"
EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")

accepted = 0
for fname in sorted(os.listdir(EVENTS_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    path = os.path.join(EVENTS_DIR, fname)
    events = []
    with open(path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    if not events:
        print(f"  {fname}: 0 events (skipped)")
        continue
    print(f"  {fname}: {len(events)} events")
    for i in range(0, len(events), 500):
        batch = events[i:i+500]
        r = httpx.post(f"{API}/events/ingest", json={"events": batch}, timeout=30)
        batch_accepted = r.json().get("accepted", 0)
        accepted += batch_accepted
        print(f"    Batch {i//500+1}: accepted={batch_accepted}")

print(f"\nTotal accepted: {accepted}")

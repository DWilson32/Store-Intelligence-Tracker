"""
run_detection.py — Batch-process all CCTV clips through the full pipeline.

Uses pipeline/detect.py (StoreDetector) with:
  • YOLO detection
  • Multi-camera Re-ID tracker (torchreid / osnet_x0_25)
  • Zone classification from store_layout.json
  • Staff-uniform colour heuristic
  • Entry/Exit line-crossing logic

After processing, ingests all events into the API via /events/ingest.
"""

import os
import sys
import json
import glob
import httpx

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.detect import StoreDetector
from pipeline.tracker import MultiCameraTracker

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
STORE_ID = "ST1008"
CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
LAYOUT_PATH = os.path.join(PROJECT_ROOT, "data", "store_layout.json")
START_TIME = "2026-04-10T20:10:00Z"

os.makedirs(EVENTS_DIR, exist_ok=True)

CAMERA_MAP = {
    "ST1008_ground_entry.mp4":   "CAM_ENTRY_01",
    "ST1008_ground_floor.mp4":   "CAM_FLOOR_01",
    "ST1008_ground_billing.mp4": "CAM_BILLING_01",
}

# Explicit order of processing: entry first to populate Re-ID gallery,
# then floor, then billing.
CLIPS_TO_PROCESS = [
    "ST1008_ground_entry.mp4",
    "ST1008_ground_floor.mp4",
    "ST1008_ground_billing.mp4",
]

# ── Process all clips ─────────────────────────────────────────────────────────

# Initialize a single shared tracker for cross-camera Re-ID
shared_tracker = MultiCameraTracker(store_id=STORE_ID)

total_events = 0
event_files = []

# List all clips to print skipped ones
for fname in sorted(os.listdir(CLIPS_DIR)):
    if fname.endswith(".mp4") and fname not in CAMERA_MAP:
        print(f"Skipping clip {fname} (not customer-facing/excluded)")

# Process mapped clips in explicit sequence
for fname in CLIPS_TO_PROCESS:
    if fname not in CAMERA_MAP:
        continue

    camera_id = CAMERA_MAP[fname]
    clip_path = os.path.join(CLIPS_DIR, fname)
    output_path = os.path.join(EVENTS_DIR, fname.replace(".mp4", ".jsonl"))
    event_files.append(output_path)

    print(f"\n{'='*60}")
    print(f"  Clip: {fname}  |  Camera: {camera_id}")
    print(f"{'='*60}")

    detector = StoreDetector(
        clip_path=clip_path,
        store_id=STORE_ID,
        camera_id=camera_id,
        layout_path=LAYOUT_PATH,
        output_path=output_path,
        clip_start_time=START_TIME,
        tracker=shared_tracker,
    )
    n = detector.process()
    total_events += n
    print(f"  → {n} events written to {os.path.basename(output_path)}")

print(f"\n{'='*60}")
print(f"  All clips processed. Total events: {total_events}")
print(f"{'='*60}")

# ── Ingest into API ───────────────────────────────────────────────────────────

print("\nIngesting events into API...")
accepted = 0

for ef in event_files:
    if not os.path.exists(ef):
        continue
    events = []
    with open(ef) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print(f"  {os.path.basename(ef)}: {len(events)} events")

    for i in range(0, len(events), 500):
        batch = events[i : i + 500]
        try:
            r = httpx.post(
                f"{API_URL}/events/ingest",
                json={"events": batch},
                timeout=30,
            )
            batch_accepted = r.json().get("accepted", 0)
            accepted += batch_accepted
        except Exception as exc:
            print(f"    ⚠ Batch {i//500+1} failed: {exc}")

print(f"\nDone! Total accepted: {accepted}")

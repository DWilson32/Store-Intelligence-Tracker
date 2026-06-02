"""
run_detection.py — Batch-process all CCTV clips through the full pipeline.

Supports both ST1008 (Brigade Road) and ST1076 (Mumbai).
Uses pipeline/detect.py (StoreDetector) with YOLO detection, Re-ID tracking,
and layout-based zone assignment.
After processing, automatically ingests all generated events into the API.
"""

import argparse
import os
import sys
import json
import httpx

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.detect import StoreDetector
from pipeline.tracker import MultiCameraTracker

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
LAYOUT_PATH = os.path.join(PROJECT_ROOT, "data", "store_layout.json")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")

os.makedirs(EVENTS_DIR, exist_ok=True)

STORE_CONFIGS = {
    "ST1008": {
        "clips_dir": os.path.join(PROJECT_ROOT, "data", "clips"),
        "start_time": "2026-04-10T20:10:00Z",
        "camera_start_times": {
            "ST1008_ground_entry.mp4": "2026-04-10T20:10:00Z",
            "ST1008_ground_floor.mp4": "2026-04-10T20:10:00Z",
            "ST1008_ground_billing.mp4": "2026-04-10T20:10:00Z",
        },
        "camera_map": {
            "ST1008_ground_entry.mp4": "CAM_ENTRY_01",
            "ST1008_ground_floor.mp4": "CAM_FLOOR_01",
            "ST1008_ground_billing.mp4": "CAM_BILLING_01",
        },
        "clip_order": ["ST1008_ground_entry.mp4", "ST1008_ground_floor.mp4", "ST1008_ground_billing.mp4"],
        "output_format": "internal",
    },
    "ST1076": {
        "clips_dir": os.path.join(PROJECT_ROOT, "data", "clips", "Store 2"),
        "start_time": "2026-03-08T18:10:00Z",
        "camera_start_times": {
            "entry 1.mp4": "2026-03-08T18:10:00Z",
            "entry 2.mp4": "2026-03-08T18:10:00Z",
            "zone.mp4": "2026-03-08T18:10:00Z",
            "billing_area.mp4": "2026-03-08T18:27:00Z",
        },
        "camera_map": {
            "entry 1.mp4": "CAM_ENTRY_01",
            "entry 2.mp4": "CAM_ENTRY_02",
            "zone.mp4": "CAM_ZONE_01",
            "billing_area.mp4": "CAM_BILLING_01",
        },
        "clip_order": ["entry 1.mp4", "entry 2.mp4", "zone.mp4", "billing_area.mp4"],
        "output_format": "sample",
    },
}

def run_store_pipeline(store_id: str) -> list[str]:
    config = STORE_CONFIGS[store_id]
    clips_dir = config["clips_dir"]
    start_time = config["start_time"]
    camera_map = config["camera_map"]
    clip_order = config["clip_order"]
    output_format = config["output_format"]
    camera_start_times = config.get("camera_start_times", {})

    print(f"\n============================================================")
    print(f"  STARTING PIPELINE FOR STORE: {store_id} ({output_format} format)")
    print(f"============================================================")

    # Initialize a single shared tracker for cross-camera Re-ID in this store
    shared_tracker = MultiCameraTracker(store_id=store_id)
    total_store_events = 0
    event_files = []

    # Verify clips directory exists
    if not os.path.exists(clips_dir):
        print(f"⚠ Clips directory not found: {clips_dir}")
        return []

    # Process mapped clips in explicit sequence
    for fname in clip_order:
        if fname not in camera_map:
            continue

        camera_id = camera_map[fname]
        clip_path = os.path.join(clips_dir, fname)
        
        if not os.path.exists(clip_path):
            print(f"⚠ Clip file not found: {clip_path}")
            continue

        # Standardise output filename (replace space with underscores)
        out_name = fname.replace(" ", "_").replace(".mp4", ".jsonl")
        output_path = os.path.join(EVENTS_DIR, out_name)
        event_files.append(output_path)

        # Clear output if it already exists to avoid duplicate entries when re-running
        if os.path.exists(output_path):
            os.remove(output_path)

        clip_time = camera_start_times.get(fname, start_time)
        print(f"\nProcessing {fname} (Camera: {camera_id}, Start: {clip_time}) -> {out_name}...")

        detector = StoreDetector(
            clip_path=clip_path,
            store_id=store_id,
            camera_id=camera_id,
            layout_path=LAYOUT_PATH,
            output_path=output_path,
            clip_start_time=clip_time,
            tracker=shared_tracker,
            output_format=output_format,
        )
        n = detector.process()
        total_store_events += n
        print(f"  → Generated {n} events")

    print(f"\nStore {store_id} processing complete. Total events: {total_store_events}")
    return event_files

def main():
    parser = argparse.ArgumentParser(description="Run Store Intelligence Detection Pipeline")
    parser.add_argument(
        "--store_id",
        choices=["ST1008", "ST1076"],
        default=None,
        help="Specific store ID to run (default: runs both)",
    )
    args = parser.parse_args()

    stores_to_run = [args.store_id] if args.store_id else list(STORE_CONFIGS.keys())
    all_event_files = []

    for store_id in stores_to_run:
        files = run_store_pipeline(store_id)
        all_event_files.extend(files)

    # ── Ingest into API ───────────────────────────────────────────────────────────
    print("\n============================================================")
    print("  INGESTING GENERATED EVENTS INTO API...")
    print("============================================================")
    accepted = 0

    for ef in all_event_files:
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
                print(f"    ⚠ Batch {i//500+1} failed to ingest: {exc}")

    print(f"\nIngestion finished. Total accepted: {accepted}")

if __name__ == "__main__":
    main()

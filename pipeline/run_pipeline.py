#!/usr/bin/env python3
"""
run_pipeline.py — Store-agnostic runner for the Store Intelligence detection pipeline.
Scans clips, matches them to store layout cameras, aligns timestamps, runs detect.py,
merges outputs, and ingests events into the API.
"""

import argparse
import os
import json
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta, timezone

def wait_for_api(api_url, max_wait=60):
    start_time = time.time()
    print(f"Waiting for API at {api_url}/health ...")
    while time.time() - start_time < max_wait:
        try:
            with urllib.request.urlopen(f"{api_url}/health", timeout=2) as response:
                if response.status == 200:
                    print("API is up and healthy.")
                    return True
        except Exception:
            pass
        time.sleep(2)
    print(f"API not available after {max_wait} seconds.")
    return False

def get_store_details(layout):
    stores = {}
    if "stores" in layout:
        for sid, sdata in layout["stores"].items():
            stores[sid] = sdata
    else:
        sid = layout.get("store_id", "ST1008")
        stores[sid] = layout
    return stores

def find_store_for_clip(clip_path, store_details):
    filename = os.path.basename(clip_path).upper()
    path_upper = os.path.abspath(clip_path).upper()
    
    # 1. Direct match on store_id in filename or path
    for sid in store_details:
        if sid.upper() in filename or sid.upper() in path_upper:
            return sid
            
    # 2. Match on store_name or city in filename or path
    for sid, sdata in store_details.items():
        name = sdata.get("store_name", "").upper()
        city = sdata.get("city", "").upper()
        if (name and name in filename) or (city and city in filename):
            return sid
        if (name and name in path_upper) or (city and city in path_upper):
            return sid
            
    # 3. Match Store number suffix: "STORE 2", "STORE_2", "STORE2" in path
    import re
    m = re.search(r"STORE[_\s-]?(\d+)", path_upper)
    if m:
        idx = int(m.group(1)) - 1
        sids = list(store_details.keys())
        if 0 <= idx < len(sids):
            return sids[idx]
            
    # 4. Fallback if only one store exists
    if len(store_details) == 1:
        return list(store_details.keys())[0]
        
    return None

def find_camera_for_clip(clip_path, camera_configs):
    filename = os.path.basename(clip_path).lower()
    
    # 1. Direct match on camera_id key in filename
    for cam_id in camera_configs:
        if cam_id.lower() in filename:
            return cam_id
            
    # 2. Check camera description or type keywords
    is_entry = any(k in filename for k in ["entry", "entrance", "exit", "door", "gate"])
    is_billing = any(k in filename for k in ["billing", "checkout", "counter", "cashier", "pos", "queue"])
    is_floor = any(k in filename for k in ["floor", "zone", "aisle", "shelf", "display", "main"])
    
    candidates = []
    for cam_id, cconfig in camera_configs.items():
        cam_type = cconfig.get("type", "").lower()
        desc = cconfig.get("description", "").lower()
        
        type_match = False
        if is_entry and cam_type == "entry":
            type_match = True
        elif is_billing and cam_type == "billing":
            type_match = True
        elif is_floor and cam_type == "floor":
            type_match = True
            
        desc_match = False
        if is_entry and any(k in desc for k in ["entry", "entrance", "exit"]):
            desc_match = True
        elif is_billing and any(k in desc for k in ["billing", "checkout", "counter"]):
            desc_match = True
        elif is_floor and any(k in desc for k in ["floor", "zone", "aisle", "shelf"]):
            desc_match = True
            
        if type_match or desc_match:
            candidates.append((cam_id, cconfig))
            
    if len(candidates) == 1:
        return candidates[0][0]
        
    if len(candidates) > 1:
        # Multiple candidates (e.g. entry 1, entry 2). Match suffix number
        import re
        num_match = re.search(r"\d+", filename)
        if num_match:
            num_str = num_match.group(0)
            for cam_id, cconfig in candidates:
                if num_str in cam_id or num_str in cconfig.get("description", ""):
                    return cam_id
            idx = int(num_str) - 1
            candidates.sort(key=lambda x: x[0])
            if 0 <= idx < len(candidates):
                return candidates[idx][0]
        return candidates[0][0]
        
    # Check if type name itself is in the filename
    for cam_id, cconfig in camera_configs.items():
        cam_type = cconfig.get("type", "").lower()
        if cam_type in filename:
            return cam_id
            
    if camera_configs:
        return list(camera_configs.keys())[0]
        
    return None

def determine_start_time(store_id, camera_id, filename, db_path=None):
    filename_lower = filename.lower()
    
    # 1. Try to parse timestamp from filename
    import re
    iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}[:_]\d{2}[:_]\d{2}Z?"
    m = re.search(iso_pattern, filename, re.IGNORECASE)
    if m:
        ts = m.group(0).replace("_", ":")
        if not ts.endswith("Z"):
            ts += "Z"
        return ts
        
    alt_pattern = r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})"
    m = re.search(alt_pattern, filename)
    if m:
        year, month, day, hour, minute, second = m.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
        
    # 2. Known store defaults
    if store_id == "ST1008":
        return "2026-04-10T20:10:00Z"
    if store_id == "ST1076":
        if "billing" in filename_lower or "checkout" in filename_lower or "billing" in camera_id.lower():
            return "2026-03-08T18:27:00Z"
        else:
            return "2026-03-08T18:10:00Z"
            
    # 3. Query DB for POS transactions to align timelines
    if db_path and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp FROM pos_transactions WHERE store_id = ? ORDER BY timestamp DESC LIMIT 1",
                (store_id,)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                txn_ts = row[0]
                try:
                    dt = datetime.fromisoformat(txn_ts.replace("Z", "+00:00"))
                    if "billing" in filename_lower or "checkout" in filename_lower or "billing" in camera_id.lower():
                        start_dt = dt - timedelta(minutes=1)
                    else:
                        start_dt = dt - timedelta(minutes=15)
                    return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
        except Exception as e:
            print(f"Error querying DB for start time: {e}")
            
    return "2026-01-01T12:00:00Z"

def ingest_events(api_url, events):
    batch_size = 500
    accepted_count = 0
    
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        payload = json.dumps({"events": batch}).encode("utf-8")
        
        req = urllib.request.Request(
            f"{api_url}/events/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                accepted_count += resp_data.get("accepted", 0)
        except urllib.error.HTTPError as e:
            print(f"  Batch {i//batch_size+1} failed with HTTP {e.code}: {e.read().decode('utf-8')}")
        except Exception as e:
            print(f"  Batch {i//batch_size+1} failed to ingest: {e}")
            
    return accepted_count

def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Batch Pipeline Runner")
    parser.add_argument("--clips_dir", required=True, help="Directory containing CCTV clips")
    parser.add_argument("--layout", required=True, help="Path to store layout configuration")
    parser.add_argument("--output_dir", required=True, help="Directory for output event files")
    parser.add_argument("--api_url", default="http://localhost:8000", help="FastAPI API URL")
    parser.add_argument("--yolo_model", default=None, help="Optional YOLO model path override")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.layout, "r", encoding="utf-8") as f:
        layout = json.load(f)

    store_details = get_store_details(layout)
    print(f"Loaded layout containing {len(store_details)} stores: {list(store_details.keys())}")

    # Find SQLite DB file to query for start-time alignment
    db_paths = [
        "/data/store_intelligence.db",
        "data/store_intelligence.db",
        "../data/store_intelligence.db",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store_intelligence.db")
    ]
    db_path = None
    for p in db_paths:
        if os.path.exists(p):
            db_path = p
            break

    # Scan clips
    clips = sorted(list(Path(args.clips_dir).glob("**/*.mp4")))
    if not clips:
        print(f"Error: No .mp4 files found in {args.clips_dir}")
        sys.exit(1)

    print(f"Found {len(clips)} clip files to process.")

    merged_output_path = os.path.join(args.output_dir, "events_all.jsonl")
    all_events = []

    for clip in clips:
        clip_path = str(clip)
        filename = clip.name
        
        # 1. Map to Store ID
        store_id = find_store_for_clip(clip_path, store_details)
        if not store_id:
            print(f"Skipping {filename}: could not identify store_id in filename or path.")
            continue
            
        # 2. Map to Camera ID
        cams = store_details[store_id].get("cameras", {})
        camera_id = find_camera_for_clip(clip_path, cams)
        if not camera_id:
            print(f"Skipping {filename}: could not identify camera_id in filename.")
            continue

        # 3. Determine clip start time
        start_time = determine_start_time(store_id, camera_id, filename, db_path)
        
        out_name = f"events_{store_id}_{camera_id}.jsonl"
        output_file = os.path.join(args.output_dir, out_name)
        
        print(f"\nProcessing clip: {filename}")
        print(f"  Store ID   : {store_id}")
        print(f"  Camera ID  : {camera_id}")
        print(f"  Start Time : {start_time}")
        print(f"  Output File: {output_file}")

        # Choose output format: Store 2 (ST1076) uses the sample schema format, others use internal
        # We can dynamically decide this: if the store config has Mumbai, or format is set
        output_format = "sample" if "1076" in store_id or "MUM" in store_id or "MUM" in store_details[store_id].get("store_name", "") else "internal"

        # Construct detection command
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "detect.py"),
            "--clip", clip_path,
            "--store_id", store_id,
            "--camera_id", camera_id,
            "--layout", args.layout,
            "--output", output_file,
            "--start_time", start_time,
            "--output_format", output_format
        ]
        if args.yolo_model:
            cmd.extend(["--yolo", args.yolo_model])

        # Run detection subprocess
        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: detection failed for {filename}: {e}")
            continue

        # Read generated events
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_events.append(json.loads(line))

    # Write merged events file
    with open(merged_output_path, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    print(f"\nTotal events generated: {len(all_events)}")
    print(f"Merged output saved to: {merged_output_path}")

    # Ingest into API
    if all_events:
        if wait_for_api(args.api_url):
            print("\nIngesting events into API...")
            accepted = ingest_events(args.api_url, all_events)
            print(f"Successfully ingested {accepted} of {len(all_events)} events.")
        else:
            print("\nSkipping ingestion: API is not running or healthy.")

if __name__ == "__main__":
    main()

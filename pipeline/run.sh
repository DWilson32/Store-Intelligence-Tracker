#!/usr/bin/env bash
# run.sh — Process all CCTV clips for all stores and emit events to JSONL
#
# Usage:
#   ./pipeline/run.sh --clips_dir /data/clips --layout /data/store_layout.json \
#                     --output_dir /data/events
#
# Output: one events_<store_id>_<camera_id>.jsonl per clip, then a merged events_all.jsonl
# The merged file is the input to POST /events/ingest

set -euo pipefail

CLIPS_DIR="${CLIPS_DIR:-/data/clips}"
LAYOUT="${LAYOUT:-/data/store_layout.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/events}"
API_URL="${API_URL:-http://localhost:8000}"
BATCH_SIZE="${BATCH_SIZE:-500}"
YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case $1 in
    --clips_dir)  CLIPS_DIR="$2";  shift 2 ;;
    --layout)     LAYOUT="$2";     shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --api_url)    API_URL="$2";    shift 2 ;;
    --yolo_model) YOLO_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
export YOLO_MODEL

echo "=== Store Intelligence Detection Pipeline ==="
echo "Clips dir : $CLIPS_DIR"
echo "Layout    : $LAYOUT"
echo "Output dir: $OUTPUT_DIR"
echo "API       : $API_URL"
echo ""

# Camera ID mapping based on filename conventions
# Expected filenames: <STORE_ID>_<camera_type>.mp4
#   e.g. STORE_BLR_002_entry.mp4, STORE_BLR_002_floor.mp4, STORE_BLR_002_billing.mp4
camera_type_to_id() {
  case "$1" in
    entry)   echo "CAM_ENTRY_01" ;;
    floor)   echo "CAM_FLOOR_01" ;;
    floor2)  echo "CAM_FLOOR_02" ;;
    floor3)  echo "CAM_FLOOR_03" ;;
    billing) echo "CAM_BILLING_01" ;;
    *)       echo "CAM_UNKNOWN_01" ;;
  esac
}

MERGED_OUTPUT="$OUTPUT_DIR/events_all.jsonl"
> "$MERGED_OUTPUT"  # truncate / create

declare -a CLIP_FILES
mapfile -t CLIP_FILES < <(find "$CLIPS_DIR" -name "*.mp4" | sort)

if [[ ${#CLIP_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No .mp4 files found in $CLIPS_DIR"
  exit 1
fi

echo "Found ${#CLIP_FILES[@]} clips to process"
echo ""

for clip in "${CLIP_FILES[@]}"; do
  filename=$(basename "$clip" .mp4)
  # Extract store_id and camera type from filename
  # Convention: STORE_BLR_002_entry → store_id=STORE_BLR_002, cam_type=entry
  store_id=$(echo "$filename" | sed 's/_\(entry\|floor\|floor2\|floor3\|billing\|main\)$//')
  store_id="${store_id%_ground}"
  cam_type=$(echo "$filename" | grep -oE '(entry|floor2|floor3|floor|billing|main)$' || echo "unknown")
  camera_id=$(camera_type_to_id "$cam_type")

  output_file="$OUTPUT_DIR/events_${filename}.jsonl"

  if [[ "$camera_id" == "CAM_UNKNOWN_01" ]]; then
    echo "Skipping non-customer clip: $filename"
    echo ""
    continue
  fi

  echo "Processing: $filename"
  echo "  Store   : $store_id"
  echo "  Camera  : $camera_id"
  echo "  Output  : $output_file"

  python pipeline/detect.py \
    --clip "$clip" \
    --store_id "$store_id" \
    --camera_id "$camera_id" \
    --layout "$LAYOUT" \
    --output "$output_file" \
    || { echo "  WARNING: Detection failed for $clip — continuing"; continue; }

  EVENT_COUNT=$(wc -l < "$output_file" 2>/dev/null || echo 0)
  echo "  Events  : $EVENT_COUNT"

  cat "$output_file" >> "$MERGED_OUTPUT"
  echo ""
done

TOTAL_EVENTS=$(wc -l < "$MERGED_OUTPUT")
echo "=== Detection complete. Total events: $TOTAL_EVENTS ==="
echo "Merged output: $MERGED_OUTPUT"
echo ""

# ---------------------------------------------------------------------------
# Ingest events into the API in batches of BATCH_SIZE
# ---------------------------------------------------------------------------
echo "=== Ingesting events into API ==="
echo "Waiting for API at $API_URL/health ..."

MAX_WAIT=60
WAITED=0
until curl -sf "$API_URL/health" > /dev/null 2>&1; do
  sleep 2
  WAITED=$((WAITED + 2))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: API not available after ${MAX_WAIT}s"
    exit 1
  fi
done
echo "API is up."

INGESTED=0
FAILED=0
BATCH=()
BATCH_COUNT=0

while IFS= read -r line; do
  BATCH+=("$line")
  if [[ ${#BATCH[@]} -ge $BATCH_SIZE ]]; then
    BATCH_JSON=$(printf '%s\n' "${BATCH[@]}" | jq -sc '.')
    HTTP_STATUS=$(curl -s -o /tmp/ingest_resp.json -w "%{http_code}" \
      -X POST "$API_URL/events/ingest" \
      -H "Content-Type: application/json" \
      -d "{\"events\": $BATCH_JSON}")

    if [[ "$HTTP_STATUS" == "200" || "$HTTP_STATUS" == "207" ]]; then
      ACCEPTED=$(jq '.accepted // 0' /tmp/ingest_resp.json 2>/dev/null || echo 0)
      INGESTED=$((INGESTED + ACCEPTED))
      BATCH_COUNT=$((BATCH_COUNT + 1))
      echo "  Batch $BATCH_COUNT: HTTP $HTTP_STATUS — accepted $ACCEPTED"
    else
      echo "  Batch $BATCH_COUNT: HTTP $HTTP_STATUS — FAILED"
      FAILED=$((FAILED + ${#BATCH[@]}))
    fi
    BATCH=()
  fi
done < "$MERGED_OUTPUT"

# Flush remaining
if [[ ${#BATCH[@]} -gt 0 ]]; then
  BATCH_JSON=$(printf '%s\n' "${BATCH[@]}" | jq -sc '.')
  HTTP_STATUS=$(curl -s -o /tmp/ingest_resp.json -w "%{http_code}" \
    -X POST "$API_URL/events/ingest" \
    -H "Content-Type: application/json" \
    -d "{\"events\": $BATCH_JSON}")
  ACCEPTED=$(jq '.accepted // 0' /tmp/ingest_resp.json 2>/dev/null || echo 0)
  INGESTED=$((INGESTED + ACCEPTED))
  echo "  Final batch: HTTP $HTTP_STATUS — accepted $ACCEPTED"
fi

echo ""
echo "=== Ingestion complete ==="
echo "  Ingested : $INGESTED events"
echo "  Failed   : $FAILED events"
echo ""
echo "API metrics: $API_URL/stores/STORE_BLR_002/metrics"

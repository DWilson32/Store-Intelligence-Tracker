# Store Intelligence — Brigade Bangalore (ST1008)

Real-time store analytics pipeline: CCTV footage → structured events → live API.

**Store**: Brigade Road, Bangalore | **Store ID**: ST1008 | **Date**: 10-Apr-2026

## Quick Start (5 commands)

```bash
# 1. Clone and enter
git clone <repo-url> store-intelligence && cd store-intelligence

# 2. Copy your CCTV clips into data/clips/
#    Expected naming: ST1008_entry.mp4, ST1008_floor.mp4,
#    ST1008_floor2.mp4, ST1008_floor3.mp4, ST1008_billing.mp4
mkdir -p data/clips
# cp /path/to/your/clips/*.mp4 data/clips/

# 3. Start the API
docker compose up --build -d

# 4. Load POS transactions (two options):
#    Option A — using the raw Purplle CSV:
python scripts/load_pos.py --csv data/Brigade_Bangalore_10_April_26.csv --raw
#    Option B — using the pre-converted pipeline CSV:
python scripts/load_pos.py --csv data/pos_transactions.csv

# 5. Run detection pipeline on clips → emits events → ingests into API
docker compose exec api bash pipeline/run.sh \
  --clips_dir /data/clips \
  --layout /data/store_layout.json \
  --output_dir /data/events \
  --api_url http://localhost:8000
```

API docs: http://localhost:8000/docs

---

## Data Files (already included)

| File | Description |
|---|---|
| `data/store_layout.json` | Approximate zone definitions for ST1008 derived from the Brigade Road layout |
| `data/Brigade_Road_store_layout.xlsx` | Original visual floor-layout workbook supplied with the challenge |
| `data/Brigade_Bangalore_10_April_26.csv` | Raw POS line-item export: 101 rows, 24 orders |
| `data/pos_transactions.csv` | Converted POS transaction table: 24 orders, 10-Apr-2026, INR 34,831.74 NMV |

### Zone Map — Brigade Bangalore

| Camera | Zones |
|---|---|
| CAM_ENTRY_01 | Entry/Exit threshold |
| CAM_FLOOR_01 / 02 / 03 | SKIN_CARE, MAKEUP, HAIR_CARE, BATH_BODY |
| CAM_BILLING_01 | BILLING, FRAGRANCE, PERSONAL_CARE |

---

## Architecture

```
CCTV Clips (ST1008)
    │
    ▼
pipeline/detect.py     YOLOv8n + ByteTrack + OSNet Re-ID
    │  emits structured JSON events per person per state change
    ▼
pipeline/run.sh        batches events → POST /events/ingest
    │
    ▼
SQLite DB              events + pos_transactions tables
    │
    ▼
app/metrics.py         GET /stores/ST1008/metrics
app/funnel.py          GET /stores/ST1008/funnel
app/heatmap.py         GET /stores/ST1008/heatmap
app/anomalies.py       GET /stores/ST1008/anomalies
app/health.py          GET /health
    │
    ▼
dashboard/live_dashboard.py   rich terminal dashboard
```

---

## Running the Detection Pipeline

```bash
# Single clip (debugging):
python pipeline/detect.py \
  --clip data/clips/ST1008_entry.mp4 \
  --store_id ST1008 \
  --camera_id CAM_ENTRY_01 \
  --layout data/store_layout.json \
  --output data/events/ST1008_entry.jsonl \
  --start_time 2026-04-10T11:00:00Z

# All clips (auto-ingests into API):
bash pipeline/run.sh \
  --clips_dir data/clips \
  --layout data/store_layout.json \
  --output_dir data/events \
  --api_url http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/events/ingest` | Ingest up to 500 events. Idempotent by `event_id`. |
| GET | `/stores/ST1008/metrics` | Unique visitors, conversion rate, queue depth, dwell |
| GET | `/stores/ST1008/funnel` | Entry → Zone → Billing → Purchase with drop-off % |
| GET | `/stores/ST1008/heatmap` | Zone frequency + dwell normalised 0–100 |
| GET | `/stores/ST1008/anomalies` | Active anomalies with severity and suggested actions |
| GET | `/health` | DB status + per-store feed freshness |

---

## Loading POS Data

```bash
# From raw Purplle sales export:
python scripts/load_pos.py \
  --csv data/Brigade_Bangalore_10_April_26.csv \
  --raw

# From pre-converted pipeline format:
python scripts/load_pos.py --csv data/pos_transactions.csv
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:////data/store_intelligence.db` | DB connection |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `DETECTION_CONFIDENCE_MIN` | `0.05` | Minimum detector confidence retained for person events |
| `API_URL` | `http://localhost:8000` | Used by `run.sh` and dashboard |
| `STORE_ID` | `ST1008` | Store shown in dashboard |

Analytics endpoints default to the latest available store data when `hours` is omitted. This keeps the supplied 10-Apr-2026 challenge dataset visible during review even if the API is run on a later date. Pass `?hours=24` or another value for wall-clock live operation.

---

## Live Dashboard

You can access the analytics visualizations using two dashboard interfaces:

### Option A — Web Dashboard (Recommended)
The web dashboard is fully containerized and starts automatically on port `3000` alongside the containers:
* **Web UI URL**: [http://localhost:3000](http://localhost:3000)

### Option B — Terminal Dashboard
You can run the interactive rich terminal-based dashboard locally:
```bash
STORE_ID=ST1008 python dashboard/live_dashboard.py
```

* **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

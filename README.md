# Store Intelligence Tracker — Multi-Store Analytics Pipeline

A generic, store-agnostic computer vision and analytics pipeline: CCTV footage → structured behavioral events → FastAPI server → live web dashboard.

Supported Stores:
* **Store 1**: Brigade Road, Bangalore (`ST1008`)
* **Store 2**: Mumbai (`ST1076`)
* *Extensible*: Works dynamically with any store layout defined in `store_layout.json`.

## Quick Start (5 commands)

```bash
# 1. Clone and enter
git clone <repo-url> store-intelligence && cd store-intelligence

# 2. Copy your CCTV clips into data/clips/
#    Expected naming:
#      - Store 1: ST1008_ground_entry.mp4, ST1008_ground_floor.mp4, ST1008_ground_billing.mp4, etc.
#      - Store 2: Store 2/entry 1.mp4, Store 2/entry 2.mp4, Store 2/zone.mp4, Store 2/billing_area.mp4
mkdir -p data/clips

# 3. Start the API & Dashboard containers
docker compose up --build -d

# 4. Load POS transactions into database (pre-loads both stores)
docker compose exec api python scripts/load_pos.py --csv data/pos_transactions.csv

# 5. Run the store-agnostic detection pipeline (processes all clips & auto-ingests)
docker compose exec api bash pipeline/run.sh \
  --clips_dir /data/clips \
  --layout /data/store_layout.json \
  --output_dir /data/events \
  --api_url http://localhost:8000
```

API docs (Swagger UI): http://localhost:8000/docs


---

## Data Files (already included)

| File | Description |
|---|---|
| `data/store_layout.json` | Approximate zone definitions for ST1008 derived from the Brigade Road layout |
| `data/Brigade_Road_store_layout.xlsx` | Original visual floor-layout workbook supplied with the challenge |
| `data/Brigade_Bangalore_10_April_26.csv` | Raw POS line-item export: 101 rows, 24 orders |
| `data/pos_transactions.csv` | Converted POS transaction table: 24 orders, 10-Apr-2026, INR 34,831.74 NMV |

### Zone Map — Layout Details

#### **Store 1: Brigade Bangalore (ST1008)**
| Camera | Zones |
|---|---|
| CAM_ENTRY_01 | Entry/Exit threshold |
| CAM_FLOOR_01 / 02 / 03 | SKIN_CARE, MAKEUP, HAIR_CARE, BATH_BODY |
| CAM_BILLING_01 | BILLING, FRAGRANCE, PERSONAL_CARE |

#### **Store 2: Mumbai (ST1076)**
| Camera | Zones |
|---|---|
| CAM_ENTRY_01 / 02 | Entry/Exit thresholds |
| CAM_ZONE_01 | Left Shelf (`PURPLLE_MUM_1076_Z01`), Center Display (`PURPLLE_MUM_1076_Z02`), Lipstick Aisle (`PURPLLE_MUM_1076_Z03`) |
| CAM_BILLING_01 | Billing Counter Queue (`PURPLLE_MUM_1076_Z_BILLING_01`) |


---

## Architecture

```
CCTV Clips (Any Store)
    │
    ▼
pipeline/detect.py     YOLOv8n + ByteTrack (runs per-store based on layout config)
    │  emits structured JSON events (compatible with standard & sample schemas)
    ▼
pipeline/run_pipeline.py  Store-agnostic orchestrator (auto-aligns start times via POS DB)
    │
    ▼
SQLite DB              events + pos_transactions tables
    │
    ▼
app/metrics.py         GET /stores/{store_id}/metrics
app/funnel.py          GET /stores/{store_id}/funnel
app/heatmap.py         GET /stores/{store_id}/heatmap
app/anomalies.py       GET /stores/{store_id}/anomalies
app/health.py          GET /health
    │
    ▼
dashboard/web/         Live visual dashboard (fully dynamic layout grid & store selector)
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
| GET | `/stores` | Returns layout configurations for all registered stores. |
| POST | `/events/ingest` | Ingest up to 500 events. Idempotent by `event_id`. |
| GET | `/stores/{store_id}/metrics` | Unique visitors, conversion rate, queue depth, average dwell |
| GET | `/stores/{store_id}/funnel` | Entry → Zone → Billing → Purchase with drop-off % |
| GET | `/stores/{store_id}/heatmap` | Zone frequency + dwell normalized 0–100 |
| GET | `/stores/{store_id}/anomalies` | Active anomalies with severity and suggested actions |
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

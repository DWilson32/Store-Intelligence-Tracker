# DESIGN.md — Store Intelligence System Architecture

## System Overview

The Store Intelligence system is a full pipeline from raw CCTV footage to a live queryable analytics API.

```
CCTV clips → Detection Pipeline → events.jsonl → POST /events/ingest → SQLite DB
                                                                            ↓
Dashboard ←─────────── GET /metrics, /funnel, /heatmap, /anomalies ────────┘
```

---

## Stage 1 — Detection Pipeline (`pipeline/`)

- **detect.py**: Detects persons (YOLOv8n), tracks them (ByteTrack), identifies staff via torso HSV color analysis (supports pink/khaki/beige/blue/white/black), and evaluates threshold crossings.
- **tracker.py**: Manages visitor identity. Employs `ReIDGallery` using OSNet cosine similarity embeddings (0.75 threshold for active tracks, 0.70 for re-entry within 10 minutes) with a trajectory-based fallback.
- **emit.py**: Standardizes events to JSONL via Pydantic. Supports both internal schema and standard sample schema (with extended metadata fields like group size, face visibility, age predictions, etc.).

### Edge Case Handling

| Edge Case | Approach |
|---|---|
| **Group entry** | ByteTrack tracks individual boxes independently &rarr; distinct ENTRY events. |
| **Staff exclusion** | Torso HSV color check flags `is_staff=true` (excluded from analytics). Support pink uniforms in Store 2. |
| **Re-entry** | Cosine similarity match in 10-minute exit buffer &rarr; REENTRY event. |
| **Partial occlusion** | Raw confidence propagated. Bounding boxes degrade gracefully via IoU. |
| **Empty periods** | No events generated; API returns 0s without crashing. |
| **Cross-camera** | Shared tracking state sequences clips dynamically based on transaction start-time alignment. |

---

## Stage 2 — Event Schema

Standardized JSON matching requirements: UUID v4 `event_id`, ISO-8601 timestamps, and a metadata block storing queue metrics, zones, and session sequences. Extended columns are included to fully ingest the Sample format schema (`gender_pred`, `age_pred`, `is_face_hidden`, `group_id`, etc.).

---

## Stage 3 — Intelligence API (`app/`)

FastAPI + async SQLAlchemy + SQLite (aiosqlite) for non-blocking execution.

- **Ingestion** (`ingestion.py`): Idempotent batch insertion (up to 500 events) using SELECT-before-INSERT. Dynamically translates/maps sample schema formats to standard database columns.
- **Metrics** (`metrics.py`): Real-time queries. Excludes staff. Correlates POS transactions with billing zone visits (5-minute window).
- **Funnel** (`funnel.py`): Session-level funnel tracking using stable `visitor_id` deduplication.
- **Anomalies** (`anomalies.py`): Automated detectors (`DEAD_ZONE`, queue spikes, conversion drops) with suggested action strings.
- **Health** (`health.py`): Monitors feed lag (>10 mins) and database availability (HTTP 503 on database failure).
- **Middleware**: Thread-correlated `trace_id` and latency logs for every HTTP request.
- **Dynamic Config**: Prioritizes reading the volume-mounted `/data/store_layout.json` to enable zero-rebuild store configuration updates.

---

## Stage 4 — Live Web Dashboard (`dashboard/web/`)

Interactive web UI served at `http://localhost:3000` (KPI cards, funnel drop-off flows, zone dwell heatmaps, auto-refresh every 5s). Fully dynamic grid layout that parses the store definitions and builds grids for any arbitrary stores loaded via layout JSON. A store dropdown selector allows navigating between Store 1 (ST1008) and Store 2 (ST1076).

---

## AI-Assisted Decisions

- **Re-Entry Threshold (Claude)**: Suggested cosine similarity levels (0.75 active / 0.70 lookback). Validated and adopted.
- **POS Window (Override)**: Claude suggested 3 minutes. Overrode to 5 minutes to strictly enforce specification limits.
- **Database Choice (Claude)**: Suggested Postgres + TimescaleDB. Chose SQLite for simple, zero-dependency docker-compose execution, leaving PostgreSQL parameters ready for production.
- **Mock POS for ST1076 (Claude)**: Recommended generating mock POS transaction corresponding to the billing camera time sequence so conversion/pos graphs render correctly on the dashboard. Validated and integrated into POS loader.


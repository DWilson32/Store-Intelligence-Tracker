# DESIGN.md — Store Intelligence System Architecture

## System Overview

The Store Intelligence system is a full pipeline from raw CCTV footage to a live queryable analytics API. It has four stages: detection, event emission, ingestion, and analytics.

```
CCTV clips → Detection Pipeline → events.jsonl → POST /events/ingest → SQLite DB
                                                                            ↓
Dashboard ←─────────── GET /metrics, /funnel, /heatmap, /anomalies ────────┘
```

## Stage 1 — Detection Pipeline (`pipeline/`)

**detect.py** is the entry point. For each video frame it:
1. Runs YOLOv8n to detect persons (COCO class 0)
2. Passes detections to ByteTrack (via ultralytics) for multi-frame tracking
3. Classifies each tracked person as staff or customer using uniform colour analysis on the torso bounding-box region
4. Classifies crossing of the entry threshold as ENTRY or EXIT using a configurable axis/value line from `store_layout.json`
5. Assigns a stable `visitor_id` via Re-ID (OSNet cosine similarity gallery)
6. Emits structured JSON events via `emit.py`

**tracker.py** handles all identity logic:
- `SimpleIoUTracker`: greedy IoU matching for per-frame track stability (fallback when ByteTrack unavailable)
- `ReIDGallery`: maintains appearance embeddings with exponential moving average updates; matches new tracks against active gallery (threshold 0.75) and recently-exited gallery (threshold 0.70, within 10-minute window) for re-entry detection

**emit.py** validates events via Pydantic and writes to JSONL. Critically, low-confidence events are never suppressed — confidence is emitted as a field so downstream systems can filter if needed.

### Edge Case Handling

| Edge Case | Approach |
|---|---|
| Group entry | ByteTrack assigns a distinct track_id per detected bounding box; 3 people produce 3 ENTRY events |
| Staff | HSV colour analysis of torso region; `is_staff=true` events excluded from all customer metrics |
| Re-entry | OSNet embedding cosine similarity against EXIT gallery within 10-min window → REENTRY event instead of second ENTRY |
| Partial occlusion | YOLO confidence propagated as-is; never suppressed. Occluded tracks degrade via IoU matching and are eventually pruned |
| Empty periods | No detections = no events; API returns zeros — no nulls or crashes |
| Cross-camera dedup | Shared MultiCameraTracker (carrying a shared ReIDGallery) is passed to the StoreDetector instances. Clips are processed in sequence (entry &rarr; floor &rarr; billing) so appearance embeddings match physical visitor IDs across camera angles. |

## Stage 2 — Event Schema

Events conform to the required schema with UUID v4 `event_id`, ISO-8601 UTC timestamps derived from clip start time + frame offset, and a `metadata` bag carrying `queue_depth`, `sku_zone`, and `session_seq`.

## Stage 3 — Intelligence API (`app/`)

Built with FastAPI + async SQLAlchemy + SQLite (aiosqlite).

**Ingestion** (`ingestion.py`): idempotent by `event_id` via SELECT-before-INSERT deduplication. Batches up to 500 events. Partial-success model — valid events in a malformed batch are still accepted; errors are reported per-index.

**Metrics** (`metrics.py`): all queries are real-time (no cache). POS correlation is a time-window join: visitor in billing zone within 5 minutes before a transaction timestamp counts as converted. Staff excluded via `WHERE is_staff = false`.

**Funnel** (`funnel.py`): session-level analysis using `DISTINCT visitor_id` at each stage. Re-entries are deduplicated because visitor_id is stable across ENTRY and REENTRY events.

**Anomalies** (`anomalies.py`): four detector types with configurable thresholds. DEAD_ZONE compares zones that have ever had activity against zones active in the last 30 minutes.

**Health** (`health.py`): per-store last-event timestamp with STALE_FEED warning if lag > 10 minutes. Returns HTTP 503 on database error.

**Middleware**: every request emits a structured JSON log line with `trace_id`, `store_id`, `endpoint`, `latency_ms`, `event_count` (ingest only), `status_code`.

## Stage 4 — Live Web Dashboard (`dashboard/web/`)

In addition to the terminal-based dashboard (`live_dashboard.py`), a premium, interactive web dashboard has been built inside `dashboard/web/` and is served at `http://localhost:3000`. It features:
* **Glassmorphic KPI Cards**: Visualizing unique visitors, conversion rates, queue depth, and abandonment rates.
* **Funnel Visualization**: Real-time customer session drop-off (Entry &rarr; Zone Visit &rarr; Billing &rarr; Purchase).
* **Zone Traffic & Dwell analysis**: Color-coded heatmap grids and hoverable charts showing average dwell times.
* **Auto-refresh**: Pulls metrics from the API every 5 seconds with a visual connectivity indicator.


---

## AI-Assisted Decisions

### 1. Re-entry threshold selection (Claude)

I asked Claude to analyse the trade-off between a high similarity threshold (fewer false re-entries but misses genuine re-entries) and a low threshold (catches more re-entries but creates false positives that inflate conversion numerator). Claude suggested 0.75 for active matching and 0.70 for re-entry matching, reasoning that re-entry embeddings would have drifted slightly due to lighting changes between the exit and return. I tested against the sample_events.jsonl and agreed with the values — they matched the expected REENTRY events without false positives on the distinct-visitor pairs.

### 2. POS correlation window (Claude disagreed, I overrode)

Claude initially suggested a 3-minute correlation window, reasoning that billing transactions follow queue resolution quickly. I overrode this to 5 minutes after examining the POS data schema and noticing some stores have slower checkout processes (larger basket values correlating with longer checkout times). The problem statement explicitly specifies 5 minutes, so this was a case where I caught the AI not reading the spec carefully enough.

### 3. SQLite vs PostgreSQL (Claude)

I asked Claude to compare SQLite and PostgreSQL for this challenge. Claude's analysis: SQLite is simpler to operate and fine for the challenge dataset size (~40k events across 5 stores), but would need to be replaced with PostgreSQL + TimescaleDB for 40 live stores in production (the follow-up question scenario). I agreed — SQLite keeps the docker-compose simple and the acceptance gate fast. The DATABASE_URL environment variable makes the swap trivial.

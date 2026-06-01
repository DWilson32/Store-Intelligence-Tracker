# CHOICES.md — Three Design Decisions

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### Options Considered

| Model | Pros | Cons |
|---|---|---|
| YOLOv8n | Fast, well-documented, runs on CPU, COCO-pretrained for persons | Smallest variant — lower mAP than larger models |
| YOLOv8m | Better accuracy at 1080p | ~3× slower inference; GPU strongly recommended |
| RT-DETR | Transformer-based, better occlusion handling | Heavier; less community tooling for tracking integration |
| MediaPipe | Very fast | Weaker on occluded/partial bodies; no ByteTrack integration |
| GPT-4V / Claude Vision | Could classify zones and staff in one pass | 20+ API calls per second is impractical; cost prohibitive; latency 2–5s per frame |

### What AI Suggested

Claude suggested YOLOv8m as the primary recommendation, citing better handling of partial occlusion (a known challenge in the footage). It also suggested considering RT-DETR for the billing area camera where queue density is highest.

### What I Chose and Why

I chose YOLOv8n for the following reasons:

1. **CPU compatibility**: The docker-compose setup must work on any reviewer machine, including those without NVIDIA GPUs. YOLOv8n runs at ~15fps on a modern CPU at 1080p with FRAME_SKIP=2; YOLOv8m would be 3× slower and potentially fail the live demo.

2. **FRAME_SKIP compensates for speed/accuracy trade-off**: By processing every other frame (FRAME_SKIP=2), the effective temporal resolution is ~8fps — sufficient for tracking at walking speed. The IoU tracker maintains identity between skipped frames.

3. **ByteTrack integration is native in ultralytics**: No extra setup, no dependency conflicts. ByteTrack's second association step (using low-confidence detections) handles partial occlusions better than simple IoU matching.

**Where I disagree with AI**: Claude was right that YOLOv8m is more accurate, but it optimised for accuracy without accounting for the deployment constraint (no guaranteed GPU). In production with dedicated inference hardware, I would upgrade to YOLOv8m or RT-DETR and remove FRAME_SKIP.

---

## Decision 2: Event Schema Design

### Options Considered

Three schema approaches were evaluated:

**Option A — Flat event per detection** (one row per detection, no session concept)
- Pro: Simple to produce; easy to stream
- Con: No session identity; funnel and conversion are impossible to compute

**Option B — Session-centric schema** (one row per session with embedded event array)
- Pro: Richer per-session analytics
- Con: Requires buffering entire sessions before emission; breaks real-time streaming; re-entry mid-session is ambiguous

**Option C — Required schema** (event-per-state-change, session reconstructed at query time via visitor_id)
- Pro: Append-only streaming; session is a query-time concept; supports real-time ingestion
- Con: Slightly more complex API queries; requires visitor_id stability (solved by Re-ID)

### What AI Suggested

Claude analysed Option B vs C and suggested Option C, reasoning that retail analytics systems benefit from an event-sourcing pattern — you can always reconstruct sessions from events, but you can't decompose session records into events. It also suggested keeping `metadata` as an open object (via Pydantic `extra="allow"`) to allow future fields without schema migrations. I agreed with both points.

For the `BILLING_QUEUE_ABANDON` event specifically, Claude suggested emitting it immediately when a visitor leaves the billing zone and resolving the purchase/abandon status at query time using POS correlation. This was correct — the alternative (waiting for POS data before emitting the event) would introduce latency into the event stream and couple the detection pipeline to the POS system.

### What I Chose and Why

Option C (the required schema) was chosen. The key design insight: `visitor_id` is the session key. All session-level metrics (funnel, dwell, conversion) are computed at query time using `GROUP BY visitor_id`. This is clean, correct, and append-only.

One deliberate decision: `confidence` is always emitted, even for values near zero. The problem statement explicitly says "do not suppress low-confidence detections." This means the API can filter by confidence if needed, and the ground-truth evaluation can assess calibration. Suppressing low-confidence events would hide detection failures.

---

## Decision 3: API Architecture — FastAPI + SQLite (sync-writes, async-reads)

### Options Considered

| Stack | Pros | Cons |
|---|---|---|
| FastAPI + SQLite | Simple, zero-dependency deploy, acceptable for ~40k events | Not suitable for 40 live stores at real-time rates |
| FastAPI + PostgreSQL | Production-grade, concurrent writes, window functions | Requires Postgres container; heavier docker-compose |
| FastAPI + Redis (event store) | Sub-millisecond metrics with pre-aggregated counters | Volatile; needs persistence layer too; over-engineered for challenge |
| Django + DRF | Batteries included | Heavier; async support is bolted-on; slower startup |

### What AI Suggested

Claude suggested PostgreSQL with a `hypertable` on `(store_id, timestamp)` via TimescaleDB, reasoning that time-series queries (metrics over a rolling window) benefit enormously from TimescaleDB's chunk-based storage. It also suggested Redis for the current queue depth (volatile, sub-second updated) and PostgreSQL for historical data.

I agreed this is the correct production architecture, but overrode it for the challenge for one reason: **the acceptance gate requires `docker compose up` on a clean machine with no manual steps**. Adding TimescaleDB requires a custom PostgreSQL image and a longer startup sequence. The DATABASE_URL environment variable is already parameterised, so swapping to PostgreSQL for production is a one-line change.

### Where I agreed with AI

The async SQLAlchemy setup (with `aiosqlite` for SQLite) was Claude's suggestion. It keeps the API non-blocking under concurrent requests — the `/metrics` endpoint does 4 sequential DB queries, and async execution means other requests are not blocked during those queries. This would matter at production load even before switching to PostgreSQL.

Claude also suggested the structured logging middleware pattern (trace_id threaded through every log line), which I implemented verbatim. This is a production requirement, not a nice-to-have: without trace_id correlation, debugging a 207 on `/events/ingest` across multiple log lines is painful.

---

## Decision 4: Cross-Camera Re-ID State Sharing & Camera Feeds Remapping

### Options Considered
When processing the 5 raw CCTV footage clips sequentially through the detection pipeline:

1. **Option A: Process all clips independently with separate tracker instances.**
   * *Pro*: Code requires no modifications; simple loops.
   * *Con*: Physical people are assigned different visitor IDs on every camera feed. This breaks the conversion funnel completely (rendering `Zone Visit` and `Billing Queue` stages 0%). Furthermore, non-customer-facing videos (like `storeroom` and `outside`) skew counts.

2. **Option B: Use a shared MultiCameraTracker instance and explicitly sequence clips.**
   * *Pro*: The appearance-based Re-ID gallery (`ReIDGallery`) accumulates customer embeddings from the ground-floor entry camera (`CAM_ENTRY_01`), enabling accurate matching when the same individuals appear on the floor (`CAM_FLOOR_01`) or billing counter (`CAM_BILLING_01`) cameras. Excludes `storeroom` and `outside` feeds completely from metrics generation.
   * *Con*: Requires minor adjustments to pass the tracker reference between detectors.

### What AI Suggested
Claude identified that the conversion funnel stage counts were dropping to 0% due to independent visitor IDs per camera and called out the "cross-camera Re-ID limitation." However, it did not suggest sharing the tracker reference to bridge this gap.

### What I Chose and Why
I chose **Option B**. The target store layout (`store_layout.json`) defines zones scoped to ground floor and cash counters.
1. Remapped the camera mappings: `ST1008_floor3` was remapped to the actual billing camera `CAM_BILLING_01` (since it captures the billing counter), and the original `ST1008_billing` was identified as outside the store and excluded. `ST1008_floor2` was identified as the storeroom and excluded.
2. Initialized a single `MultiCameraTracker` instance in `scripts/run_detection.py` and passed it to the `StoreDetector` instances.
3. Processed clips in order: `ground_entry` &rarr; `ground_floor` &rarr; `ground_billing`.
This resulted in a fully-functional, session-deduplicated conversion funnel showing that out of 4 entries, 3 visited zones and 1 entered the cash counter queue, matching actual physical activity.


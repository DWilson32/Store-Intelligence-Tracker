# CHOICES.md — Three Design Decisions

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### Options Considered

| Model | Pros | Cons |
|---|---|---|
| **YOLOv8n** | Fast, runs on CPU, COCO-pretrained for persons | Lower mAP than larger YOLO variants |
| **YOLOv8m** | Better accuracy at 1080p | 3× slower; GPU required |
| **RT-DETR** | Better occlusion handling | Heavier; less track tool integration |
| **GPT-4V / VLM** | One-pass classification | Latency (2-5s) and API cost make it impractical |

### AI Recommendation vs Decision
- **AI Recommendation**: YOLOv8m/RT-DETR to optimize for high occlusion density.
- **Selected**: YOLOv8n + ByteTrack + FRAME_SKIP=2.
- **Rationale**: Meets the deployment constraint of CPU running on reviewer machines. ByteTrack's multi-stage low-confidence association handles occlusion sufficiently. In production, we would upgrade to YOLOv8m on dedicated GPU hardware.

---

## Decision 2: Event Schema Design

### Options Considered
- **Option A (Flat event per detection)**: Simple, but makes query-time session tracking and funnels impossible.
- **Option B (Session-centric records)**: Breaks real-time streaming due to caching/buffering requirements.
- **Option C (Event-per-state-change)**: [Selected] Append-only event store; sessions compiled at query time.

### AI Recommendation vs Decision
- **AI Recommendation**: Option C with extensible metadata fields.
- **Selected**: Option C.
- **Rationale**: Keeps the ingestion layer decoupled from business query logic. Propagation of raw detection confidence values is preserved so the API/evaluation can filter downstream.

---

## Decision 3: API Architecture — FastAPI + SQLite

### Options Considered

| Stack | Pros | Cons |
|---|---|---|
| **FastAPI + SQLite** | Zero-dependency, single-command run | Not concurrent-write optimized |
| **FastAPI + PostgreSQL** | Production-ready, concurrent | Requires dedicated Postgres container |
| **FastAPI + Redis** | Sub-millisecond queue tracking | Volatile; over-engineered for challenge |

### AI Recommendation vs Decision
- **AI Recommendation**: PostgreSQL + TimescaleDB for time-series, Redis for queue states.
- **Selected**: FastAPI + SQLite (async reads, sync writes).
- **Rationale**: Tailored for the "zero-dependency, single-command run" acceptance gate. Environment variables allow swapping to PostgreSQL/TimescaleDB with one line. Async SQLAlchemy/aiosqlite keeps API endpoints non-blocking.

---

## Decision 4: Cross-Camera Re-ID State Sharing

### Options Considered
- **Option A (Independent camera trackers)**: Skews conversion funnels to 0% as visitor IDs change across angles.
- **Option B (Shared MultiCameraTracker)**: [Selected] Shares `ReIDGallery` and sequences clips (entry &rarr; floor &rarr; billing) to preserve identity. Excludes non-customer feeds.

### AI Recommendation vs Decision
- **AI Recommendation**: None (did not address cross-camera tracker matching).
- **Selected**: Option B. Remapped camera channels and passed a single tracker instance to ensure visitor continuity.

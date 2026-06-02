"""
models.py — Pydantic request/response models for Store Intelligence API
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# ---------------------------------------------------------------------------
# Inbound event schema (mirrors pipeline emit.py schema exactly)
# ---------------------------------------------------------------------------

class EventMetadataIn(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None
    # Extended fields for sample event schema
    gender_pred: Optional[str] = None
    age_pred: Optional[int] = None
    age_bucket: Optional[str] = None
    is_face_hidden: Optional[bool] = None
    group_id: Optional[str] = None
    group_size: Optional[int] = None
    zone_name: Optional[str] = None
    zone_type: Optional[str] = None
    is_revenue_zone: Optional[str] = None
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None
    queue_join_ts: Optional[str] = None
    queue_served_ts: Optional[str] = None
    queue_exit_ts: Optional[str] = None
    wait_seconds: Optional[int] = None
    queue_position: Optional[int] = None

    class Config:
        extra = "allow"


VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
}


class StoreEventIn(BaseModel):
    event_id:   str
    store_id:   str
    camera_id:  str
    visitor_id: str
    event_type: str
    timestamp:  str
    zone_id:    Optional[str] = None
    dwell_ms:   int = 0
    is_staff:   bool = False
    confidence: float
    metadata:   EventMetadataIn = Field(default_factory=EventMetadataIn)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type: {v}. Must be one of {VALID_EVENT_TYPES}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Ingest request / response
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    events: list[StoreEventIn] = Field(..., max_length=500)


class IngestError(BaseModel):
    index: int
    event_id: Optional[str]
    error: str


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    duplicates: int
    errors: list[IngestError] = []


# ---------------------------------------------------------------------------
# /metrics response
# ---------------------------------------------------------------------------

class ZoneDwellStat(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class MetricsResponse(BaseModel):
    store_id: str
    as_of: str                          # ISO-8601 UTC
    unique_visitors: int
    conversion_rate: float              # 0.0–1.0
    avg_dwell_ms_per_zone: list[ZoneDwellStat]
    current_queue_depth: int
    abandonment_rate: float             # 0.0–1.0
    data_window_hours: int = 24


# ---------------------------------------------------------------------------
# /funnel response
# ---------------------------------------------------------------------------

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float                 # % lost vs previous stage


class FunnelResponse(BaseModel):
    store_id: str
    as_of: str
    stages: list[FunnelStage]
    total_sessions: int


# ---------------------------------------------------------------------------
# /heatmap response
# ---------------------------------------------------------------------------

class HeatmapZone(BaseModel):
    zone_id: str
    sku_zone: Optional[str]
    visit_count: int
    avg_dwell_ms: float
    normalised_score: float             # 0–100
    data_confidence: bool               # False if < 20 sessions


class HeatmapResponse(BaseModel):
    store_id: str
    as_of: str
    zones: list[HeatmapZone]


# ---------------------------------------------------------------------------
# /anomalies response
# ---------------------------------------------------------------------------

class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str                           # QUEUE_SPIKE, CONVERSION_DROP, DEAD_ZONE, STALE_FEED
    severity: str                       # INFO, WARN, CRITICAL
    zone_id: Optional[str] = None
    description: str
    suggested_action: str
    detected_at: str


class AnomaliesResponse(BaseModel):
    store_id: str
    as_of: str
    anomalies: list[Anomaly]


# ---------------------------------------------------------------------------
# /health response
# ---------------------------------------------------------------------------

class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_timestamp: Optional[str]
    lag_minutes: Optional[float]
    status: str                         # OK, STALE_FEED, NO_DATA


class HealthResponse(BaseModel):
    status: str                         # ok, degraded, error
    version: str = "1.0.0"
    database: str
    stores: list[StoreFeedStatus]
    checked_at: str

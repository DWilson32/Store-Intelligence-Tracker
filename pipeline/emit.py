"""
emit.py — Event schema definition and JSONL emission for Store Intelligence Pipeline

All events emitted by the detection pipeline conform to the required schema.
This module handles:
  - UUID v4 generation per event
  - Schema validation via Pydantic
  - Buffered JSONL output with flush
  - Low-confidence events are KEPT (never suppressed) — confidence is a signal, not a filter
  - Dual output: internal format (StoreEvent) and sample-aligned format (SampleEvent*)
"""

import json
import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("emit")


# ---------------------------------------------------------------------------
# Event type catalogue
# ---------------------------------------------------------------------------

class EventType:
    ENTRY               = "ENTRY"
    EXIT                = "EXIT"
    ZONE_ENTER          = "ZONE_ENTER"
    ZONE_EXIT           = "ZONE_EXIT"
    ZONE_DWELL          = "ZONE_DWELL"
    BILLING_QUEUE_JOIN  = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY             = "REENTRY"

    ALL = {
        ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
        BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY,
    }


# ---------------------------------------------------------------------------
# Pydantic schema — INTERNAL format (used for ST1008)
# ---------------------------------------------------------------------------

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None

    class Config:
        extra = "allow"  # future-proof


class StoreEvent(BaseModel):
    event_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id:   str
    camera_id:  str
    visitor_id: str
    event_type: str
    timestamp:  str          # ISO-8601 UTC
    zone_id:    Optional[str] = None
    dwell_ms:   int = 0
    is_staff:   bool = False
    confidence: float        # always emitted — never suppressed
    metadata:   EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v not in EventType.ALL:
            raise ValueError(f"Unknown event_type: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        # Clamp to [0, 1]; negative values are treated as 0
        return max(0.0, min(1.0, float(v)))

    def to_jsonl(self) -> str:
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# Pydantic schemas — SAMPLE format (aligned with evaluator sample_events.jsonl)
# Used for Store 2 (ST1076) and evaluation
# ---------------------------------------------------------------------------

class SampleEntryExitEvent(BaseModel):
    """Entry/exit event matching sample_events.jsonl format."""
    event_type: str                       # lowercase: "entry", "exit"
    id_token: str                         # visitor identifier
    store_code: str                       # store identifier
    camera_id: str
    event_timestamp: str                  # ISO-8601
    is_staff: bool = False
    gender_pred: Optional[str] = None     # "M" / "F"
    age_pred: Optional[int] = None
    age_bucket: Optional[str] = None      # "25-34"
    is_face_hidden: bool = False
    group_id: Optional[str] = None        # "G_10" or null
    group_size: Optional[int] = None      # 2 or null

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class SampleZoneEvent(BaseModel):
    """Zone entered/exited event matching sample_events.jsonl format."""
    event_type: str                       # "zone_entered", "zone_exited"
    track_id: int
    store_id: str
    camera_id: str
    zone_id: str
    zone_name: str                        # "Left Shelf"
    zone_type: str                        # "SHELF", "DISPLAY", "BILLING"
    is_revenue_zone: str = "Yes"          # "Yes" / "No"
    event_time: str                       # ISO-8601
    zone_hotspot_x: float
    zone_hotspot_y: float
    gender: Optional[str] = None
    age: Optional[int] = None
    age_bucket: Optional[str] = None

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class SampleQueueEvent(BaseModel):
    """Queue completed/abandoned event matching sample_events.jsonl format."""
    queue_event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str                       # "queue_completed", "queue_abandoned"
    track_id: int
    store_id: str
    camera_id: str
    zone_id: str
    zone_name: str = "Billing Counter Queue"
    zone_type: str = "BILLING"
    is_revenue_zone: str = "Yes"
    queue_join_ts: str                    # ISO-8601
    queue_served_ts: Optional[str] = None # null if abandoned
    queue_exit_ts: str                    # ISO-8601
    wait_seconds: int
    queue_position_at_join: int
    abandoned: bool
    zone_hotspot_x: float
    zone_hotspot_y: float
    gender: Optional[str] = None
    age: Optional[int] = None
    age_bucket: Optional[str] = None

    def to_jsonl(self) -> str:
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# Age bucket helper
# ---------------------------------------------------------------------------

AGE_BUCKETS = [
    (0, 17, "0-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 100, "65+"),
]


def age_to_bucket(age: int) -> str:
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "25-34"


# ---------------------------------------------------------------------------
# Emitter — buffers events and writes to JSONL (INTERNAL format)
# ---------------------------------------------------------------------------

class EventEmitter:
    BUFFER_SIZE = 100  # flush every N events

    def __init__(self, store_id: str, camera_id: str, output_path: str):
        self.store_id = store_id
        self.camera_id = camera_id
        self.output_path = output_path
        self._buffer: list[str] = []
        self.event_count = 0
        self._file = open(output_path, "a", buffering=1)  # line-buffered
        logger.info(f"EventEmitter: writing to {output_path}")

    def emit(
        self,
        event_type: str,
        visitor_id: str,
        timestamp: str,
        zone_id: Optional[str],
        dwell_ms: int,
        is_staff: bool,
        confidence: float,
        session_seq: int,
        metadata: Optional[dict] = None,
    ) -> StoreEvent:
        """
        Construct, validate, and buffer a StoreEvent.
        Low-confidence events ARE emitted — confidence is preserved as metadata.
        """
        meta_dict = metadata or {}
        meta_dict["session_seq"] = session_seq

        event = StoreEvent(
            store_id=self.store_id,
            camera_id=self.camera_id,
            visitor_id=visitor_id,
            event_type=event_type,
            timestamp=timestamp,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=confidence,
            metadata=EventMetadata(**meta_dict),
        )

        line = event.to_jsonl()
        self._buffer.append(line)
        self.event_count += 1

        if len(self._buffer) >= self.BUFFER_SIZE:
            self._flush_buffer()

        if event_type in (EventType.ENTRY, EventType.EXIT, EventType.REENTRY):
            logger.debug(f"[{self.store_id}] {event_type}: visitor={visitor_id} "
                         f"staff={is_staff} conf={confidence:.2f} t={timestamp}")

        return event

    def _flush_buffer(self):
        if self._buffer:
            self._file.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()

    def flush(self):
        self._flush_buffer()
        self._file.flush()

    def __del__(self):
        try:
            self.flush()
            self._file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sample Emitter — SAMPLE format (aligned with evaluator expectations)
# ---------------------------------------------------------------------------

class SampleEventEmitter:
    """
    Emits events in the evaluator's sample_events.jsonl format.
    Handles three event schemas: entry/exit, zone, and queue events.
    """
    BUFFER_SIZE = 100

    def __init__(self, store_id: str, camera_id: str, output_path: str):
        self.store_id = store_id
        self.camera_id = camera_id
        self.output_path = output_path
        self._buffer: list[str] = []
        self.event_count = 0
        self._file = open(output_path, "a", buffering=1)
        logger.info(f"SampleEventEmitter: writing to {output_path}")

    def emit_entry_exit(
        self,
        event_type: str,
        visitor_id: str,
        timestamp: str,
        is_staff: bool,
        gender_pred: Optional[str] = None,
        age_pred: Optional[int] = None,
        is_face_hidden: bool = False,
        group_id: Optional[str] = None,
        group_size: Optional[int] = None,
    ):
        """Emit an entry or exit event in sample format."""
        # Map internal types to sample lowercase types
        type_map = {"ENTRY": "entry", "EXIT": "exit", "REENTRY": "entry"}
        evt_type = type_map.get(event_type, event_type.lower())

        event = SampleEntryExitEvent(
            event_type=evt_type,
            id_token=visitor_id,
            store_code=self.store_id.lower().replace("st", "store_"),
            camera_id=self.camera_id.lower().replace("cam_", "cam"),
            event_timestamp=timestamp,
            is_staff=is_staff,
            gender_pred=gender_pred,
            age_pred=age_pred,
            age_bucket=age_to_bucket(age_pred) if age_pred else None,
            is_face_hidden=is_face_hidden,
            group_id=group_id,
            group_size=group_size,
        )
        self._write(event.to_jsonl())

    def emit_zone(
        self,
        event_type: str,
        track_id: int,
        timestamp: str,
        zone_id: str,
        zone_name: str,
        zone_type: str,
        is_revenue_zone: str,
        cx: float,
        cy: float,
        gender: Optional[str] = None,
        age: Optional[int] = None,
    ):
        """Emit a zone entered/exited event in sample format."""
        type_map = {"ZONE_ENTER": "zone_entered", "ZONE_EXIT": "zone_exited"}
        evt_type = type_map.get(event_type, event_type.lower())

        event = SampleZoneEvent(
            event_type=evt_type,
            track_id=track_id,
            store_id=self.store_id,
            camera_id=self.camera_id,
            zone_id=zone_id,
            zone_name=zone_name,
            zone_type=zone_type,
            is_revenue_zone=is_revenue_zone,
            event_time=timestamp,
            zone_hotspot_x=round(cx, 1),
            zone_hotspot_y=round(cy, 1),
            gender=gender,
            age=age,
            age_bucket=age_to_bucket(age) if age else None,
        )
        self._write(event.to_jsonl())

    def emit_queue(
        self,
        event_type: str,
        track_id: int,
        zone_id: str,
        zone_name: str,
        queue_join_ts: str,
        queue_served_ts: Optional[str],
        queue_exit_ts: str,
        wait_seconds: int,
        queue_position_at_join: int,
        abandoned: bool,
        cx: float,
        cy: float,
        gender: Optional[str] = None,
        age: Optional[int] = None,
    ):
        """Emit a queue completed/abandoned event in sample format."""
        evt_type = "queue_abandoned" if abandoned else "queue_completed"

        event = SampleQueueEvent(
            event_type=evt_type,
            track_id=track_id,
            store_id=self.store_id,
            camera_id=self.camera_id,
            zone_id=zone_id,
            zone_name=zone_name,
            queue_join_ts=queue_join_ts,
            queue_served_ts=queue_served_ts,
            queue_exit_ts=queue_exit_ts,
            wait_seconds=wait_seconds,
            queue_position_at_join=queue_position_at_join,
            abandoned=abandoned,
            zone_hotspot_x=round(cx, 1),
            zone_hotspot_y=round(cy, 1),
            gender=gender,
            age=age,
            age_bucket=age_to_bucket(age) if age else None,
        )
        self._write(event.to_jsonl())

    def _write(self, line: str):
        self._buffer.append(line)
        self.event_count += 1
        if len(self._buffer) >= self.BUFFER_SIZE:
            self._flush_buffer()

    def _flush_buffer(self):
        if self._buffer:
            self._file.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()

    def flush(self):
        self._flush_buffer()
        self._file.flush()

    def __del__(self):
        try:
            self.flush()
            self._file.close()
        except Exception:
            pass

"""
emit.py — Event schema definition and JSONL emission for Store Intelligence Pipeline

All events emitted by the detection pipeline conform to the required schema.
This module handles:
  - UUID v4 generation per event
  - Schema validation via Pydantic
  - Buffered JSONL output with flush
  - Low-confidence events are KEPT (never suppressed) — confidence is a signal, not a filter
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
# Pydantic schema
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
# Emitter — buffers events and writes to JSONL
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

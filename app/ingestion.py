"""
ingestion.py — POST /events/ingest

Features:
  - Idempotent by event_id (duplicate events are silently skipped, not errored)
  - Batch insert up to 500 events
  - Partial success: malformed events return 207 with per-event error list
  - Validates schema via Pydantic before DB write
  - Structured logging with event_count
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, get_session
from app.models import IngestError, IngestResponse, StoreEventIn

logger = logging.getLogger("ingestion")
router = APIRouter()


async def _upsert_events(session: AsyncSession, events: list[StoreEventIn]) -> dict:
    """
    Insert events, ignoring duplicates (idempotent by event_id).
    Returns {accepted, duplicates, errors}.
    """
    accepted = 0
    duplicates = 0
    errors: list[IngestError] = []

    # Fetch existing event_ids in this batch to detect duplicates cheaply
    incoming_ids = list(dict.fromkeys(e.event_id for e in events))
    existing_result = await session.execute(
        select(EventRecord.event_id).where(EventRecord.event_id.in_(incoming_ids))
    )
    existing_ids = {row[0] for row in existing_result.fetchall()}

    records_to_insert = []
    seen_in_batch: set[str] = set()
    for i, ev in enumerate(events):
        if ev.event_id in existing_ids or ev.event_id in seen_in_batch:
            duplicates += 1
            continue
        seen_in_batch.add(ev.event_id)

        records_to_insert.append({
            "event_id":   ev.event_id,
            "store_id":   ev.store_id,
            "camera_id":  ev.camera_id,
            "visitor_id": ev.visitor_id,
            "event_type": ev.event_type,
            "timestamp":  ev.timestamp,
            "zone_id":    ev.zone_id,
            "dwell_ms":   ev.dwell_ms,
            "is_staff":   ev.is_staff,
            "confidence": ev.confidence,
            "queue_depth": ev.metadata.queue_depth if ev.metadata else None,
            "sku_zone":   ev.metadata.sku_zone if ev.metadata else None,
            "session_seq": ev.metadata.session_seq if ev.metadata else None,
        })

    if records_to_insert:
        try:
            await session.execute(
                sqlite_insert(EventRecord).prefix_with("OR IGNORE"),
                records_to_insert,
            )
            await session.commit()
            accepted = len(records_to_insert)
        except Exception as e:
            await session.rollback()
            logger.error(f"Bulk insert failed: {e}")
            # Fall back to row-by-row insert to maximise partial success
            for rec in records_to_insert:
                try:
                    session.add(EventRecord(**rec))
                    await session.commit()
                    accepted += 1
                except Exception as row_err:
                    await session.rollback()
                    errors.append(IngestError(
                        index=-1,
                        event_id=rec.get("event_id"),
                        error=str(row_err),
                    ))

    return {"accepted": accepted, "duplicates": duplicates, "errors": errors}


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Accepts batches of up to 500 events.
    - Idempotent: same event_id submitted twice is a no-op (not an error).
    - Partial success: malformed events are rejected; valid ones are accepted.
    - Returns 200 if all events accepted; 207 if some rejected.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "accepted": 0,
                "rejected": 1,
                "duplicates": 0,
                "errors": [{"index": -1, "event_id": None, "error": "invalid JSON body"}],
            },
        )

    raw_events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(raw_events, list):
        return JSONResponse(
            status_code=422,
            content={
                "accepted": 0,
                "rejected": 1,
                "duplicates": 0,
                "errors": [{"index": -1, "event_id": None, "error": "events must be a list"}],
            },
        )

    if len(raw_events) > 500:
        return JSONResponse(
            status_code=422,
            content={
                "accepted": 0,
                "rejected": len(raw_events),
                "duplicates": 0,
                "errors": [{"index": -1, "event_id": None, "error": "batch size must be <= 500"}],
            },
        )

    valid_events: list[StoreEventIn] = []
    parse_errors: list[IngestError] = []

    for i, raw_event in enumerate(raw_events):
        event_id = raw_event.get("event_id") if isinstance(raw_event, dict) else None
        try:
            ev = StoreEventIn.model_validate(raw_event)
        except ValidationError as exc:
            parse_errors.append(IngestError(
                index=i,
                event_id=event_id,
                error="; ".join(err["msg"] for err in exc.errors()),
            ))
            continue

        if not ev.store_id or not ev.visitor_id:
            parse_errors.append(IngestError(
                index=i,
                event_id=ev.event_id,
                error="store_id and visitor_id are required",
            ))
        else:
            valid_events.append(ev)

    result = await _upsert_events(session, valid_events)

    total_errors = parse_errors + result["errors"]
    accepted = result["accepted"]
    duplicates = result["duplicates"]

    # Attach event_count to request state for middleware logging
    request.state.event_count = accepted

    response = IngestResponse(
        accepted=accepted,
        rejected=len(parse_errors) + len(result["errors"]),
        duplicates=duplicates,
        errors=total_errors,
    )

    status_code = 207 if total_errors else 200
    return JSONResponse(content=response.model_dump(), status_code=status_code)

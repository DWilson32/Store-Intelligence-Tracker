"""
health.py — GET /health

Service health check used by on-call engineers.
Returns:
  - Database connectivity status
  - Last event timestamp per store
  - STALE_FEED warning if any store has > 10 min lag
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, get_session, get_db_status
from app.models import HealthResponse, StoreFeedStatus

logger = logging.getLogger("health")
router = APIRouter()

STALE_FEED_THRESHOLD_MIN = 10


@router.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_session)):
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_threshold = (now - timedelta(minutes=STALE_FEED_THRESHOLD_MIN)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # DB status
    db_status = await get_db_status()

    # Per-store last event timestamp
    store_result = await session.execute(
        select(
            EventRecord.store_id,
            func.max(EventRecord.timestamp).label("last_event"),
        ).group_by(EventRecord.store_id)
    )
    rows = store_result.fetchall()

    store_statuses: list[StoreFeedStatus] = []
    has_stale = False

    for row in rows:
        store_id = row.store_id
        last_ts = row.last_event

        if last_ts is None:
            status = "NO_DATA"
            lag_min = None
        elif last_ts < stale_threshold:
            status = "STALE_FEED"
            has_stale = True
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                lag_min = round((now - last_dt).total_seconds() / 60, 1)
            except Exception:
                lag_min = None
        else:
            status = "OK"
            lag_min = None

        store_statuses.append(StoreFeedStatus(
            store_id=store_id,
            last_event_timestamp=last_ts,
            lag_minutes=lag_min,
            status=status,
        ))

    # Overall service status
    if db_status["status"] != "ok":
        overall = "error"
    elif has_stale:
        overall = "degraded"
    else:
        overall = "ok"

    response = HealthResponse(
        status=overall,
        database=db_status["status"],
        stores=store_statuses,
        checked_at=now_str,
    )

    status_code = 200 if overall != "error" else 503
    return JSONResponse(content=response.model_dump(), status_code=status_code)

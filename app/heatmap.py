"""
heatmap.py — GET /stores/{store_id}/heatmap

Zone visit frequency + avg dwell, normalised 0–100 for grid rendering.
data_confidence = False when fewer than 20 sessions in the window.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, get_session
from app.metrics import resolve_analysis_window
from app.models import HeatmapResponse, HeatmapZone

logger = logging.getLogger("heatmap")
router = APIRouter()

LOW_CONFIDENCE_SESSION_THRESHOLD = 20


@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    hours: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    anchor, window_start, window_end = await resolve_analysis_window(session, store_id, hours)

    # Unique session count for confidence flag
    session_count_result = await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp < window_end,
        )
    )
    total_sessions = session_count_result.scalar() or 0
    low_confidence = total_sessions < LOW_CONFIDENCE_SESSION_THRESHOLD

    # Zone stats
    zone_result = await session.execute(
        select(
            EventRecord.zone_id,
            EventRecord.sku_zone,
            func.count(EventRecord.id).label("visit_count"),
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
        ).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL", "ZONE_EXIT"]),
            EventRecord.zone_id != None,
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp < window_end,
        ).group_by(EventRecord.zone_id, EventRecord.sku_zone)
    )
    rows = zone_result.fetchall()

    if not rows:
        return HeatmapResponse(
            store_id=store_id,
            as_of=anchor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            zones=[],
        )

    # Normalise visit_count to 0–100
    max_visits = max(row.visit_count for row in rows) or 1

    zones = [
        HeatmapZone(
            zone_id=row.zone_id,
            sku_zone=row.sku_zone,
            visit_count=row.visit_count,
            avg_dwell_ms=round(row.avg_dwell or 0, 1),
            normalised_score=round(row.visit_count / max_visits * 100, 1),
            data_confidence=not low_confidence,
        )
        for row in rows
        if row.zone_id
    ]
    zones.sort(key=lambda z: z.normalised_score, reverse=True)

    return HeatmapResponse(
        store_id=store_id,
        as_of=anchor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        zones=zones,
    )

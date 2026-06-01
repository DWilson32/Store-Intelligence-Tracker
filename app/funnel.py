"""
funnel.py — GET /stores/{store_id}/funnel

Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase
Session is the unit of analysis — re-entries must NOT double-count a visitor.

Funnel stages:
  1. Entry          — unique visitor sessions that had an ENTRY event
  2. Zone Visit     — sessions that also had at least one ZONE_ENTER event
  3. Billing Queue  — sessions that also had BILLING_QUEUE_JOIN
  4. Purchase       — sessions with a POS-correlated conversion
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction, get_session
from app.models import FunnelResponse, FunnelStage
from app.metrics import _parse_ts, POS_CORRELATION_WINDOW_MIN, resolve_analysis_window

logger = logging.getLogger("funnel")
router = APIRouter()

DEFAULT_WINDOW_HOURS = 24


@router.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str,
    hours: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Returns the conversion funnel for the store.
    Session-level deduplication: a visitor_id that re-entered is counted ONCE.
    """
    anchor, window_start, window_end = await resolve_analysis_window(session, store_id, hours)

    # Stage 1: unique ENTRY sessions (non-staff, deduplicated by visitor_id)
    entry_result = await session.execute(
        select(func.distinct(EventRecord.visitor_id)).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp < window_end,
        )
    )
    entry_visitors = {row[0] for row in entry_result.fetchall()}

    # Stage 2: visitors with at least one zone visit
    zone_result = await session.execute(
        select(func.distinct(EventRecord.visitor_id)).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ZONE_ENTER",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp < window_end,
            EventRecord.visitor_id.in_(entry_visitors),
        )
    )
    zone_visitors = {row[0] for row in zone_result.fetchall()}

    # Stage 3: visitors who joined billing queue
    billing_result = await session.execute(
        select(func.distinct(EventRecord.visitor_id)).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp < window_end,
            EventRecord.visitor_id.in_(entry_visitors),
        )
    )
    billing_visitors = {row[0] for row in billing_result.fetchall()}

    # Stage 4: POS-correlated purchasers (same logic as metrics.py)
    pos_result = await session.execute(
        select(POSTransaction.timestamp).where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= window_start,
            POSTransaction.timestamp < window_end,
        )
    )
    pos_timestamps = [row[0] for row in pos_result.fetchall()]

    converted_ids: set[str] = set()
    for txn_ts_str in pos_timestamps:
        txn_dt = _parse_ts(txn_ts_str)
        if txn_dt is None:
            continue
        win_open = (txn_dt - timedelta(minutes=POS_CORRELATION_WINDOW_MIN)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        billing_window_result = await session.execute(
            select(func.distinct(EventRecord.visitor_id)).where(
                EventRecord.store_id == store_id,
                EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
                EventRecord.zone_id.ilike("%BILLING%"),
                EventRecord.is_staff == False,
                EventRecord.timestamp >= win_open,
                EventRecord.timestamp <= txn_ts_str,
                EventRecord.visitor_id.in_(entry_visitors),
            )
        )
        for row in billing_window_result.fetchall():
            converted_ids.add(row[0])

    # Build funnel stages
    stage_counts = [
        ("Entry",         len(entry_visitors)),
        ("Zone Visit",    len(zone_visitors)),
        ("Billing Queue", len(billing_visitors)),
        ("Purchase",      len(converted_ids)),
    ]

    stages: list[FunnelStage] = []
    for i, (name, count) in enumerate(stage_counts):
        if i == 0:
            drop_off_pct = 0.0
        else:
            prev_count = stage_counts[i - 1][1]
            drop_off_pct = (
                round((1 - count / prev_count) * 100, 1) if prev_count > 0 else 0.0
            )
        stages.append(FunnelStage(stage=name, count=count, drop_off_pct=drop_off_pct))

    return FunnelResponse(
        store_id=store_id,
        as_of=anchor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        stages=stages,
        total_sessions=len(entry_visitors),
    )

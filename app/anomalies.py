"""
anomalies.py — GET /stores/{store_id}/anomalies

Detects and returns active operational anomalies:
  - BILLING_QUEUE_SPIKE   — queue depth exceeds threshold
  - CONVERSION_DROP       — conversion rate significantly below 7-day average
  - DEAD_ZONE             — no zone visits in last 30 minutes
  - STALE_FEED            — no events in last 10 minutes

Each anomaly includes severity (INFO/WARN/CRITICAL) and a suggested_action string.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction, get_session
from app.models import Anomaly, AnomaliesResponse
from app.metrics import _parse_ts, POS_CORRELATION_WINDOW_MIN, resolve_analysis_window

logger = logging.getLogger("anomalies")
router = APIRouter()

# Thresholds
QUEUE_SPIKE_WARN     = 5
QUEUE_SPIKE_CRITICAL = 10
CONVERSION_DROP_WARN = 0.25    # 25% below 7-day avg
CONVERSION_DROP_CRIT = 0.50    # 50% below 7-day avg
DEAD_ZONE_MIN        = 30
STALE_FEED_MIN       = 10


async def _compute_conversion_rate(session, store_id: str, window_start: str, window_end: str) -> float:
    """Reusable conversion rate computation for a given time window."""
    visitors_result = await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp <= window_end,
        )
    )
    visitors = visitors_result.scalar() or 0
    if visitors == 0:
        return 0.0

    pos_result = await session.execute(
        select(POSTransaction.timestamp).where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= window_start,
            POSTransaction.timestamp <= window_end,
        )
    )
    pos_timestamps = [row[0] for row in pos_result.fetchall()]

    converted: set[str] = set()
    for txn_ts in pos_timestamps:
        txn_dt = _parse_ts(txn_ts)
        if txn_dt is None:
            continue
        win_open = (txn_dt - timedelta(minutes=POS_CORRELATION_WINDOW_MIN)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        res = await session.execute(
            select(func.distinct(EventRecord.visitor_id)).where(
                EventRecord.store_id == store_id,
                EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
                EventRecord.zone_id.ilike("%BILLING%"),
                EventRecord.is_staff == False,
                EventRecord.timestamp >= win_open,
                EventRecord.timestamp <= txn_ts,
            )
        )
        for row in res.fetchall():
            converted.add(row[0])

    return len(converted) / visitors


@router.get("/stores/{store_id}/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(
    store_id: str,
    hours: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    anchor, _, _ = await resolve_analysis_window(session, store_id, hours)
    now_str = anchor.strftime("%Y-%m-%dT%H:%M:%SZ")
    anomalies: list[Anomaly] = []

    # -------------------------------------------------------------------------
    # 1. BILLING_QUEUE_SPIKE — high queue depth in last 30 min
    # -------------------------------------------------------------------------
    thirty_min_ago = (anchor - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    join_res = await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= thirty_min_ago,
        )
    )
    abandon_res = await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_ABANDON",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= thirty_min_ago,
        )
    )
    queue_depth = max(0, (join_res.scalar() or 0) - (abandon_res.scalar() or 0))

    if queue_depth >= QUEUE_SPIKE_CRITICAL:
        anomalies.append(Anomaly(
            type="BILLING_QUEUE_SPIKE",
            severity="CRITICAL",
            zone_id="BILLING",
            description=f"Billing queue depth is {queue_depth} (threshold: {QUEUE_SPIKE_CRITICAL})",
            suggested_action="Open additional checkout counters immediately. Alert floor manager.",
            detected_at=now_str,
        ))
    elif queue_depth >= QUEUE_SPIKE_WARN:
        anomalies.append(Anomaly(
            type="BILLING_QUEUE_SPIKE",
            severity="WARN",
            zone_id="BILLING",
            description=f"Billing queue depth is {queue_depth} (threshold: {QUEUE_SPIKE_WARN})",
            suggested_action="Monitor queue. Consider calling additional staff to checkout.",
            detected_at=now_str,
        ))

    # -------------------------------------------------------------------------
    # 2. CONVERSION_DROP — today vs 7-day average
    # -------------------------------------------------------------------------
    today_start = anchor.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    today_rate = await _compute_conversion_rate(session, store_id, today_start, now_str)

    week_start = (anchor - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    week_end = (anchor - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    week_rate = await _compute_conversion_rate(session, store_id, week_start, week_end)

    if week_rate > 0.01:  # Only compare if we have a meaningful baseline
        drop = (week_rate - today_rate) / week_rate
        if drop >= CONVERSION_DROP_CRIT:
            anomalies.append(Anomaly(
                type="CONVERSION_DROP",
                severity="CRITICAL",
                description=(
                    f"Today's conversion {today_rate:.1%} is {drop:.0%} below "
                    f"7-day average {week_rate:.1%}"
                ),
                suggested_action=(
                    "Investigate pricing, staffing, and product availability. "
                    "Check if POS system is recording transactions correctly."
                ),
                detected_at=now_str,
            ))
        elif drop >= CONVERSION_DROP_WARN:
            anomalies.append(Anomaly(
                type="CONVERSION_DROP",
                severity="WARN",
                description=(
                    f"Today's conversion {today_rate:.1%} is {drop:.0%} below "
                    f"7-day average {week_rate:.1%}"
                ),
                suggested_action="Review floor staffing levels and promotional activity.",
                detected_at=now_str,
            ))

    # -------------------------------------------------------------------------
    # 3. DEAD_ZONE — no zone visits in last 30 minutes
    # -------------------------------------------------------------------------
    zone_result = await session.execute(
        select(func.distinct(EventRecord.zone_id)).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ZONE_ENTER",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= thirty_min_ago,
        )
    )
    active_zones = {row[0] for row in zone_result.fetchall() if row[0]}

    # Get all zones that have EVER had activity
    all_zones_result = await session.execute(
        select(func.distinct(EventRecord.zone_id)).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ZONE_ENTER",
            EventRecord.is_staff == False,
        )
    )
    all_zones = {row[0] for row in all_zones_result.fetchall() if row[0]}
    dead_zones = all_zones - active_zones

    for zone in dead_zones:
        anomalies.append(Anomaly(
            type="DEAD_ZONE",
            severity="INFO",
            zone_id=zone,
            description=f"Zone '{zone}' has had no customer visits in the last {DEAD_ZONE_MIN} minutes",
            suggested_action=(
                f"Check if zone '{zone}' is accessible and properly stocked. "
                "Consider repositioning nearby displays."
            ),
            detected_at=now_str,
        ))

    # -------------------------------------------------------------------------
    # 4. STALE_FEED — no events in last 10 minutes
    # -------------------------------------------------------------------------
    ten_min_ago = (anchor - timedelta(minutes=STALE_FEED_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    latest_result = await session.execute(
        select(func.max(EventRecord.timestamp)).where(
            EventRecord.store_id == store_id,
        )
    )
    latest_ts = latest_result.scalar()

    if latest_ts is None:
        anomalies.append(Anomaly(
            type="STALE_FEED",
            severity="WARN",
            description="No events have ever been received for this store",
            suggested_action="Verify detection pipeline is running and pointing to correct store_id.",
            detected_at=now_str,
        ))
    elif latest_ts < ten_min_ago:
        anomalies.append(Anomaly(
            type="STALE_FEED",
            severity="CRITICAL",
            description=f"Last event received at {latest_ts} — feed is stale (>{STALE_FEED_MIN} min lag)",
            suggested_action=(
                "Check camera feeds and detection pipeline. "
                "Verify network connectivity from detection server to API."
            ),
            detected_at=now_str,
        ))

    return AnomaliesResponse(
        store_id=store_id,
        as_of=now_str,
        anomalies=anomalies,
    )

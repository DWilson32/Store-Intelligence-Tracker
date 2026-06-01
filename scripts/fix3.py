new_content = '''
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction, get_session
from app.models import MetricsResponse, ZoneDwellStat

logger = logging.getLogger("metrics")
router = APIRouter()

POS_CORRELATION_WINDOW_MIN = 5
DEFAULT_WINDOW_HOURS = 24


def _parse_ts(ts_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


@router.get("/stores/{store_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    store_id: str,
    hours: int = DEFAULT_WINDOW_HOURS,
    session: AsyncSession = Depends(get_session),
):
    now = datetime.now(timezone.utc)
    window_start = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    thirty_min_ago = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Unique visitors deduplicated with 30s window
    entry_rows = (await session.execute(
        select(EventRecord.visitor_id, EventRecord.timestamp).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["ENTRY", "REENTRY"]),
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
        ).order_by(EventRecord.timestamp)
    )).fetchall()

    unique_visitors = 0
    last_entry_dt = None
    for _, ts_str in entry_rows:
        dt = _parse_ts(ts_str)
        if dt is None:
            continue
        if last_entry_dt is None or (dt - last_entry_dt).total_seconds() > 30:
            unique_visitors += 1
            last_entry_dt = dt

    # 2. Conversion rate
    converted_visitors = 0
    if unique_visitors > 0:
        pos_rows = (await session.execute(
            select(POSTransaction.timestamp).where(
                POSTransaction.store_id == store_id,
                POSTransaction.timestamp >= window_start,
            )
        )).fetchall()
        pos_timestamps = [row[0] for row in pos_rows]
        converted_ids = set()
        for txn_ts_str in pos_timestamps:
            txn_dt = _parse_ts(txn_ts_str)
            if txn_dt is None:
                continue
            win_open = (txn_dt - timedelta(minutes=POS_CORRELATION_WINDOW_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
            billing_rows = (await session.execute(
                select(func.distinct(EventRecord.visitor_id)).where(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
                    EventRecord.zone_id.ilike("%BILLING%"),
                    EventRecord.is_staff == False,
                    EventRecord.timestamp >= win_open,
                    EventRecord.timestamp <= txn_ts_str,
                )
            )).fetchall()
            for row in billing_rows:
                converted_ids.add(row[0])
        converted_visitors = len(converted_ids)

    conversion_rate = converted_visitors / unique_visitors if unique_visitors > 0 else 0.0

    # 3. Zone dwell
    dwell_rows = (await session.execute(
        select(
            EventRecord.zone_id,
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            func.count(EventRecord.id).label("visit_count"),
        ).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["ZONE_EXIT", "ZONE_DWELL"]),
            EventRecord.zone_id != None,
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
        ).group_by(EventRecord.zone_id)
    )).fetchall()

    zone_dwell_stats = [
        ZoneDwellStat(
            zone_id=row.zone_id,
            avg_dwell_ms=round(row.avg_dwell or 0, 1),
            visit_count=row.visit_count,
        )
        for row in dwell_rows if row.zone_id
    ]

    # 4. Queue depth
    queue_joins = (await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= thirty_min_ago,
        )
    )).scalar() or 0

    queue_abandons = (await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_ABANDON",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= thirty_min_ago,
        )
    )).scalar() or 0

    queue_depth = max(0, queue_joins - queue_abandons)

    # 5. Abandonment rate
    total_joins = (await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
        )
    )).scalar() or 0

    total_abandons = (await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id))).where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_ABANDON",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= window_start,
        )
    )).scalar() or 0

    abandonment_rate = total_abandons / total_joins if total_joins > 0 else 0.0

    return MetricsResponse(
        store_id=store_id,
        as_of=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_ms_per_zone=zone_dwell_stats,
        current_queue_depth=queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        data_window_hours=hours,
    )
'''

with open("/app/app/metrics.py", "w") as f:
    f.write(new_content)
print("metrics.py rewritten successfully")

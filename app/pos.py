"""
pos.py — GET /stores/{store_id}/pos

Returns POS transaction data with product-level detail from the Brigade CSV,
enriched with visitor correlation status from the events table.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction, get_session
from app.metrics import _parse_ts, POS_CORRELATION_WINDOW_MIN, resolve_analysis_window
from pydantic import BaseModel

logger = logging.getLogger("pos")
router = APIRouter()


class POSTransactionOut(BaseModel):
    transaction_id: str
    timestamp: str
    basket_value: float
    correlated_visitor: Optional[str] = None


class POSSummaryResponse(BaseModel):
    store_id: str
    as_of: str
    total_transactions: int
    total_revenue: float
    avg_basket_value: float
    correlated_count: int
    transactions: list[POSTransactionOut]


@router.get("/stores/{store_id}/pos", response_model=POSSummaryResponse)
async def get_pos_summary(
    store_id: str,
    hours: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Returns POS transaction summary for the store, including per-transaction
    visitor correlation status.
    """
    anchor, window_start, window_end = await resolve_analysis_window(
        session, store_id, hours
    )

    pos_result = await session.execute(
        select(POSTransaction).where(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= window_start,
            POSTransaction.timestamp < window_end,
        ).order_by(POSTransaction.timestamp)
    )
    pos_rows = pos_result.scalars().all()

    # For each POS transaction, find correlated visitor using the 5-min window
    transactions: list[POSTransactionOut] = []
    correlated_count = 0
    total_revenue = 0.0

    for txn in pos_rows:
        total_revenue += txn.basket_value

        txn_dt = _parse_ts(txn.timestamp)
        correlated_visitor = None

        if txn_dt is not None:
            win_open = (
                txn_dt - timedelta(minutes=POS_CORRELATION_WINDOW_MIN)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            visitor_result = await session.execute(
                select(EventRecord.visitor_id).where(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type.in_(
                        ["BILLING_QUEUE_JOIN", "ZONE_ENTER"]
                    ),
                    EventRecord.zone_id.ilike("%BILLING%"),
                    EventRecord.is_staff == False,
                    EventRecord.timestamp >= win_open,
                    EventRecord.timestamp <= txn.timestamp,
                ).limit(1)
            )
            row = visitor_result.first()
            if row:
                correlated_visitor = row[0]
                correlated_count += 1

        transactions.append(
            POSTransactionOut(
                transaction_id=txn.transaction_id,
                timestamp=txn.timestamp,
                basket_value=txn.basket_value,
                correlated_visitor=correlated_visitor,
            )
        )

    total = len(transactions)
    avg_basket = total_revenue / total if total > 0 else 0.0

    return POSSummaryResponse(
        store_id=store_id,
        as_of=anchor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_transactions=total,
        total_revenue=round(total_revenue, 2),
        avg_basket_value=round(avg_basket, 2),
        correlated_count=correlated_count,
        transactions=transactions,
    )

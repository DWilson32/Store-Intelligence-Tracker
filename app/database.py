"""
database.py — Async SQLAlchemy setup for Store Intelligence API

Uses SQLite (via aiosqlite) for simplicity — swap DATABASE_URL env var
for PostgreSQL in production (asyncpg driver).

Tables:
  events      — all ingested store events (append-only)
  pos_txns    — POS transaction records for purchase correlation

Design note: SQLite works fine for the challenge dataset (~40k events).
At 40 live stores with real-time feeds, switch to PostgreSQL + TimescaleDB
with a hypertable on (store_id, timestamp). See CHOICES.md for full analysis.
"""

import logging
import os

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table, Text,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger("database")

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite+aiosqlite:////data/store_intelligence.db",
)

# SQLite-specific: StaticPool + check_same_thread=False for async usage
CONNECT_ARGS: dict = {}
POOL_CLASS = None

if DATABASE_URL.startswith("sqlite"):
    CONNECT_ARGS = {"check_same_thread": False}
    POOL_CLASS = StaticPool

engine_kwargs: dict = {
    "echo": os.environ.get("DB_ECHO", "0") == "1",
    "connect_args": CONNECT_ARGS,
}
if POOL_CLASS is not None:
    engine_kwargs["poolclass"] = POOL_CLASS

engine = create_async_engine(DATABASE_URL, **engine_kwargs)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class EventRecord(Base):
    __tablename__ = "events"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    event_id    = Column(String(36), unique=True, nullable=False, index=True)  # UUID dedup
    store_id    = Column(String(64), nullable=False, index=True)
    camera_id   = Column(String(64), nullable=False)
    visitor_id  = Column(String(64), nullable=False, index=True)
    event_type  = Column(String(32), nullable=False, index=True)
    timestamp   = Column(String(32), nullable=False, index=True)  # ISO-8601 string
    zone_id     = Column(String(64), nullable=True)
    dwell_ms    = Column(Integer, default=0)
    is_staff    = Column(Boolean, default=False)
    confidence  = Column(Float, nullable=False)
    queue_depth = Column(Integer, nullable=True)
    sku_zone    = Column(String(64), nullable=True)
    session_seq = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_type", "store_id", "event_type"),
        Index("ix_events_visitor", "store_id", "visitor_id"),
    )


class POSTransaction(Base):
    __tablename__ = "pos_transactions"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    store_id       = Column(String(64), nullable=False, index=True)
    transaction_id = Column(String(64), unique=True, nullable=False)
    timestamp      = Column(String(32), nullable=False, index=True)
    basket_value   = Column(Float, nullable=False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_db():
    """Create all tables if they don't exist."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Database initialised: {DATABASE_URL}")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        raise


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_status() -> dict:
    """Used by /health endpoint."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok", "url": DATABASE_URL.split("///")[-1]}
    except Exception as e:
        return {"status": "error", "error": str(e)}

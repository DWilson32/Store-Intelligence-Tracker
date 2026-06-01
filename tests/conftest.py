"""
conftest.py — Shared async test fixtures for Store Intelligence API tests.

Uses a single in-memory SQLite engine shared across all test modules.
Each test class gets a fresh schema via the autouse setup_db fixture.
"""

import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import app AFTER env var is set
from app.main import app
from app.database import Base, get_session

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_session():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_session] = override_get_session


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def db_session():
    async with TestSessionLocal() as session:
        yield session

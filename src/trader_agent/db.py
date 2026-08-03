"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from trader_agent.config import settings

# -- Synchronous engine (migrations, scripts, simple queries) -----------------

sync_engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=False,
)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)

# -- Async engine (ingestion services, high-throughput paths) -----------------

async_engine = create_async_engine(
    settings.async_database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)


def get_sync_session() -> Session:
    """Create a synchronous DB session. Caller is responsible for closing."""
    return SyncSessionLocal()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async DB session with automatic cleanup."""
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

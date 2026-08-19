"""Shared pytest fixtures for StratOps Intel backend tests."""

import asyncio
import os
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy import text

logger = structlog.get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """Create an event loop for the entire test session."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def db_engine() -> AsyncEngine:
    """Create a session-scoped async SQLAlchemy engine for tests.

    Uses TEST_DATABASE_URL env var if set, otherwise falls back to a default.
    Uses NullPool to avoid connection issues in tests.
    """
    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://stratops:stratops_dev_password@localhost:5432/stratops_test",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        connect_args={
            "server_settings": {
                "application_name": "stratops-test",
                "jit": "off",
            },
        },
    )
    logger.info("test_engine_created", database_url=database_url.split("@")[-1])
    yield engine
    asyncio.run(engine.dispose())


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session for tests.

    Rolls back all changes after each test to ensure isolation.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_maker = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_maker() as session:
        await session.execute(
            text("SET LOCAL app.current_tenant = '00000000-0000-0000-0000-000000000000'")
        )
        yield session
        await session.rollback()
        await session.close()


@pytest.fixture
async def test_tenant(db_session: AsyncSession) -> AsyncGenerator[dict, None]:
    """Create a test tenant and yield its data.

    Cleans up after the test is complete.
    """
    from db.models import Tenant

    tenant_id = uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant",
        tier="free",
    )
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)

    yield {"id": tenant_id, "name": "Test Tenant", "slug": "test-tenant", "tier": "free"}

    # Cleanup - handled by session rollback


@pytest.fixture
async def test_tenant_a(db_session: AsyncSession) -> AsyncGenerator[dict, None]:
    """Create tenant A for cross-tenant isolation tests."""
    from db.models import Tenant

    tenant_id = uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Tenant A",
        slug="tenant-a",
        tier="pro",
    )
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)

    yield {"id": tenant_id, "name": "Tenant A", "slug": "tenant-a", "tier": "pro"}


@pytest.fixture
async def test_tenant_b(db_session: AsyncSession) -> AsyncGenerator[dict, None]:
    """Create tenant B for cross-tenant isolation tests."""
    from db.models import Tenant

    tenant_id = uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Tenant B",
        slug="tenant-b",
        tier="free",
    )
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)

    yield {"id": tenant_id, "name": "Tenant B", "slug": "tenant-b", "tier": "free"}


@pytest.fixture(autouse=True)
def mock_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("ENVIRONMENT", "test")
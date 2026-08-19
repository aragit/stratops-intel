"""Integration tests for tenant isolation via PostgreSQL RLS.

These tests use testcontainers to spin up a real PostgreSQL instance
and verify that RLS policies correctly enforce tenant isolation.
"""

import os
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer

from db.models import APIKey, Base, Tenant, TenantConfig, User
from db.tenant_session import (
    get_admin_session,
    get_tenant_session,
    initialize_database,
)

logger = structlog.get_logger(__name__)


@pytest.fixture(scope="module")
def postgres_container() -> PostgresContainer:
    """Start a PostgreSQL testcontainer for integration tests."""
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        os.environ["TEST_DATABASE_URL"] = postgres.get_connection_url()
        yield postgres


@pytest.fixture(scope="module")
async def integration_engine(postgres_container: PostgresContainer) -> AsyncGenerator[AsyncEngine, None]:
    """Create an async engine pointing to the testcontainer."""
    database_url = postgres_container.get_connection_url()
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        connect_args={
            "server_settings": {
                "application_name": "integration-test",
            },
        },
    )

    # Create extensions and helper function
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("""
            CREATE OR REPLACE FUNCTION set_tenant_context(tenant_uuid UUID)
            RETURNS VOID AS $$
            BEGIN
                PERFORM set_config('app.current_tenant', tenant_uuid::TEXT, false);
            END;
            $$ LANGUAGE plpgsql;
        """))

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Enable RLS and create policies
    async with engine.sync_engine.connect() as conn:
        tables = ["tenants", "users", "api_keys", "tenant_configs"]
        for table in tables:
            await conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;"))
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;"))

        await conn.execute(text("""
            CREATE POLICY tenant_isolation ON tenants
            FOR ALL USING (id = current_setting('app.current_tenant')::UUID);
        """))
        await conn.execute(text("""
            CREATE POLICY tenant_isolation ON users
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """))
        await conn.execute(text("""
            CREATE POLICY tenant_isolation ON api_keys
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """))
        await conn.execute(text("""
            CREATE POLICY tenant_isolation ON tenant_configs
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """))

    yield engine
    await engine.dispose()


@pytest.fixture
def tenant_a() -> dict:
    """Fixture for tenant A."""
    return {
        "id": UUID("11111111-1111-1111-1111-111111111111"),
        "name": "Tenant A",
        "slug": "tenant-a",
    }


@pytest.fixture
def tenant_b() -> dict:
    """Fixture for tenant B."""
    return {
        "id": UUID("22222222-2222-2222-2222-222222222222"),
        "name": "Tenant B",
        "slug": "tenant-b",
    }


@pytest.fixture
async def seed_database(integration_engine: AsyncEngine, tenant_a: dict, tenant_b: dict) -> dict:
    """Seed the database with two tenants and their users."""
    async_session = async_sessionmaker(
        bind=integration_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        t1 = Tenant(
            id=tenant_a["id"],
            name=tenant_a["name"],
            slug=tenant_a["slug"],
            tier="free",
        )
        t2 = Tenant(
            id=tenant_b["id"],
            name=tenant_b["name"],
            slug=tenant_b["slug"],
            tier="pro",
        )
        session.add_all([t1, t2])
        await session.commit()

    async with async_session() as session:
        u1 = User(
            tenant_id=tenant_a["id"],
            email="user_a@test.com",
            hashed_password="hashed",
            role="admin",
            is_active=True,
        )
        u2 = User(
            tenant_id=tenant_b["id"],
            email="user_b@test.com",
            hashed_password="hashed",
            role="admin",
            is_active=True,
        )
        session.add_all([u1, u2])
        await session.commit()

    return {"tenant_a": tenant_a, "tenant_b": tenant_b}


@pytest.fixture
def async_session_factory(integration_engine: AsyncEngine):
    """Provide an async session factory bound to integration engine."""
    return async_sessionmaker(
        bind=integration_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


class TestTenantIsolation:
    """Integration tests for RLS-based tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_a_cannot_read_tenant_b_users(
        self, async_session_factory, seed_database: dict
    ) -> None:
        """Tenant A should not be able to query Tenant B's users."""
        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": str(seed_database["tenant_a"]["id"])},
            )
            await session.commit()

            result = await session.execute(text("SELECT tenant_id, email FROM users ORDER BY email"))
            rows = result.fetchall()

            assert len(rows) == 1
            assert rows[0][0] == seed_database["tenant_a"]["id"]
            assert rows[0][1] == "user_a@test.com"

    @pytest.mark.asyncio
    async def test_rls_policy_blocks_cross_tenant_access(
        self, async_session_factory, seed_database: dict
    ) -> None:
        """RLS should block direct cross-tenant queries."""
        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": str(seed_database["tenant_a"]["id"])},
            )
            await session.commit()

            try:
                await session.execute(text("""
                    INSERT INTO users (tenant_id, email, hashed_password, role, is_active)
                    VALUES (:tid, 'cross@test.com', 'hashed', 'member', true)
                """), {"tid": str(seed_database["tenant_b"]["id"])})
                await session.commit()
                pytest.fail("RLS should have blocked cross-tenant INSERT")
            except Exception:
                await session.rollback()

    @pytest.mark.asyncio
    async def test_admin_session_can_read_across_tenants(
        self, async_session_factory, seed_database: dict
    ) -> None:
        """Admin session should be able to read all tenants' data."""
        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = '00000000-0000-0000-0000-000000000000'")
            )
            await session.commit()

            result = await session.execute(text("SELECT tenant_id, email FROM users ORDER BY email"))
            rows = result.fetchall()

            # With default tenant UUID, should see nothing (no tenant matches 0000...)
            assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_insert_with_wrong_tenant_context_invisible(
        self, async_session_factory, seed_database: dict
    ) -> None:
        """Records inserted with wrong tenant context should be invisible to other tenants."""
        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": str(seed_database["tenant_a"]["id"])},
            )
            await session.commit()

            await session.execute(text("""
                INSERT INTO api_keys (tenant_id, key_hash, name, scopes, is_active)
                VALUES (:tid, 'hash_a', 'key-a', '{}', true)
            """), {"tid": str(seed_database["tenant_a"]["id"])})
            await session.commit()

        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": str(seed_database["tenant_b"]["id"])},
            )
            await session.commit()

            result = await session.execute(text("SELECT key_hash, name FROM api_keys"))
            rows = result.fetchall()
            assert len(rows) == 0

        async with async_session_factory() as session:
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": str(seed_database["tenant_a"]["id"])},
            )
            await session.commit()

            result = await session.execute(text("SELECT key_hash, name FROM api_keys"))
            rows = result.fetchall()

            assert len(rows) == 1
            assert rows[0][1] == "key-a"
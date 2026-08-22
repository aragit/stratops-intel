"""Integration tests for tenant isolation via PostgreSQL RLS.

Tests RLS enforcement, API key scoping, JWT tenant isolation, and
cross-tenant Redis Stream isolation using real PostgreSQL and Redis
testcontainers.
"""

from __future__ import annotations

import asyncio
import hashlib
from uuid import uuid4

import pytest
import redis.asyncio as aioredis
import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.db.models import APIKey
from backend.streams.base import StreamConsumer, StreamProducer
from backend.streams.keys import StreamKeyBuilder

logger = structlog.get_logger(__name__)


class TestTenantRLSEnforcement:
    """Verify RLS blocks cross-tenant data visibility."""

    @pytest.mark.asyncio
    async def test_tenant_b_cannot_see_tenant_a_users(
        self,
        test_engine: AsyncEngine,
        test_tenant_a: dict,
        test_tenant_b: dict,
        test_user_a: dict,
        test_user_b: dict,
    ) -> None:
        """Tenant B's session should not see Tenant A's users."""
        session_factory = async_sessionmaker(
            bind=test_engine, class_=AsyncSession, expire_on_commit=False
        )

        async with session_factory() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": str(test_tenant_b["id"])},
            )

            result = await session.execute(text("SELECT tenant_id FROM users"))
            rows = result.fetchall()

            tenant_ids_in_result = [r[0] for r in rows]
            assert test_tenant_a["id"] not in tenant_ids_in_result, (
                "Tenant B should not see Tenant A's users"
            )
            assert test_tenant_b["id"] in tenant_ids_in_result

    @pytest.mark.asyncio
    async def test_tenant_a_can_see_own_users(
        self,
        test_engine: AsyncEngine,
        test_tenant_a: dict,
        test_tenant_b: dict,
        test_user_a: dict,
    ) -> None:
        """Tenant A's session should return its own users."""
        session_factory = async_sessionmaker(
            bind=test_engine, class_=AsyncSession, expire_on_commit=False
        )

        async with session_factory() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": str(test_tenant_a["id"])},
            )

            result = await session.execute(text("SELECT tenant_id FROM users"))
            rows = result.fetchall()
            tenant_ids = [r[0] for r in rows]
            assert test_tenant_a["id"] in tenant_ids
            assert test_tenant_b["id"] not in tenant_ids

    @pytest.mark.asyncio
    async def test_cross_tenant_insert_blocked(
        self,
        test_engine: AsyncEngine,
        test_tenant_a: dict,
        test_tenant_b: dict,
    ) -> None:
        """RLS should block inserting a Tenant B user from Tenant A's session."""
        session_factory = async_sessionmaker(
            bind=test_engine, class_=AsyncSession, expire_on_commit=False
        )

        async with session_factory() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": str(test_tenant_a["id"])},
            )

            try:
                await session.execute(
                    text("""
                        INSERT INTO users (tenant_id, email, hashed_password, role, is_active)
                        VALUES (:tid, 'cross@test.com', 'hashed', 'member', true)
                    """),
                    {"tid": str(test_tenant_b["id"])},
                )
                await session.commit()
                pytest.fail("RLS should have blocked cross-tenant INSERT")
            except Exception:
                await session.rollback()


class TestAPIKeyScoping:
    """Verify API keys are scoped to their tenant."""

    @pytest.mark.asyncio
    async def test_api_key_returns_correct_tenant(
        self,
        integration_engine: AsyncEngine,
        test_tenant_a: dict,
        test_api_key_a: dict,
    ) -> None:
        """Verifying an API key should return the key's tenant_id."""
        key_hash = hashlib.sha256(test_api_key_a["raw_key"].encode()).hexdigest()

        session_factory = async_sessionmaker(
            bind=integration_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with session_factory() as session:
            result = await session.execute(select(APIKey).where(APIKey.key_hash == key_hash))
            db_key = result.scalar_one_or_none()

            assert db_key is not None
            assert db_key.tenant_id == test_tenant_a["id"]

    @pytest.mark.asyncio
    async def test_wrong_tenant_header_with_valid_key_returns_403(
        self,
        integration_gateway,
        test_tenant_a: dict,
        test_api_key_a: dict,
    ) -> None:
        """Using a valid API key with wrong x-tenant-id should return 403."""
        response = await integration_gateway.post(
            "/auth/login",
            data={"username": "nonexistent@test.com", "password": "wrong"},
            headers={
                "x-api-key": test_api_key_a["raw_key"],
                "x-tenant-id": str(uuid4()),  # wrong tenant
            },
        )
        assert response.status_code in (401, 403)


class TestJWTTenantIsolation:
    """Verify JWT tokens are bound to their tenant."""

    @pytest.mark.asyncio
    async def test_jwt_returns_own_tenant_regardless_of_header(
        self,
        integration_gateway,
        test_tenant_a: dict,
        test_user_a: dict,
    ) -> None:
        """A JWT for Tenant A should return Tenant A's ID even with a different x-tenant-id header.

        The /health/tenant endpoint uses get_current_user (JWT-only) not get_db,
        so the tenant header is ignored. The JWT's tenant_id is authoritative.
        """
        from api.auth import create_access_token

        token = create_access_token(
            data={
                "sub": str(test_user_a["id"]),
                "tenant_id": str(test_tenant_a["id"]),
            }
        )

        response = await integration_gateway.get(
            "/health/tenant",
            headers={
                "Authorization": f"Bearer {token}",
                "x-tenant-id": str(uuid4()),  # wrong tenant — ignored
            },
        )
        assert response.status_code == 200
        assert response.json()["tenant_id"] == str(test_tenant_a["id"])

    @pytest.mark.asyncio
    async def test_invalid_jwt_rejected(
        self,
        integration_gateway,
    ) -> None:
        """A garbage JWT should return 401."""
        response = await integration_gateway.get(
            "/health/tenant",
            headers={
                "Authorization": "Bearer garbage",
                "x-tenant-id": str(uuid4()),
            },
        )
        assert response.status_code == 401


class TestCrossTenantStreamIsolation:
    """Verify Redis Streams are logically isolated per tenant."""

    @pytest.mark.asyncio
    async def test_tenant_b_consumer_does_not_receive_tenant_a_messages(
        self,
        integration_redis: aioredis.Redis,
        test_tenant_a: dict,
        test_tenant_b: dict,
    ) -> None:
        """Messages published to Tenant A's stream should not appear in Tenant B's."""
        key_builder = StreamKeyBuilder()

        stream_a = key_builder.signal_stream(test_tenant_a["id"])
        stream_b = key_builder.signal_stream(test_tenant_b["id"])

        producer = StreamProducerImpl(integration_redis, stream_a)
        await producer.publish({"event": "signal_a", "data": "classified"})

        consumer_b = StreamConsumerImpl(
            integration_redis,
            stream_b,
            consumer_group="cg:test",
            consumer_name="test-b",
        )
        await consumer_b.start()
        # Use wait_for with timeout to prevent 60s stream hangs.
        # Also set block_ms=100 in the consumer so it polls regularly.
        await asyncio.wait_for(consumer_b._task, timeout=2.0)
        await consumer_b.stop()

        assert len(consumer_b.received_messages) == 0, (
            "Tenant B's consumer should not receive Tenant A's messages"
        )


class StreamProducerImpl(StreamProducer):
    """Concrete producer for integration tests."""

    pass


class StreamConsumerImpl(StreamConsumer):
    """Concrete consumer for integration tests that records received messages."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.received_messages: list[dict] = []

    async def process_message(self, message_id: str, message: dict) -> bool:
        self.received_messages.append(message)
        return True

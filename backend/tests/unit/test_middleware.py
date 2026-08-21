"""Unit tests for gateway middleware (tenant context + distributed rate limiter)."""

from __future__ import annotations

from unittest import mock

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from backend.api.middleware import RateLimitMiddleware, TenantContextMiddleware


@pytest.fixture
async def fake_redis():
    """Provide a fresh in-memory Redis bound to the test's event loop."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _build_app(redis_client, max_requests: int = 3) -> FastAPI:
    """Build a minimal app wired like the gateway (TenantContext outermost)."""
    app = FastAPI()
    app.state.redis = redis_client

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    # Execution order: TenantContext -> RateLimit -> route
    app.add_middleware(RateLimitMiddleware, max_requests=max_requests, window_seconds=60)
    app.add_middleware(TenantContextMiddleware)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    """Create an async client that shares the test's event loop with the app."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class TestRateLimitMiddleware:
    """Tests for the Redis sliding-window rate limiter."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, fake_redis):
        async with _client(_build_app(fake_redis, max_requests=3)) as client:
            for _ in range(3):
                response = await client.get(
                    "/ping", headers={"x-tenant-id": "00000000-0000-0000-0000-000000000001"}
                )
                assert response.status_code == 200
                assert response.headers["X-RateLimit-Limit"] == "3"

            assert int(response.headers["X-RateLimit-Remaining"]) == 0

    @pytest.mark.asyncio
    async def test_returns_429_over_limit(self, fake_redis):
        async with _client(_build_app(fake_redis, max_requests=2)) as client:
            assert (await client.get("/ping")).status_code == 200
            assert (await client.get("/ping")).status_code == 200

            limited = await client.get("/ping")
            assert limited.status_code == 429
            assert limited.json()["detail"] == "Rate limit exceeded"
            assert limited.headers["Retry-After"] == "60"
            assert limited.headers["X-RateLimit-Remaining"] == "0"

    @pytest.mark.asyncio
    async def test_tenant_scoped_isolation(self, fake_redis):
        """One tenant exhausting its quota must not affect another tenant."""
        tenant_a = "00000000-0000-0000-0000-000000000001"
        tenant_b = "00000000-0000-0000-0000-000000000002"

        async with _client(_build_app(fake_redis, max_requests=1)) as client:
            assert (await client.get("/ping", headers={"x-tenant-id": tenant_a})).status_code == 200
            assert (await client.get("/ping", headers={"x-tenant-id": tenant_a})).status_code == 429
            assert (await client.get("/ping", headers={"x-tenant-id": tenant_b})).status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_key_includes_tenant_and_ip(self, fake_redis):
        tenant_id = "00000000-0000-0000-0000-000000000042"

        async with _client(_build_app(fake_redis, max_requests=5)) as client:
            await client.get("/ping", headers={"x-tenant-id": tenant_id})

        keys = await fake_redis.keys("ratelimit:*")
        assert keys == [f"ratelimit:{tenant_id}:127.0.0.1"]

    @pytest.mark.asyncio
    async def test_anonymous_fallback_key(self, fake_redis):
        async with _client(_build_app(fake_redis, max_requests=5)) as client:
            await client.get("/ping")

        keys = await fake_redis.keys("ratelimit:*")
        assert keys == ["ratelimit:anonymous:127.0.0.1"]

    @pytest.mark.asyncio
    async def test_fail_open_when_redis_unavailable(self):
        """Requests proceed when the Redis backend raises."""
        broken_redis = mock.AsyncMock()
        broken_redis.eval.side_effect = ConnectionError("redis down")

        async with _client(_build_app(broken_redis, max_requests=1)) as client:
            assert (await client.get("/ping")).status_code == 200

    @pytest.mark.asyncio
    async def test_fail_open_without_redis_backend(self):
        """Requests proceed when no redis client is attached to app state."""
        async with _client(_build_app(None)) as client:
            assert (await client.get("/ping")).status_code == 200


class TestTenantContextMiddleware:
    """Tests for tenant header extraction."""

    @pytest.mark.asyncio
    async def test_valid_uuid_sets_state(self, fake_redis):
        captured: dict = {}
        app = FastAPI()
        app.state.redis = fake_redis

        @app.get("/whoami")
        async def whoami(request: Request) -> dict:
            captured["tenant_id"] = getattr(request.state, "tenant_id", None)
            return {"ok": True}

        app.add_middleware(TenantContextMiddleware)

        async with _client(app) as client:
            response = await client.get(
                "/whoami", headers={"x-tenant-id": "00000000-0000-0000-0000-000000000001"}
            )

        assert response.status_code == 200
        assert str(captured["tenant_id"]) == "00000000-0000-0000-0000-000000000001"

    @pytest.mark.asyncio
    async def test_invalid_uuid_rejected(self, fake_redis):
        app = FastAPI()
        app.state.redis = fake_redis

        @app.get("/ping")
        async def ping() -> dict:
            return {"ok": True}

        app.add_middleware(TenantContextMiddleware)

        async with _client(app) as client:
            response = await client.get("/ping", headers={"x-tenant-id": "not-a-uuid"})

        assert response.status_code == 400

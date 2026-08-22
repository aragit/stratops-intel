"""Unit tests for gateway middleware (tenant context + distributed rate limiter)."""

from __future__ import annotations

from unittest import mock

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from backend.api.middleware import RateLimitMiddleware, TenantContextMiddleware


class _RateLimitEvalMock:
    """Mock the Lua script execution for the rate limiter sliding-window algorithm.

    The script returns [1, remaining] when allowed, [0, 0] when over limit.
    """

    def __init__(self, redis_client):
        self._redis = redis_client
        self._keys: set = set()
        self._counts: dict = {}  # key -> count

    async def __call__(self, script, *keys_and_args):
        # Extract KEYS[1] and args from the script call
        # eval(script, numkeys, *keys, *args)
        # In our usage: eval(script, 1, key, now_ms, window_ms, limit, member)
        _numkeys = keys_and_args[0] if len(keys_and_args) > 0 else 1
        key = keys_and_args[1] if len(keys_and_args) > 1 else None
        _now_ms = keys_and_args[2] if len(keys_and_args) > 2 else None
        _window_ms = keys_and_args[3] if len(keys_and_args) > 3 else None
        limit = keys_and_args[4] if len(keys_and_args) > 4 else None
        _member = keys_and_args[5] if len(keys_and_args) > 5 else None

        if key is None:
            return [0, 0]

        self._keys.add(key)
        # Initialize count if not present
        if key not in self._counts:
            self._counts[key] = 0

        # Simulate the Lua algorithm: track count and enforce limit
        self._counts[key] = self._counts.get(key, 0) + 1
        count = self._counts[key]

        # Also store the key in fakeredis so that fake_redis.keys() works
        try:
            await self._redis.set(key, "1")
        except Exception:
            pass

        if count <= limit:
            remaining = limit - count
            return [1, remaining]
        return [0, 0]


@pytest.fixture
async def fake_redis():
    """Provide a fresh in-memory Redis bound to the test's event loop."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Patch eval to simulate the rate limiter Lua script execution
    eval_mock = _RateLimitEvalMock(redis)
    redis.eval = eval_mock.__call__

    return redis


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

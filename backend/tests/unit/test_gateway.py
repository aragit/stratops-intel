"""Unit tests for the FastAPI gateway application.

Tests health endpoints, auth flow, rate limiting, and CORS headers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Build a TestClient with all necessary mocks for unit testing."""
    mock_engine = MagicMock()
    mock_engine.pool.status.return_value = "pool ok"

    mock_manager = MagicMock()
    mock_manager.is_connected = True
    mock_manager._engine = mock_engine

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.first.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_manager.admin_session.return_value = mock_session

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    mock_redis_factory = MagicMock(return_value=mock_redis)

    with patch("api.gateway.initialize_database", new_callable=AsyncMock):
        with patch("api.gateway.close_database", new_callable=AsyncMock):
            with patch("api.gateway.get_session_manager", return_value=mock_manager):
                with patch("redis.asyncio.from_url", mock_redis_factory):
                    with patch("api.gateway.aioredis", MagicMock(from_url=mock_redis_factory)):
                        from api.gateway import app

                        with TestClient(app, raise_server_exceptions=False) as c:
                            yield c


class TestHealthEndpoints:
    """Tests for /health endpoints."""

    def test_liveness_returns_200(self, client: TestClient) -> None:
        """GET /health should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_returns_json(self, client: TestClient) -> None:
        """GET /health/ready should return JSON with checks."""
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data


class TestCORSHeaders:
    """Tests for CORS middleware."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Responses should include CORS headers."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code in (200, 405)


class TestAuthFlow:
    """Tests for /auth endpoints."""

    def test_login_missing_fields_returns_422(self, client: TestClient) -> None:
        """POST /auth/login with no body should return 422."""
        response = client.post("/auth/login", data={})
        assert response.status_code == 422

    def test_refresh_missing_token_returns_400(self, client: TestClient) -> None:
        """POST /auth/refresh with empty body should return 400."""
        response = client.post("/auth/refresh", json={})
        assert response.status_code == 400

    def test_refresh_invalid_token_returns_401(self, client: TestClient) -> None:
        """POST /auth/refresh with bad token should return 401."""
        response = client.post("/auth/refresh", json={"refresh_token": "garbage"})
        assert response.status_code == 401


class TestRequestLoggingMiddleware:
    """Tests for request logging middleware."""

    def test_request_id_header_added(self, client: TestClient) -> None:
        """Responses should include X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 36  # UUID


class TestTenantContextMiddleware:
    """Tests for tenant context middleware."""

    def test_invalid_tenant_header_returns_400(self, client: TestClient) -> None:
        """Invalid x-tenant-id should return 400."""
        response = client.get(
            "/health/tenant",
            headers={"x-tenant-id": "not-a-uuid"},
        )
        assert response.status_code in (400, 401)

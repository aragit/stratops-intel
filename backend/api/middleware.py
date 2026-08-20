"""ASGI middleware for the FastAPI gateway.

Provides tenant context extraction, request logging, and per-tenant
in-memory rate limiting.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Extract ``x-tenant-id`` header and attach it to ``request.state``.

    If the header is present and is a valid UUID it is stored as
    ``request.state.tenant_id``.  Invalid values are logged and result in
    a 400 response.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request and set tenant context on state.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream.
        """
        from uuid import UUID

        tenant_header = request.headers.get("x-tenant-id")
        if tenant_header is not None:
            try:
                request.state.tenant_id = UUID(tenant_header)
            except (ValueError, TypeError):
                logger.warning("invalid_tenant_header", raw=tenant_header)
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid x-tenant-id header format"},
                )

        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with structlog (request_id, tenant_id, method, path, status, duration)."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Log the request lifecycle.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream.
        """
        from uuid import uuid4

        request_id = str(uuid4())
        request.state.request_id = request_id
        tenant_id = getattr(request.state, "tenant_id", None)

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        logger.info(
            "http_request",
            request_id=request_id,
            tenant_id=str(tenant_id) if tenant_id else None,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per tenant.

    Default limit is 100 requests per minute per tenant.  When the limit
    is exceeded the request is rejected with 429.
    """

    def __init__(self, app: Any, max_requests: int = 100, window_seconds: int = 60) -> None:
        """Initialise the rate limiter.

        Args:
            app: The ASGI application (injected by Starlette middleware stack).
            max_requests: Maximum requests allowed in the window.
            window_seconds: Sliding window size in seconds.
        """
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _get_client_key(self, request: Request) -> str:
        """Derive a rate-limit key from tenant_id or client IP.

        Args:
            request: The incoming request.

        Returns:
            A string key for the rate limiter.
        """
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id is not None:
            return f"tenant:{tenant_id}"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Enforce the rate limit.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream, or 429 if rate limited.
        """
        key = self._get_client_key(request)
        now = time.time()
        window_start = now - self._window_seconds

        hits = self._hits[key]
        self._hits[key] = [t for t in hits if t > window_start]

        if len(self._hits[key]) >= self._max_requests:
            logger.warning("rate_limit_exceeded", key=key, limit=self._max_requests)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(self._window_seconds)},
            )

        self._hits[key].append(now)
        response = await call_next(request)
        remaining = self._max_requests - len(self._hits[key])
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response

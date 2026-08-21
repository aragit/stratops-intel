"""ASGI middleware for the FastAPI gateway.

Provides tenant context extraction, request logging, and distributed
per-tenant/per-IP rate limiting backed by Redis.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Atomic sliding-window rate limiter.
#
# KEYS[1]: sorted-set key holding request timestamps (ms score) with unique
#          members per request.
# ARGV[1]: current time in ms
# ARGV[2]: window size in ms
# ARGV[3]: max requests allowed within the window
# ARGV[4]: unique member id for this request
#
# Returns {allowed(0|1), remaining_requests}.
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window_ms)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('PEXPIRE', key, window_ms)
    return {1, limit - count - 1}
end

redis.call('PEXPIRE', key, window_ms)
return {0, 0}
"""


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Extract ``x-tenant-id`` header and attach it to ``request.state``.

    If the header is present and is a valid UUID it is stored as
    ``request.state.tenant_id``.  Invalid values are logged and result in
    a 400 response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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

        response: Response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with structlog (request_id, tenant_id, method, path, status, duration)."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
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
    """Distributed sliding-window rate limiter backed by Redis.

    Each request records a timestamp in a tenant-scoped sorted set keyed by
    ``ratelimit:{tenant_id}:{client_ip}`` (``ratelimit:anonymous:{ip}`` when
    no tenant context is present). An atomic Lua script prunes expired
    entries, enforces the limit, and returns the remaining quota, making the
    counter correct across all gateway replicas sharing the Redis backend.

    When Redis is unreachable or returns an unexpected reply the middleware
    fails open (the request proceeds) so a cache outage cannot take the API
    down; the event is logged for alerting.
    """

    def __init__(
        self,
        app: Any,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit",
    ) -> None:
        """Initialise the rate limiter.

        Args:
            app: The ASGI application (injected by Starlette middleware stack).
            max_requests: Maximum requests allowed in the window.
            window_seconds: Sliding window size in seconds.
            key_prefix: Prefix for Redis rate-limit keys.
        """
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._window_ms = window_seconds * 1000
        self._key_prefix = key_prefix
        self._script = _SLIDING_WINDOW_LUA

    def _get_client_ip(self, request: Request) -> str:
        """Resolve the client IP, honouring the first forwarded hop."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_rate_limit_key(self, request: Request) -> str:
        """Build the tenant-scoped rate-limit key.

        Args:
            request: The incoming request.

        Returns:
            A key of the form ``{prefix}:{tenant_id}:{client_ip}``. Requests
            without tenant context fall back to ``{prefix}:anonymous:{ip}``.
        """
        tenant_id = getattr(request.state, "tenant_id", None)
        tenant_part = str(tenant_id) if tenant_id else "anonymous"
        return f"{self._key_prefix}:{tenant_part}:{self._get_client_ip(request)}"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Enforce the distributed rate limit.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The response from downstream, or 429 if rate limited.
        """
        redis_client = getattr(getattr(request.app, "state", None), "redis", None)
        if redis_client is None:
            logger.warning("rate_limit_backend_unavailable", reason="redis_not_initialized")
            response = await call_next(request)
            return response

        key = self._get_rate_limit_key(request)
        now_ms = int(time.time() * 1000)
        member = f"{now_ms}-{uuid4().hex}"

        try:
            result = await redis_client.eval(
                self._script,
                1,
                key,
                now_ms,
                self._window_ms,
                self._max_requests,
                member,
            )
        except Exception as exc:
            logger.warning(
                "rate_limit_backend_error",
                key=key,
                error=str(exc),
            )
            response = await call_next(request)
            return response

        # Fail open on unexpected replies (e.g. mocked or incompatible backends)
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            response = await call_next(request)
            return response

        allowed = int(result[0]) == 1
        remaining = max(0, int(result[1]))

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                key=key,
                limit=self._max_requests,
                window_seconds=self._window_seconds,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "Retry-After": str(self._window_seconds),
                    "X-RateLimit-Limit": str(self._max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

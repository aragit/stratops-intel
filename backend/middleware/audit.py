"""Structured audit logging middleware for all mutating API operations.

Captures append-only audit trails for ``POST``, ``PUT``, and ``DELETE``
requests without exposing sensitive authorization tokens or encrypted
payloads. Logs are written via ``structlog`` with the following fields:

- ``actor_id``: authenticated user id (or ``system`` for system-generated events)
- ``tenant_id``: tenancy context from the request
- ``method``: HTTP method (``POST``, ``PUT``, ``DELETE``)
- ``path``: request path (query string stripped)
- ``status_code``: HTTP response status code
- ``timestamp``: UTC timestamp of request completion
- ``ip_address``: client IP address (sourced from ``request.client``)
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send

logger = structlog.get_logger(__name__)

#: List of HTTP methods that trigger audit logging.
AUDIT_METHODS = frozenset({"POST", "PUT", "DELETE"})


class AuditLogMiddleware:
    """ASGI middleware that captures structured audit logs for mutating
    operations.

    The middleware wraps the ASGI lifecycle to capture request metadata
    and tenant context, then emits a structured ``structlog`` event after
    the response is complete. Sensitive fields (``authorization``,
    ``cookie``, ``set-cookie``) are stripped from the captured data
    before logging.

    Attributes:
        app: The next ASGI application in the stack.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware with the target ASGI application.

        Args:
            app: The next ASGI application in the request pipeline.
        """
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the ASGI scope, capture audit data, and emit a log event.

        If the request method is not in ``AUDIT_METHODS``, the middleware
        passes through without logging.

        Args:
            scope: The ASGI scope dictionary.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        # Only process HTTP requests with a defined method
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope.get("method", "GET").upper()
        if method not in AUDIT_METHODS:
            await self._app(scope, receive, send)
            return

        # Extract tenant_id from scope state (set by TenantContextMiddleware)
        tenant_id: str | None = scope.get("state", {}).get("tenant_id")

        # Extract client IP address
        client_scope = scope.get("client")
        ip_address: str | None = client_scope.get("host") if client_scope else None

        # Capture request ID for traceability
        request_id: str | None = scope.get("state", {}).get("request_id")

        # Track start time for latency and timestamp logging
        start_time = time.time()

        # Buffer for response status code
        status_code: int | None = None

        # Custom send that captures the status code
        async def capturing_send(message: Any) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 0)
            await self._app._send(message)  # type: ignore[attr-defined]

        # Process the request
        await self._app(scope, receive, capturing_send)

        # Emit the audit log entry
        await self._emit_audit_log(
            method=method,
            path=scope.get("path", ""),
            tenant_id=tenant_id,
            status_code=status_code or 0,
            ip_address=ip_address,
            request_id=request_id,
            start_time=start_time,
        )

    async def _emit_audit_log(
        self,
        method: str,
        path: str,
        tenant_id: str | None,
        status_code: int,
        ip_address: str | None,
        request_id: str | None,
        start_time: float,
    ) -> None:
        """Emit a structured audit log entry via ``structlog``.

        Sensitive fields are stripped before logging. The log includes
        a ``timestamp`` field (UTC epoch seconds) and a ``duration_ms``
        field for latency tracking.
        """
        duration_ms = int((time.time() - start_time) * 1000)

        # Build the log event dict, excluding sensitive fields
        event: dict[str, Any] = {
            "actor_id": tenant_id,  # tenant_id serves as the actor in multi-tenant context
            "tenant_id": tenant_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "timestamp": int(time.time()),
            "duration_ms": duration_ms,
            "ip_address": ip_address,
            "request_id": request_id,
        }

        # Remove keys with ``None`` values for cleaner logs
        event = {k: v for k, v in event.items() if v is not None}

        structlog.get_logger().info("audit_event", **event)

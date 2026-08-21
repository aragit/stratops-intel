"""Unit tests for the AuditLogMiddleware structured audit logging module."""

from __future__ import annotations

from unittest import mock

import pytest


class TestAuditLogMiddlewareInit:
    """Tests for AuditLogMiddleware construction."""

    def test_init_stores_app(self) -> None:
        """Init should store the ASGI app."""
        from backend.middleware.audit import AuditLogMiddleware

        mock_app = mock.MagicMock()
        middleware = AuditLogMiddleware(mock_app)
        assert middleware._app is mock_app


class TestAuditLogMiddlewareMethods:
    """Tests for AuditLogMiddleware methods."""

    @pytest.mark.asyncio
    @mock.patch("backend.middleware.audit.structlog.get_logger")
    async def test_emit_audit_log_structure(self, mock_logger) -> None:
        """emit_audit_log should call structlog with the expected fields."""
        from backend.middleware.audit import AuditLogMiddleware

        middleware = AuditLogMiddleware(mock.MagicMock())

        # Call the emit method with test data
        # The _emit_audit_log method is async, we need to await it
        await middleware._emit_audit_log(
            method="POST",
            path="/api/signals",
            tenant_id="t-1",
            status_code=201,
            ip_address="127.0.0.1",
            request_id="req-123",
            start_time=1000.0,
        )

        # Verify structlog was called - mock_logger is the mock of get_logger
        mock_logger.return_value.info.assert_called_once()

        # Verify the event dict contains the expected keys
        call_args = mock_logger.return_value.info.call_args
        call_kwargs = call_args[1] if call_args and len(call_args) > 1 else {}
        expected_keys = {
            "method",
            "path",
            "status_code",
            "timestamp",
            "duration_ms",
            "ip_address",
            "request_id",
            "tenant_id",
        }
        assert expected_keys.issubset(set(call_kwargs.keys()))


class TestAuditLogMiddlewareFiltering:
    """Tests that AuditLogMiddleware only logs audited methods."""

    def test_audited_methods(self) -> None:
        """POST/PUT/DELETE should be audited; GET should not."""
        from backend.middleware.audit import AUDIT_METHODS

        assert "POST" in AUDIT_METHODS
        assert "GET" not in AUDIT_METHODS
        assert "PUT" in AUDIT_METHODS
        assert "DELETE" in AUDIT_METHODS

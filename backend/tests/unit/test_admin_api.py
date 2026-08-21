"""Unit tests for the Admin API endpoints."""

from __future__ import annotations

from backend.api.v1.admin import router


class TestAdminAPI:
    """Tests for the admin API router."""

    def test_router_has_endpoints(self) -> None:
        """Router should have the three expected endpoints."""
        routes = [route.path for route in router.routes]
        assert "/v1/admin/tenants/{tenant_id}/costs" in routes
        assert "/v1/admin/tenants/{tenant_id}/purge" in routes
        assert "/v1/admin/health/system" in routes

    def test_costs_endpoint_path(self) -> None:
        """Costs endpoint path should contain 'costs'."""
        routes = [route.path for route in router.routes]
        costs_route = [r for r in routes if "costs" in r]
        assert len(costs_route) == 1

    def test_purge_endpoint_path(self) -> None:
        """Purge endpoint path should contain 'purge'."""
        routes = [route.path for route in router.routes]
        purge_route = [r for r in routes if "purge" in r]
        assert len(purge_route) == 1

    def test_health_endpoint_path(self) -> None:
        """Health endpoint path should contain 'health'."""
        routes = [route.path for route in router.routes]
        health_route = [r for r in routes if "health" in r]
        assert len(health_route) == 1

"""Admin & Tenant Management API for StratOps-Intel.

Exposes administrative endpoints for tenant metrics, system health,
and on-demand data retention cleanup.

Endpoints
---------
- ``GET /v1/admin/tenants/{id}/costs``: Detailed token usage and USD
  cost summary for a tenant.
- ``POST /v1/admin/tenants/{id}/purge``: Triggers data retention
  cleanup on demand for a tenant.
- ``GET /v1/admin/health/system``: Aggregated health check for
  Postgres, Neo4j, Redis, and MinIO.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.billing.cost_tracker import CostTracker
from backend.security.encryption import FieldEncryptor
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1/admin", tags=["admin"])


# Dependency: retrieve the active CostTracker instance from the app state
def _get_cost_tracker(app: Any) -> CostTracker:  # noqa: ARG001
    return app.state.cost_tracker  # type: ignore[no-any-return]


def _get_encryptor(app: Any) -> FieldEncryptor:  # noqa: ARG001
    return app.state.encryptor  # type: ignore[no-any-return]


@router.get(
    "/tenants/{tenant_id}/costs",
    summary="Retrieve tenant cost summary",
    description="Returns detailed token usage and USD cost summary for a tenant "
    "within the current billing period.",
)
async def tenant_costs(
    request: Request,
    tenant_id: str,
) -> JSONResponse:
    """Return the tenant's token usage and cost summary.

    The usage is drawn from the ``CostTracker`` which stores cumulative
    prompt/completion tokens and cost in Redis, clamped to the tenant's
    tier limits (Free/Pro/Enterprise).

    Args:
        request: FastAPI request object, used to retrieve the ``CostTracker
            `` instance from ``app.state``.
        tenant_id: Tenant identifier.

    Returns:
        JSON response with token counts, costs, tier, and usage percentages.
    """
    cost_tracker: CostTracker = request.app.state.cost_tracker
    summary = await cost_tracker.get_tenant_costs(tenant_id, None, None)
    return JSONResponse(content=summary)


@router.post(
    "/tenants/{tenant_id}/purge",
    summary="Trigger on-demand data retention purge",
    description="Executes the ``RetentionEngine`` to purge expired signals "
    "and intelligence chunks for the specified tenant. Returns counts of "
    "purged records per table.",
)
async def tenant_purge(
    request: Request,
    tenant_id: str,
) -> JSONResponse:
    """Trigger an on-demand data retention purge for a tenant.

    Retrieves the ``RetentionEngine`` from the application state and
    calls ``purge_expired_data``. The retention threshold is determined
    by the tenant's tier (Free: 90 days, Pro: 365 days, Enterprise:
    2555 days).

    Args:
        request: FastAPI request object, used to retrieve the
            ``RetentionEngine `` instance from ``app.state``.
        tenant_id: Tenant identifier.

    Returns:
        JSON response with counts of purged records per table.
    """
    retention_engine = request.app.state.retention_engine
    result = await retention_engine.purge_expired_data()
    return JSONResponse(content=result)


@router.get(
    "/health/system",
    summary="System aggregated health check",
    description="Returns the health status of core infrastructure: "
    "PostgreSQL, Neo4j, Redis, and MinIO.",
)
async def system_health(
    request: Request,
) -> JSONResponse:
    """Return aggregated health status for all core infrastructure services.

    The health check pings each backend service and reports overall
    status as ``healthy`` if all individual checks pass, or ``degraded``
    if any check fails. Individual service statuses are included in the
    response for detailed diagnostics.

    Args:
        request: FastAPI request object, used to retrieve service clients
            from ``app.state``.

    Returns:
        JSON response with per-service health status and a global
        ``overall`` status.
    """
    # Collect health checks from the application state
    health_checks: dict[str, dict[str, Any]] = {}

    # Postgres check
    try:
        pg_engine = request.app.state.postgres_engine
        async with pg_engine.begin() as conn:
            await conn.execute("SELECT 1")
        health_checks["postgres"] = {"status": "healthy", "detail": "connection successful"}
    except Exception as e:
        health_checks["postgres"] = {"status": "unhealthy", "detail": str(e)}

    # Neo4j check
    try:
        neo4j_client = request.app.state.neo4j_client
        # Simple reachability ping via async context
        async with neo4j_client.get_session() as session:
            await session.run("RETURN 1")
        health_checks["neo4j"] = {"status": "healthy", "detail": "connection successful"}
    except Exception as e:
        health_checks["neo4j"] = {"status": "unhealthy", "detail": str(e)}

    # Redis check
    try:
        redis_client = request.app.state.redis_client
        await redis_client.ping()
        health_checks["redis"] = {"status": "healthy", "detail": "connection successful"}
    except Exception as e:
        health_checks["redis"] = {"status": "unhealthy", "detail": str(e)}

    # MinIO check (MinIO SDK health check)
    try:
        minio_client = request.app.state.minio_client
        # MinIO Python SDK health check
        await minio_client.stat_bucket("health")
        health_checks["minio"] = {"status": "healthy", "detail": "connection successful"}
    except Exception as e:
        health_checks["minio"] = {"status": "unhealthy", "detail": str(e)}

    # Determine overall status
    all_healthy = all(
        h.get("status") == "healthy" for h in health_checks.values()
    )
    overall = "healthy" if all_healthy else "degraded"

    return JSONResponse(
        content={
            "overall": overall,
            "checks": health_checks,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )


# Alias for FastAPI's ``include_router`` compatibility
admin_router = router

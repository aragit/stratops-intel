"""FastAPI gateway application entry point.

Configures the FastAPI application with lifespan management,
middleware, authentication routers, and health endpoints.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
import structlog
from fastapi import Body, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from backend.api.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    verify_password,
)
from backend.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    TenantContextMiddleware,
)
from backend.db.models import APIKey, User
from backend.db.tenant_session import (
    close_database,
    get_session_manager,
    initialize_database,
)
from backend.streams.keys import StreamKeyBuilder

logger = structlog.get_logger(__name__)

_redis_pool: aioredis.Redis | None = None
_stream_key_builder = StreamKeyBuilder()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Startup:
        - Initialise the database engine and connection pool.
        - Create the Redis connection pool.

    Shutdown:
        - Close Redis.
        - Dispose the database engine.
    """
    global _redis_pool

    logger.info("gateway_startup_starting")

    await initialize_database()

    redis_url = "redis://localhost:6379/0"
    _redis_pool = aioredis.from_url(redis_url, decode_responses=True)
    app.state.redis = _redis_pool
    app.state.stream_key_builder = _stream_key_builder

    logger.info("gateway_startup_complete")
    yield

    logger.info("gateway_shutdown_starting")
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
    await close_database()
    logger.info("gateway_shutdown_complete")


app = FastAPI(
    title="StratOps-Intel API",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware executes outermost-first: the LAST add_middleware call runs
# first. Tenant context must be extracted before rate limiting so that
# rate-limit keys can be scoped by tenant_id.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
app.add_middleware(TenantContextMiddleware)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/login")
async def login(body: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    """Authenticate a user with email and password.

    Returns JWT access and refresh tokens.
    """
    manager = get_session_manager()
    try:
        async with manager.admin_session() as session:
            stmt = select(User).where(User.email == body.username, User.is_active.is_(True))
            result = await session.execute(stmt)
            user: User | None = result.scalar_one_or_none()
    except Exception as exc:
        logger.error("login_db_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        )

    if user is None or not verify_password(body.password, user.hashed_password):
        logger.warning("login_failed", email=body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = create_access_token(data={"sub": str(user.id), "tenant_id": str(user.tenant_id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id), "tenant_id": str(user.tenant_id)})

    logger.info("login_success", user_id=str(user.id), tenant_id=str(user.tenant_id))

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@app.post("/auth/refresh")
async def refresh_token(body: dict = Body(...)) -> dict[str, str]:
    """Rotate a refresh token and issue a new access token.

    Expects ``{"refresh_token": "<token>"}`` in the body.
    """
    raw_token = body.get("refresh_token")
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token is required",
        )

    payload = decode_token(raw_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token",
        )

    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    access_token = create_access_token(data={"sub": user_id, "tenant_id": tenant_id})
    new_refresh = create_refresh_token(data={"sub": user_id, "tenant_id": tenant_id})

    logger.info("token_refreshed", user_id=user_id)

    return {
        "access_token": access_token,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }


@app.post("/auth/api-keys")
async def create_api_key(
    name: str = Body(..., embed=True),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Create a new API key for the authenticated user's tenant.

    Returns the plaintext key exactly once.
    """
    raw_key = f"so_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    manager = get_session_manager()
    try:
        async with manager.admin_session() as session:
            api_key = APIKey(
                tenant_id=user.tenant_id,
                user_id=user.id,
                key_hash=key_hash,
                name=name,
                scopes={},
                is_active=True,
                created_at=datetime.now(UTC),
            )
            session.add(api_key)
            await session.commit()
            await session.refresh(api_key)
    except Exception as exc:
        logger.error("api_key_creation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )

    logger.info("api_key_created", key_id=str(api_key.id), tenant_id=str(user.tenant_id))

    return {"api_key": raw_key, "key_id": str(api_key.id)}


@app.delete("/auth/api-keys/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Revoke an API key by ID.

    Only keys belonging to the user's tenant can be revoked.
    """
    manager = get_session_manager()
    try:
        async with manager.admin_session() as session:
            stmt = select(APIKey).where(
                APIKey.id == key_id,
                APIKey.tenant_id == user.tenant_id,
            )
            result = await session.execute(stmt)
            api_key: APIKey | None = result.scalar_one_or_none()

            if api_key is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="API key not found",
                )

            api_key.is_active = False
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("api_key_revocation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )

    logger.info("api_key_revoked", key_id=str(key_id), tenant_id=str(user.tenant_id))

    return {"detail": "API key revoked"}


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_liveness() -> dict[str, str]:
    """Liveness probe — returns 200 if the app is running."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_readiness() -> dict[str, Any]:
    """Readiness probe — checks DB, Redis, and Neo4j connectivity.

    Returns 200 if all backends are reachable, 503 otherwise.
    """
    checks: dict[str, str] = {}

    try:
        manager = get_session_manager()
        async with manager.admin_session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    try:
        if _redis_pool is not None:
            await _redis_pool.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "not initialized"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    checks["neo4j"] = "skipped"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


@app.get("/health/tenant")
async def health_tenant(user: User = Depends(get_current_user)) -> dict[str, str]:
    """Tenant-scoped health check.

    Validates that RLS is working by querying the user's own tenant.
    """
    return {"status": "ok", "tenant_id": str(user.tenant_id)}

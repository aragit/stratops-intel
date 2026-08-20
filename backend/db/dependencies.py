"""FastAPI dependencies for tenant-aware database access.

These dependencies extract tenant and API-key information from the incoming
HTTP request, validate credentials against the database, and yield an
:class:`AsyncSession` with the correct RLS context configured.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID

import structlog
from fastapi import Header, HTTPException, Request, status
from sqlalchemy import select

from db.models import APIKey, Tenant
from db.tenant_session import TenantSessionManager, get_session_manager

logger = structlog.get_logger(__name__)

__all__ = [
    "get_db",
    "get_admin_db",
    "verify_api_key",
]


async def verify_api_key(request: Request) -> tuple[UUID, list[str]]:
    """Validate the ``x-api-key`` header against the ``api_keys`` table.

    Args:
        request: The incoming FastAPI request.

    Returns:
        A ``(tenant_id, scopes)`` tuple on success.

    Raises:
        HTTPException: 401 if the key is missing, invalid, inactive, or
            expired.
    """
    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    manager = get_session_manager()

    try:
        async with manager.admin_session() as session:
            stmt = select(APIKey).where(APIKey.key_hash == key_hash)
            result = await session.execute(stmt)
            db_key: APIKey | None = result.scalar_one_or_none()
    except Exception as exc:
        logger.error("api_key_verification_error", error=str(exc), key_hash=key_hash[:12])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during key verification",
        ) from exc

    if db_key is None:
        logger.warning("api_key_not_found", key_hash=key_hash[:12])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not db_key.is_active:
        logger.warning("api_key_revoked", key_hash=key_hash[:12], tenant_id=str(db_key.tenant_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is revoked",
        )

    if db_key.expires_at and db_key.expires_at < datetime.now(timezone.utc):
        logger.warning("api_key_expired", key_hash=key_hash[:12], tenant_id=str(db_key.tenant_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has expired",
        )

    scopes_raw: object = db_key.scopes
    if isinstance(scopes_raw, list):
        scopes: list[str] = scopes_raw
    elif isinstance(scopes_raw, dict):
        scopes = [k for k, v in scopes_raw.items() if v]
    else:
        scopes = []

    logger.debug(
        "api_key_verified",
        tenant_id=str(db_key.tenant_id),
        key_id=str(db_key.id),
        scopes=scopes,
    )

    return db_key.tenant_id, scopes


async def get_db(
    request: Request,
    api_key_header: str | None = Header(default=None, alias="x-api-key"),
    tenant_header: str | None = Header(default=None, alias="x-tenant-id"),
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a tenant-scoped :class:`AsyncSession`.

    Extracts ``x-tenant-id`` and ``x-api-key`` headers.  When an API key is
    present it is validated via :func:`verify_api_key` and the tenant derived
    from the key is cross-checked against ``x-tenant-id``.

    Args:
        request: The incoming FastAPI request.
        api_key_header: The raw ``x-api-key`` header value (or ``None``).
        tenant_header: The raw ``x-tenant-id`` header value (or ``None``).

    Yields:
        An :class:`AsyncSession` with the tenant RLS context set.

    Raises:
        HTTPException: 401 if neither a valid API key nor a valid tenant ID
            is provided, or 403 on tenant mismatch.
    """
    manager = get_session_manager()
    tenant_id: UUID | None = None
    scopes: list[str] = []

    if api_key_header is not None:
        try:
            verified_tenant_id, scopes = await verify_api_key(request)
            tenant_id = verified_tenant_id
        except HTTPException:
            raise
    elif tenant_header is not None:
        try:
            tenant_id = UUID(tenant_header)
        except (ValueError, TypeError) as exc:
            logger.warning("invalid_tenant_id", raw=tenant_header, error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant ID format",
            )

        try:
            async with manager.admin_session() as session:
                stmt = select(Tenant).where(Tenant.id == tenant_id)
                result = await session.execute(stmt)
                tenant: Tenant | None = result.scalar_one_or_none()
            if tenant is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Unknown tenant",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("tenant_validation_error", tenant_id=str(tenant_id), error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error during tenant validation",
            ) from exc
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Either x-api-key or x-tenant-id header is required",
        )

    # If both API key and tenant-id were provided, ensure they match.
    if api_key_header is not None and tenant_header is not None:
        if tenant_id != UUID(tenant_header):
            logger.warning(
                "tenant_mismatch",
                key_tenant=str(tenant_id),
                header_tenant=tenant_header,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key tenant does not match x-tenant-id",
            )

    request.state.tenant_id = tenant_id
    request.state.scopes = scopes

    logger.debug(
        "db_dependency_resolved",
        tenant_id=str(tenant_id),
        scopes=scopes,
    )

    async with manager.get_session(tenant_id) as session:
        yield session


async def get_admin_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an admin :class:`AsyncSession`.

    The session is created without a real tenant context (the null UUID
    is set) so that RLS policies evaluate but match no tenant data.

    Yields:
        An :class:`AsyncSession` for cross-tenant administrative operations.
    """
    manager = get_session_manager()

    try:
        async with manager.admin_session() as session:
            logger.debug("admin_db_dependency_resolved")
            yield session
    except Exception as exc:
        logger.error("admin_db_dependency_error", error=str(exc))
        raise

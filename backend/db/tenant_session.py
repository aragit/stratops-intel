"""Async database session factory with tenant context management.

This module provides tenant-aware SQLAlchemy session factories that enforce
PostgreSQL Row-Level Security (RLS) by setting the ``app.current_tenant``
session variable on each connection.

The primary entry point is :class:`TenantSessionManager`, which manages an
async engine and connection pool.  Module-level convenience functions are
retained for backward compatibility and delegate to a global singleton.
"""

from __future__ import annotations

import contextvars
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

logger = structlog.get_logger(__name__)

_current_tenant_id: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "current_tenant_id", default=None
)

_default_tenant_uuid = UUID("00000000-0000-0000-0000-000000000000")

_session_manager: TenantSessionManager | None = None


def _get_database_url() -> str:
    """Get the database URL from environment variables.

    Returns:
        A SQLAlchemy-compatible async database URL.
    """
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://stratops:stratops_dev_password@localhost:5432/stratops",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    return database_url


def get_database_url() -> str:
    """Get the current database URL (public alias)."""
    return _get_database_url()


def _is_testing() -> bool:
    """Return True when running under the test suite."""
    return os.getenv("TESTING", "false").lower() == "true"


class TenantSessionManager:
    """Manages an async PostgreSQL engine with per-tenant RLS context injection.

    Each call to :meth:`get_session` optionally sets the ``app.current_tenant``
    PostgreSQL session variable so that RLS policies transparently scope queries
    to the calling tenant.

    Attributes:
        database_url: The async database URL.
        pool_size: Number of connections to maintain in the pool.
        max_overflow: Maximum number of overflow connections.
    """

    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 10) -> None:
        """Initialise the manager (does not connect yet).

        Args:
            database_url: A SQLAlchemy async connection string.
            pool_size: Base pool size (default 20).
            max_overflow: Maximum overflow connections (default 10).
        """
        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None
        self._initialized: bool = False

    async def connect(self) -> None:
        """Create the async engine and session maker.

        Must be called once before any session is requested.  Idempotent:
        a second call is a no-op.
        """
        if self._engine is not None:
            logger.warning("engine_already_initialized")
            return

        engine_kwargs: dict[str, Any] = {
            "echo": os.getenv("ENVIRONMENT", "development").lower() == "development",
            "future": True,
            "connect_args": {
                "server_settings": {
                    "application_name": "stratops-intel",
                    "jit": "off",
                },
                "command_timeout": 60,
            },
        }

        if _is_testing():
            engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["pool_size"] = self._pool_size
            engine_kwargs["max_overflow"] = self._max_overflow
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_recycle"] = 3600
            engine_kwargs["pool_timeout"] = 30

        self._engine = create_async_engine(self._database_url, **engine_kwargs)
        self._session_maker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        self._initialized = True

        logger.info(
            "engine_created",
            database_url=self._database_url.split("@")[-1],
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
        )

    async def dispose(self) -> None:
        """Dispose the engine and release all pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            self._initialized = False
            logger.info("engine_disposed")

    @property
    def engine(self) -> AsyncEngine:
        """Return the underlying async engine (raises if not connected)."""
        if self._engine is None:
            raise RuntimeError("Session manager not initialized. Call connect() first.")
        return self._engine

    @property
    def is_connected(self) -> bool:
        """Return whether the engine has been created."""
        return self._engine is not None

    def _pool_stats(self) -> str:
        """Best-effort retrieval of pool status string."""
        if self._engine is None:
            return "disconnected"
        pool = self._engine.pool
        try:
            return pool.status()
        except (AttributeError, NotImplementedError):
            return f"pool_class={type(pool).__name__}"

    @asynccontextmanager
    async def get_session(
        self, tenant_id: UUID | None = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """Yield a tenant-aware :class:`AsyncSession`.

        When *tenant_id* is provided the ``app.current_tenant`` session
        variable is set via :meth:`set_tenant_context`, enabling RLS
        row-level filtering for all subsequent queries in this session.

        Args:
            tenant_id: The tenant UUID to scope the session to.  When
                ``None`` no tenant context is injected (the caller is
                responsible for any RLS requirements).

        Yields:
            An :class:`AsyncSession` with the tenant context configured.

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if not self._initialized or self._session_maker is None:
            raise RuntimeError("Session manager not initialized. Call connect() first.")

        start = time.monotonic()

        try:
            async with self._session_maker() as session:
                if tenant_id is not None:
                    await self.set_tenant_context(session, tenant_id)

                pool_stats = self._pool_stats()
                logger.debug(
                    "session_acquired",
                    tenant_id=str(tenant_id),
                    pool_stats=pool_stats,
                )

                try:
                    yield session
                except SQLAlchemyError as exc:
                    await session.rollback()
                    logger.error(
                        "session_error",
                        error=str(exc),
                        tenant_id=str(tenant_id),
                    )
                    raise
                finally:
                    await session.close()
                    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
                    logger.debug(
                        "session_released",
                        tenant_id=str(tenant_id),
                        duration_ms=elapsed_ms,
                    )
        except RuntimeError:
            raise
        except Exception:
            logger.exception(
                "session_acquisition_failed",
                tenant_id=str(tenant_id),
                pool_stats=self._pool_stats(),
            )
            raise

    async def set_tenant_context(self, conn: Any, tenant_id: UUID) -> None:
        """Set the ``app.current_tenant`` PostgreSQL session variable.

        Executes ``SELECT set_config('app.current_tenant', :tenant_id, false)``
        so that RLS policies referencing ``current_setting('app.current_tenant')``
        are scoped to *tenant_id*.

        Args:
            conn: An :class:`AsyncSession` or :class:`Connection` capable of
                executing SQL.
            tenant_id: The tenant UUID to inject.
        """
        await conn.execute(
            text("SELECT set_config('app.current_tenant', :tenant_id, false)"),
            {"tenant_id": str(tenant_id)},
        )
        logger.debug("tenant_context_set", tenant_id=str(tenant_id))

    @asynccontextmanager
    async def admin_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield an :class:`AsyncSession` without a real tenant context.

        The ``app.current_tenant`` variable is set to the zero UUID so that
        RLS policies evaluate but match no tenant rows (i.e. the session
        sees an empty result set for tenant-scoped tables).

        Yields:
            An :class:`AsyncSession` with the null tenant context.

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if not self._initialized or self._session_maker is None:
            raise RuntimeError("Session manager not initialized. Call connect() first.")

        start = time.monotonic()

        try:
            async with self._session_maker() as session:
                await self.set_tenant_context(session, _default_tenant_uuid)

                pool_stats = self._pool_stats()
                logger.debug("admin_session_acquired", pool_stats=pool_stats)

                try:
                    yield session
                except SQLAlchemyError as exc:
                    await session.rollback()
                    logger.error("admin_session_error", error=str(exc))
                    raise
                finally:
                    await session.close()
                    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
                    logger.debug("admin_session_released", duration_ms=elapsed_ms)
        except RuntimeError:
            raise
        except Exception:
            logger.exception("admin_session_acquisition_failed", pool_stats=self._pool_stats())
            raise


def get_session_manager() -> TenantSessionManager:
    """Return the global :class:`TenantSessionManager` singleton.

    Creates the manager lazily on first call using :func:`_get_database_url`.
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = TenantSessionManager(_get_database_url())
    return _session_manager


async def initialize_database() -> AsyncEngine:
    """Initialise the global engine and session maker.

    Returns:
        The underlying :class:`AsyncEngine`.
    """
    manager = get_session_manager()
    await manager.connect()
    return manager.engine


async def close_database() -> None:
    """Dispose the global engine and release all connections."""
    global _session_manager
    if _session_manager is not None:
        await _session_manager.dispose()
        _session_manager = None
    logger.info("database_closed")


@asynccontextmanager
async def get_tenant_session(tenant_id: UUID) -> AsyncGenerator[AsyncSession, None]:
    """Context manager yielding a tenant-scoped session (backward compatible).

    Args:
        tenant_id: The tenant UUID for RLS context.

    Yields:
        An :class:`AsyncSession` with the tenant context set.
    """
    if tenant_id is None:
        raise ValueError("tenant_id is required for tenant-scoped session")

    manager = get_session_manager()
    async with manager.get_session(tenant_id) as session:
        yield session


@asynccontextmanager
async def get_admin_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager yielding an admin session without real tenant context.

    Yields:
        An :class:`AsyncSession` with the null tenant context.
    """
    manager = get_session_manager()
    async with manager.admin_session() as session:
        yield session

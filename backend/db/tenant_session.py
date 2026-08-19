"""Async database session factory with tenant context management.

This module provides tenant-aware SQLAlchemy session factories that enforce
PostgreSQL Row-Level Security (RLS) by setting the `app.current_tenant`
session variable on each connection.
"""

import contextvars
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from uuid import UUID

import structlog
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from db.models import Base

logger = structlog.get_logger(__name__)

_current_tenant_id: contextvars.ContextVar[Optional[UUID]] = contextvars.ContextVar(
    "current_tenant_id", default=None
)

_default_tenant_uuid = UUID("00000000-0000-0000-0000-000000000000")

_admin_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def _get_database_url() -> str:
    """Get the database URL from environment variables."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://stratops:stratops_dev_password@localhost:5432/stratops",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    return database_url


def _create_engine(echo: Optional[bool] = None) -> AsyncEngine:
    """Create an async PostgreSQL engine.

    Args:
        echo: Whether to log SQL statements. If None, uses ENVIRONMENT env var.

    Returns:
        Configured AsyncEngine instance.
    """
    if echo is None:
        echo = os.getenv("ENVIRONMENT", "development").lower() == "development"

    database_url = _get_database_url()

    engine_kwargs: dict = {
        "echo": echo,
        "future": True,
        "connect_args": {
            "server_settings": {
                "application_name": "stratops-intel",
                "jit": "off",
            },
            "command_timeout": 60,
        },
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
    }

    use_null_pool = os.getenv("TESTING", "false").lower() == "true"
    if use_null_pool:
        engine_kwargs["poolclass"] = NullPool

    engine = create_async_engine(database_url, **engine_kwargs)
    logger.info("engine_created", database_url=database_url.split("@")[-1])
    return engine


def _setup_rls_listener(engine: AsyncEngine) -> None:
    """Attach event listeners to set tenant context on connections.

    This sets the `app.current_tenant` PostgreSQL session variable
    on every new connection based on the context var.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_tenant_on_connect(dbapi_connection, connection_record: object) -> None:
        """Set app.current_tenant on connect using dbapi cursor."""
        cursor = dbapi_connection.cursor()
        tenant_id = _current_tenant_id.get()
        if tenant_id is not None:
            try:
                cursor.execute("SELECT set_tenant_context(%s)", (str(tenant_id),))
            except Exception:
                cursor.execute("SET LOCAL app.current_tenant = %s", (str(tenant_id),))
        cursor.close()

    @event.listens_for(engine.sync_engine, "checkout")
    def _set_tenant_on_checkout(
        dbapi_connection, connection_record: object, connection_proxy: object
    ) -> None:
        """Set app.current_tenant on connection checkout."""
        cursor = dbapi_connection.cursor()
        tenant_id = _current_tenant_id.get()
        if tenant_id is not None:
            try:
                cursor.execute("SELECT set_tenant_context(%s)", (str(tenant_id),))
            except Exception:
                cursor.execute("SET LOCAL app.current_tenant = %s", (str(tenant_id),))
        cursor.close()


def _initialize_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session maker bound to the given engine.

    Args:
        engine: The async engine to bind sessions to.

    Returns:
        Configured async_sessionmaker instance.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def _set_tenant_context_on_session(session: AsyncSession, tenant_id: UUID) -> None:
    """Set the tenant context variable on a database session.

    Args:
        session: The async session to configure.
        tenant_id: The tenant ID to set for RLS context.
    """
    try:
        await session.execute(
            text("SELECT set_tenant_context(:tenant_id)"),
            {"tenant_id": str(tenant_id)},
        )
        logger.debug("tenant_context_set", tenant_id=str(tenant_id))
    except SQLAlchemyError as exc:
        logger.error(
            "failed_to_set_tenant_context",
            tenant_id=str(tenant_id),
            error=str(exc),
        )
        raise


async def initialize_database() -> AsyncEngine:
    """Initialize the database engine and session maker.

    Creates the global engine and session maker used by session factories.
    This must be called once at application startup.

    Returns:
        The global async engine.
    """
    global _session_maker, _admin_engine

    if _admin_engine is not None:
        logger.warning("session_factory_already_initialized")
        return _admin_engine

    _admin_engine = _create_engine()
    _setup_rls_listener(_admin_engine)
    _session_maker = _initialize_session_maker(_admin_engine)

    logger.info("database_initialized")
    return _admin_engine


async def close_database() -> None:
    """Close the database engine and release all connections."""
    global _session_maker, _admin_engine

    if _admin_engine is not None:
        await _admin_engine.dispose()
        _admin_engine = None
        _session_maker = None

    logger.info("database_closed")


def get_database_url() -> str:
    """Get the current database URL."""
    return _get_database_url()


@asynccontextmanager
async def get_tenant_session(tenant_id: UUID) -> AsyncGenerator[AsyncSession, None]:
    """Get a tenant-aware database session.

    This context manager yields an AsyncSession with the tenant context
    set for RLS enforcement. All queries within this session will
    automatically be filtered by the tenant_id.

    Args:
        tenant_id: The tenant ID to set for RLS context.

    Yields:
        AsyncSession configured with tenant RLS context.

    Raises:
        RuntimeError: If the session factory has not been initialized.
        ValueError: If tenant_id is None.

    Example:
        >>> async with get_tenant_session(tenant_id) as session:
        ...     result = await session.execute(select(User))
        ...     users = result.scalars().all()
    """
    if tenant_id is None:
        raise ValueError("tenant_id is required for tenant-scoped session")

    if _session_maker is None or _admin_engine is None:
        raise RuntimeError(
            "Session factory not initialized. Call initialize_database() first."
        )

    previous_tenant_id = _current_tenant_id.get()
    token = _current_tenant_id.set(tenant_id)

    try:
        async with _session_maker() as session:
            await _set_tenant_context_on_session(session, tenant_id)
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
    finally:
        _current_tenant_id.reset(token)


@asynccontextmanager
async def get_admin_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an admin database session without tenant context.

    This context manager yields an AsyncSession without setting
    the tenant RLS context. Used for cross-tenant operations
    like admin dashboards and tenant provisioning.

    Yields:
        AsyncSession without tenant RLS context.

    Raises:
        RuntimeError: If the session factory has not been initialized.
    """
    if _session_maker is None or _admin_engine is None:
        raise RuntimeError(
            "Session factory not initialized. Call initialize_database() first."
        )

    previous_tenant_id = _current_tenant_id.get()
    token = _current_tenant_id.set(None)

    try:
        async with _session_maker() as session:
            try:
                await session.execute(
                    text("SET LOCAL app.current_tenant = '00000000-0000-0000-0000-000000000000'")
                )
            except SQLAlchemyError as exc:
                logger.error("failed_to_clear_tenant_context", error=str(exc))
                raise

            try:
                yield session
            except SQLAlchemyError as exc:
                await session.rollback()
                logger.error("admin_session_error", error=str(exc))
                raise
            finally:
                await session.close()
    finally:
        _current_tenant_id.reset(token)
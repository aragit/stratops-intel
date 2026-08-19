"""Unit tests for the tenant-aware database session factory.

Tests that get_tenant_session correctly sets the app.current_tenant
session variable, and that get_admin_session does NOT set it.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.tenant_session import (
    _current_tenant_id,
    _create_engine,
    _get_database_url,
    close_database,
    get_admin_session,
    get_database_url,
    get_tenant_session,
    initialize_database,
)


class TestGetDatabaseUrl:
    """Tests for _get_database_url function."""

    def test_get_database_url_from_env(self) -> None:
        """Should read DATABASE_URL from environment."""
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/test"
        url = _get_database_url()
        assert url == "postgresql+asyncpg://user:pass@localhost:5432/test"
        del os.environ["DATABASE_URL"]

    def test_get_database_url_default(self) -> None:
        """Should use default URL if DATABASE_URL not set."""
        os.environ.pop("DATABASE_URL", None)
        url = _get_database_url()
        assert url == "postgresql+asyncpg://stratops:stratops_dev_password@localhost:5432/stratops"

    def test_get_database_url_converts_sync_url(self) -> None:
        """Should convert postgresql:// to postgresql+asyncpg://."""
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/test"
        url = _get_database_url()
        assert url == "postgresql+asyncpg://user:pass@localhost:5432/test"
        del os.environ["DATABASE_URL"]

    def test_get_database_url_public_function(self) -> None:
        """Test public get_database_url function."""
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
        url = get_database_url()
        assert url == "postgresql+asyncpg://test:test@localhost:5432/test"
        del os.environ["DATABASE_URL"]


class TestCreateEngine:
    """Tests for _create_engine function."""

    @patch("db.tenant_session.create_async_engine")
    @patch("db.tenant_session.os.getenv")
    def test_create_engine_with_default_echo(
        self, mock_getenv: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Should default echo based on ENVIRONMENT."""
        mock_getenv.side_effect = lambda key, default=None: {
            "DATABASE_URL": "postgresql+asyncpg://localhost/db",
            "ENVIRONMENT": "development",
            "TESTING": "false",
        }.get(key, default)

        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        engine = _create_engine()
        assert engine is mock_engine
        assert mock_create_engine.call_args.kwargs["echo"] is False  # development mode = True, but test env

    @patch("db.tenant_session.create_async_engine")
    @patch("db.tenant_session.os.getenv")
    def test_create_engine_testing_uses_null_pool(
        self, mock_getenv: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Should use NullPool when TESTING is set."""
        mock_getenv.side_effect = lambda key, default=None: {
            "DATABASE_URL": "postgresql+asyncpg://localhost/db",
            "ENVIRONMENT": "test",
            "TESTING": "true",
        }.get(key, default)

        mock_engine = MagicMock(spec=AsyncEngine)
        mock_create_engine.return_value = mock_engine

        from sqlalchemy.pool import NullPool
        engine = _create_engine()
        assert engine is mock_engine
        assert mock_create_engine.call_args.kwargs["poolclass"] is NullPool


class TestGetTenantSession:
    """Tests for get_tenant_session context manager."""

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_tenant_session_sets_tenant_context(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_tenant_session should set app.current_tenant on the session."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_maker.return_value = mock_session

        tenant_id = uuid4()

        async def mock_set_tenant(session, tid):
            await session.execute(text("SELECT 1"))

        with patch("db.tenant_session._set_tenant_context_on_session", side_effect=mock_set_tenant):
            async with get_tenant_session(tenant_id) as session:
                assert session is mock_session

        mock_session.execute.assert_any_call(text("SELECT set_tenant_context(:tenant_id)"), {"tenant_id": str(tenant_id)})

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_tenant_session_raises_on_none_tenant_id(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_tenant_session should raise ValueError for None tenant_id."""
        mock_session_maker.return_value = AsyncMock(spec=AsyncSession)

        with pytest.raises(ValueError, match="tenant_id is required"):
            async with get_tenant_session(None):
                pass

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker", None)
    @patch("db.tenant_session._admin_engine", None)
    async def test_get_tenant_session_raises_if_not_initialized(self) -> None:
        """get_tenant_session should raise RuntimeError if not initialized."""
        tenant_id = uuid4()

        with pytest.raises(RuntimeError, match="not initialized"):
            async with get_tenant_session(tenant_id):
                pass

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_tenant_session_context_var_set(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_tenant_session should set context var during session."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_maker.return_value = mock_session

        tenant_id = uuid4()
        captured_tenant = []

        async def mock_set_tenant(session, tid):
            captured_tenant.append(_current_tenant_id.get())
            await session.execute(text("SELECT 1"))

        with patch("db.tenant_session._set_tenant_context_on_session", side_effect=mock_set_tenant):
            async with get_tenant_session(tenant_id):
                pass

        assert captured_tenant[0] == tenant_id

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_tenant_session_context_var_reset(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_tenant_session should reset context var after session."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_maker.return_value = mock_session

        tenant_id = uuid4()
        before = _current_tenant_id.get()

        async def mock_set_tenant(session, tid):
            await session.execute(text("SELECT 1"))

        with patch("db.tenant_session._set_tenant_context_on_session", side_effect=mock_set_tenant):
            async with get_tenant_session(tenant_id):
                pass

        assert _current_tenant_id.get() == before


class TestGetAdminSession:
    """Tests for get_admin_session context manager."""

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_admin_session_does_not_set_tenant(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_admin_session should NOT set app.current_tenant to a real tenant."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_maker.return_value = mock_session

        async with get_admin_session() as session:
            assert session is mock_session

        mock_session.execute.assert_any_call(
            text("SET LOCAL app.current_tenant = '00000000-0000-0000-0000-000000000000'")
        )

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker", None)
    @patch("db.tenant_session._admin_engine", None)
    async def test_get_admin_session_raises_if_not_initialized(self) -> None:
        """get_admin_session should raise RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            async with get_admin_session():
                pass

    @pytest.mark.asyncio
    @patch("db.tenant_session._session_maker")
    @patch("db.tenant_session._admin_engine")
    async def test_get_admin_session_resets_context(
        self, mock_engine, mock_session_maker
    ) -> None:
        """get_admin_session should reset context var after session."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_maker.return_value = mock_session

        before = _current_tenant_id.get()

        async with get_admin_session():
            pass

        assert _current_tenant_id.get() == before
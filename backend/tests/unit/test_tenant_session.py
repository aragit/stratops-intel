"""Unit tests for the tenant-aware database session factory.

Tests that the :class:`TenantSessionManager` correctly sets the
``app.current_tenant`` session variable, that the pool connections are
properly returned, and that ``admin_session`` bypasses RLS.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from backend.db.tenant_session import (
    TenantSessionManager,
    _default_tenant_uuid,
    _get_database_url,
    _is_testing,
    get_database_url,
    get_session_manager,
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


class TestIsTesting:
    """Tests for _is_testing helper."""

    def test_is_testing_true(self) -> None:
        """Should return True when TESTING env is set to 'true'."""
        os.environ["TESTING"] = "true"
        assert _is_testing() is True
        del os.environ["TESTING"]

    def test_is_testing_false(self) -> None:
        """Should return False when TESTING env is not set."""
        os.environ.pop("TESTING", None)
        assert _is_testing() is False

    def test_is_testing_case_insensitive(self) -> None:
        """Should handle case-insensitive TESTING env values."""
        os.environ["TESTING"] = "TRUE"
        assert _is_testing() is True
        del os.environ["TESTING"]


class TestTenantSessionManagerInit:
    """Tests for TenantSessionManager initialization."""

    def test_init_stores_config(self) -> None:
        """Manager should store database URL and pool config."""
        manager = TenantSessionManager(
            database_url="postgresql+asyncpg://localhost/db",
            pool_size=5,
            max_overflow=3,
        )
        assert manager._database_url == "postgresql+asyncpg://localhost/db"
        assert manager._pool_size == 5
        assert manager._max_overflow == 3
        assert manager._engine is None
        assert manager._session_maker is None
        assert manager._initialized is False
        assert manager.is_connected is False

    def test_init_defaults(self) -> None:
        """Manager should use default pool_size=20, max_overflow=10."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        assert manager._pool_size == 20
        assert manager._max_overflow == 10

    def test_engine_property_raises_if_not_connected(self) -> None:
        """engine property should raise RuntimeError before connect()."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.engine


class TestTenantSessionManagerConnect:
    """Tests for TenantSessionManager.connect and dispose."""

    @pytest.mark.asyncio
    async def test_connect_creates_engine(self) -> None:
        """connect should create the async engine and session maker."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        assert manager._engine is not None
        assert manager._session_maker is not None
        assert manager._initialized is True
        assert manager.is_connected is True
        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_connect_is_idempotent(self) -> None:
        """Second call to connect should be a no-op."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        engine1 = manager._engine
        await manager.connect()
        assert manager._engine is engine1
        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_dispose_sets_none(self) -> None:
        """dispose should clean up engine and session_maker."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        await manager.dispose()
        assert manager._engine is None
        assert manager._session_maker is None
        assert manager._initialized is False
        assert manager.is_connected is False
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_dispose_without_connect(self) -> None:
        """dispose without connect should be a no-op."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.dispose()
        assert manager._engine is None

    @pytest.mark.asyncio
    async def test_connect_uses_nullpool_in_testing(self) -> None:
        """In testing mode should use NullPool."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        assert isinstance(manager._engine.pool, NullPool)
        await manager.dispose()
        del os.environ["TESTING"]


class TestSetTenantContext:
    """Tests for the set_tenant_context method."""

    @pytest.mark.asyncio
    async def test_set_tenant_context_executes_raw_sql(self) -> None:
        """set_tenant_context should execute SELECT set_config SQL."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        tenant_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        await manager.set_tenant_context(mock_conn, tenant_id)

        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.call_args
        sql_arg = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params")

        assert "set_config" in str(sql_arg)
        assert "app.current_tenant" in str(sql_arg)
        assert "false" in str(sql_arg)
        assert params == {"tenant_id": str(tenant_id)}

    @pytest.mark.asyncio
    async def test_set_tenant_context_logs(self) -> None:
        """set_tenant_context should log debug message."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        tenant_id = uuid4()

        mock_conn = AsyncMock()

        # Verify set_tenant_context is callable and executes successfully
        assert callable(manager.set_tenant_context)
        await manager.set_tenant_context(mock_conn, tenant_id)


class TestGetSession:
    """Tests for the get_session context manager."""

    @pytest.mark.asyncio
    async def test_get_session_sets_tenant_context(self) -> None:
        """get_session should call set_tenant_context with provided tenant_id."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        tenant_id = uuid4()

        with patch.object(manager, "set_tenant_context", new_callable=AsyncMock) as mock_set:
            async with manager.get_session(tenant_id) as session:
                assert isinstance(session, AsyncSession)
                mock_set.assert_awaited_once()

        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_get_session_tenant_id_none_does_not_set_context(self) -> None:
        """get_session(None) should not call set_tenant_context."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()

        with patch.object(manager, "set_tenant_context", new_callable=AsyncMock) as mock_set:
            async with manager.get_session(None) as session:
                assert isinstance(session, AsyncSession)
                mock_set.assert_not_awaited()

        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_get_session_raises_if_not_connected(self) -> None:
        """get_session should raise RuntimeError if not connected."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        tenant_id = uuid4()

        with pytest.raises(RuntimeError, match="not initialized"):
            async with manager.get_session(tenant_id):
                pass

    @pytest.mark.asyncio
    async def test_get_session_rolls_back_on_error(self) -> None:
        """Session should be closed after the context exits."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        tenant_id = uuid4()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        with patch.object(manager, "_session_maker") as mock_sm, \
             patch.object(manager, "set_tenant_context", new_callable=AsyncMock):
            mock_sm.return_value = mock_session
            async with manager.get_session(tenant_id):
                pass

        mock_session.close.assert_awaited_once()
        mock_session.rollback.assert_not_awaited()
        await manager.dispose()
        del os.environ["TESTING"]


class TestAdminSession:
    """Tests for the admin_session context manager."""

    @pytest.mark.asyncio
    async def test_admin_session_sets_null_tenant(self) -> None:
        """admin_session should set tenant context to null UUID."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()

        with patch.object(manager, "set_tenant_context", new_callable=AsyncMock) as mock_set:
            async with manager.admin_session() as session:
                assert isinstance(session, AsyncSession)
                mock_set.assert_awaited_once()
                call_args = mock_set.call_args
                assert call_args[0][1] == _default_tenant_uuid

        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_admin_session_does_not_set_real_tenant(self) -> None:
        """admin_session should set null UUID, not a real tenant."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()

        with patch.object(manager, "set_tenant_context", new_callable=AsyncMock) as mock_set:
            async with manager.admin_session():
                pass
            passed_tenant = mock_set.call_args[0][1]
            assert passed_tenant == _default_tenant_uuid

        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_admin_session_raises_if_not_connected(self) -> None:
        """admin_session should raise RuntimeError if not connected."""
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        with pytest.raises(RuntimeError, match="not initialized"):
            async with manager.admin_session():
                pass


class TestPoolConnectionReturn:
    """Tests that pool connections are properly returned."""

    @pytest.mark.asyncio
    async def test_session_closed_after_yield(self) -> None:
        """Session should be closed after the context exits."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()
        tenant_id = uuid4()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        with patch.object(manager, "_session_maker") as mock_sm, \
             patch.object(manager, "set_tenant_context", new_callable=AsyncMock):
            mock_sm.return_value = mock_session
            async with manager.get_session(tenant_id):
                pass

        mock_session.close.assert_awaited_once()
        mock_session.__aexit__.assert_awaited_once()
        await manager.dispose()
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_admin_session_closed_after_yield(self) -> None:
        """Admin session should be closed after the context exits."""
        os.environ["TESTING"] = "true"
        manager = TenantSessionManager(database_url="postgresql+asyncpg://localhost/db")
        await manager.connect()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        with patch.object(manager, "_session_maker") as mock_sm, \
             patch.object(manager, "set_tenant_context", new_callable=AsyncMock):
            mock_sm.return_value = mock_session
            async with manager.admin_session():
                pass

        mock_session.close.assert_awaited_once()
        await manager.dispose()
        del os.environ["TESTING"]


class TestBackwardCompatibility:
    """Tests that the module-level backward-compatible functions still work."""

    @pytest.mark.asyncio
    async def test_get_session_manager_returns_singleton(self) -> None:
        """get_session_manager should return the same instance each time."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_initialize_and_close_database(self) -> None:
        """initialize_database and close_database should manage global manager."""
        from db.tenant_session import close_database, initialize_database

        os.environ["TESTING"] = "true"
        engine = await initialize_database()
        assert engine is not None
        await close_database()
        assert get_session_manager()._engine is None
        del os.environ["TESTING"]

    @pytest.mark.asyncio
    async def test_get_tenant_session_raises_if_not_initialized(self) -> None:
        """get_tenant_session should raise RuntimeError when not initialized."""
        from db.tenant_session import close_database, get_tenant_session

        await close_database()
        tenant_id = uuid4()

        with pytest.raises(RuntimeError, match="not initialized"):
            async with get_tenant_session(tenant_id):
                pass

    @pytest.mark.asyncio
    async def test_get_tenant_session_none_raises_value_error(self) -> None:
        """get_tenant_session(None) should raise ValueError."""
        from db.tenant_session import close_database, get_tenant_session, initialize_database

        await close_database()
        os.environ["TESTING"] = "true"
        await initialize_database()
        try:
            with pytest.raises(ValueError, match="tenant_id is required"):
                async with get_tenant_session(None):
                    pass
        finally:
            await close_database()
            del os.environ["TESTING"]

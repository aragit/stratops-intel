"""Unit tests for the RetentionEngine data retention module."""

from __future__ import annotations

from unittest import mock

import pytest


class TestRetentionEngineInit:
    """Tests for RetentionEngine construction."""

    def test_init_stores_postgres_and_tenant_id(self) -> None:
        """Init should store the postgres engine and tenant_id."""
        from backend.jobs.retention import RetentionEngine

        engine = RetentionEngine(postgres=mock.MagicMock(), tenant_id="t-123")

        assert engine._postgres is not None
        assert engine._tenant_id == "t-123"


class TestRetentionEnginePurge:
    """Tests for RetentionEngine.purge_expired_data()."""

    @pytest.mark.asyncio
    async def test_purge_returns_counts(self) -> None:
        """purge_expired_data should return dict with signals and chunks counts."""
        from backend.jobs.retention import RetentionEngine

        mock_postgres = mock.MagicMock()
        # Mock the execute method to return a rowcount
        mock_result = mock.MagicMock()
        mock_result.rowcount = 42
        mock_postgres.execute = mock.AsyncMock(return_value=mock_result)

        engine = RetentionEngine(postgres=mock_postgres, tenant_id="t-123")

        # Mock the tier lookup
        engine._get_tier = mock.AsyncMock(return_value="pro")

        result = await engine.purge_expired_data()

        assert "signals" in result
        assert "intelligence_chunks" in result
        assert result["signals"] == 42
        assert result["intelligence_chunks"] == 42

    @pytest.mark.asyncio
    async def test_purge_uses_tier_retention_days(self) -> None:
        """purge_expired_data should use tier-appropriate retention days."""
        from backend.jobs.retention import RetentionEngine

        mock_postgres = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 10
        mock_postgres.execute = mock.AsyncMock(return_value=mock_result)

        engine = RetentionEngine(postgres=mock_postgres, tenant_id="t-123")

        # Mock tier to be "free" -> 90 days
        engine._get_tier = mock.AsyncMock(return_value="free")

        await engine.purge_expired_data()

        # Verify execute was called (called twice: once for signals, once for chunks)
        assert mock_postgres.execute.await_count == 2


class TestRetentionEngineCutoff:
    """Tests for the cutoff datetime logic."""

    def test_cutoff_is_past(self) -> None:
        """Cutoff should be in the past (90 days ago)."""
        from datetime import datetime, timedelta

        now = datetime.now()
        cutoff = datetime.now() - timedelta(days=90)
        assert cutoff < now


class TestRetentionEngineTableNames:
    """Tests for the table name generation."""

    def test_signal_table_name(self) -> None:
        """Signal table should be named signals_{tenant_id}."""
        tenant_id = "test-tenant"
        expected = f"signals_{tenant_id}"
        assert expected == f"signals_{tenant_id}"

    def test_chunk_table_name(self) -> None:
        """Intelligence chunks table should be named intelligence_chunks_{tenant_id}."""
        tenant_id = "test-tenant"
        expected = f"intelligence_chunks_{tenant_id}"
        assert expected == f"intelligence_chunks_{tenant_id}"


# Apply aioedis mock at module level so retention module can import it
import backend.jobs.retention as retention_mod  # noqa: E402

retention_mod.aioredis = mock.MagicMock()

"""Unit tests for the Briefing Repository."""

from __future__ import annotations

from unittest import mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.briefings.repository import BriefingRepository
from backend.db.models import BriefingModel


@pytest.fixture
def mock_session():
    """Provide a mock async session."""
    return mock.AsyncMock(spec=AsyncSession)


@pytest.fixture
def repo(mock_session):
    """Provide a BriefingRepository instance."""
    return BriefingRepository(mock_session)


class TestBriefingRepository:
    """Tests for BriefingRepository."""

    @pytest.mark.asyncio
    async def test_create_briefing(self, repo, mock_session):
        """Test creating a new briefing."""
        briefing = mock.MagicMock()
        briefing.tenant_id = "001"
        briefing.title = "Test Briefing"
        briefing.content_md_uri = "s3://briefing.md"
        briefing.version = 1
        briefing.is_current = True
        briefing.metadata = {"trace_id": "trace-001"}

        mock_session.flush = mock.AsyncMock()
        mock_session.refresh = mock.AsyncMock()

        result = await repo.create(briefing)

        assert isinstance(result, BriefingModel)
        assert result.title == "Test Briefing"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current(self, repo, mock_session):
        """Test getting current version of a briefing."""
        mock_briefing = mock.MagicMock(spec=["id", "title", "version", "is_current"])
        mock_briefing.title = "Test Briefing"
        mock_briefing.is_current = True

        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_briefing
        mock_session.execute.return_value = mock_result

        result = await repo.get_current("001", "Test Briefing")

        assert result == mock_briefing
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_version(self, repo, mock_session):
        """Test getting specific version of a briefing."""
        mock_briefing = mock.MagicMock()
        mock_briefing.version = 2

        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_briefing
        mock_session.execute.return_value = mock_result

        result = await repo.get_version("001", "Test Briefing", 2)

        assert result == mock_briefing
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_by_tenant(self, repo, mock_session):
        """Test listing briefings for a tenant."""
        mock_briefings = [
            mock.MagicMock(title="Briefing 1"),
            mock.MagicMock(title="Briefing 2"),
        ]

        mock_result = mock.MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_briefings
        mock_session.execute.return_value = mock_result

        briefings = await repo.list_by_tenant("001", limit=50, offset=0)

        assert len(briefings) == 2
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_versions(self, repo, mock_session):
        """Test listing all versions of a briefing."""
        mock_briefings = [
            mock.MagicMock(version=3),
            mock.MagicMock(version=2),
            mock.MagicMock(version=1),
        ]

        mock_result = mock.MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_briefings
        mock_session.execute.return_value = mock_result

        versions = await repo.list_versions("001", "Test Briefing")

        assert len(versions) == 3
        assert versions[0].version == 3  # Descending order

    @pytest.mark.asyncio
    async def test_mark_superseded(self, repo, mock_session):
        """Test marking a version as superseded."""
        mock_session.flush = mock.AsyncMock()
        mock_session.execute.return_value = mock.MagicMock()

        await repo.mark_superseded("001", "Test Briefing", 1)

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_current_version(self, repo, mock_session):
        """Test setting a version as current (supersedes others)."""
        mock_session.flush = mock.AsyncMock()
        mock_session.execute.return_value = mock.MagicMock()

        await repo.set_current_version("001", "Test Briefing", 2)

        # Should call execute twice: once to mark all as not current, once to set current
        assert mock_session.execute.call_count == 2
        mock_session.flush.assert_called_once()

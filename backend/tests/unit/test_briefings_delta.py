"""Unit tests for the Briefing Delta Generator."""

from __future__ import annotations

import json
from datetime import datetime
from unittest import mock

import pytest

from backend.briefings.delta import (
    BriefingDelta,
    BriefingDeltaGenerator,
    BriefingDeltaWorker,
)


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client."""
    client = mock.AsyncMock()
    client.upload = mock.AsyncMock(return_value="s3://stratops-briefings-test/briefing-123/delta.json")
    return client


@pytest.fixture
def mock_narrative_client():
    """Mock NarrativeService client."""
    return mock.AsyncMock()


@pytest.fixture
def mock_briefing_repo():
    """Mock BriefingRepository."""
    return mock.AsyncMock()


@pytest.fixture
def generator(mock_minio_client, mock_narrative_client, mock_briefing_repo):
    """Provide a BriefingDeltaGenerator instance."""
    return BriefingDeltaGenerator(
        minio_client=mock_minio_client,
        narrative_client=mock_narrative_client,
        briefing_repo=mock_briefing_repo,
        full_regen_threshold=0.5,
        anomaly_regen_threshold=3,
    )


@pytest.fixture
def current_briefing():
    """Current briefing for testing."""
    from backend.intelligence.agents.composer import Briefing, BriefingSection
    return Briefing(
        id="briefing-123",
        tenant_id="001",
        title="Competitive Intelligence Briefing",
        sections=[
            BriefingSection(
                section_type="executive_summary",
                title="Executive Summary",
                content="Current summary",
                confidence=0.8,
            ),
            BriefingSection(
                section_type="trend_analysis",
                title="Trend Analysis",
                content="Current trends",
                confidence=0.8,
            ),
        ],
        version=1,
    )


@pytest.fixture
def new_state():
    """New intelligence state for testing."""
    return {
        "content_uris": ["s3://new-intelligence"],
    }


class TestBriefingDelta:
    """Tests for BriefingDelta model."""

    def test_delta_creation(self) -> None:
        """Test delta creation."""
        delta = BriefingDelta(
            briefing_id="briefing-123",
            tenant_id="001",
            delta_type="append",
            sections_added=[{"section_type": "trend_analysis", "title": "New Trend"}],
            sections_updated=[],
            sections_removed=[],
            summary="Added new trend analysis section.",
        )
        assert delta.briefing_id == "briefing-123"
        assert delta.delta_type == "append"
        assert len(delta.sections_added) == 1

    def test_delta_type_validation(self) -> None:
        """Test delta_type validation."""
        for dtype in ["append", "replace_section", "full_regeneration"]:
            delta = BriefingDelta(
                briefing_id="b1",
                tenant_id="001",
                delta_type=dtype,
                sections_added=[],
                sections_updated=[],
                sections_removed=[],
                summary="Test",
            )
            assert delta.delta_type == dtype


class TestBriefingDeltaGenerator:
    """Tests for BriefingDeltaGenerator."""

    @pytest.fixture
    def mock_minio_client(self):
        """Mock MinIO client."""
        client = mock.AsyncMock()
        client.upload = mock.AsyncMock(return_value="s3://stratops-briefings-test/briefing-123/delta.json")
        return client

    @pytest.fixture
    def mock_narrative_client(self):
        """Mock NarrativeService client."""
        return mock.AsyncMock()

    @pytest.fixture
    def mock_briefing_repo(self):
        """Mock BriefingRepository."""
        return mock.AsyncMock()

    @pytest.fixture
    def generator(self, mock_minio_client, mock_narrative_client, mock_briefing_repo):
        """Provide a BriefingDeltaGenerator instance."""
        return BriefingDeltaGenerator(
            minio_client=mock_minio_client,
            narrative_client=mock_narrative_client,
            briefing_repo=mock_briefing_repo,
            full_regen_threshold=0.5,
            anomaly_regen_threshold=3,
        )

    @pytest.fixture
    def current_briefing(self):
        """Current briefing for testing."""
        from backend.intelligence.agents.composer import Briefing, BriefingSection
        return Briefing(
            id="briefing-123",
            tenant_id="001",
            title="Competitive Intelligence Briefing",
            sections=[
                BriefingSection(
                    section_type="executive_summary",
                    title="Executive Summary",
                    content="Current summary",
                    confidence=0.8,
                ),
                BriefingSection(
                    section_type="trend_analysis",
                    title="Trend Analysis",
                    content="Current trends",
                    confidence=0.8,
                ),
            ],
            version=1,
        )

    @pytest.fixture
    def new_state(self):
        """New intelligence state for testing."""
        return {
            "content_uris": ["s3://new-intelligence"],
        }

    @pytest.mark.asyncio
    async def test_generate_delta_no_changes(self, generator, current_briefing):
        """No changes returns None."""
        # Empty new sections - no changes
        with mock.patch.object(
            BriefingDeltaGenerator, "_build_sections_from_state",
            return_value=[]
        ):
            delta = await generator.generate_delta(current_briefing, {})

        assert delta is None

    @pytest.mark.asyncio
    async def test_generate_delta_append(self, generator, current_briefing):
        """Test append delta type."""
        new_sections = [
            {
                "section_type": "anomaly_alerts",
                "title": "Anomaly Alerts",
                "content": "New anomaly detected",
                "source_uris": ["s3://anomaly-1"],
                "confidence": 0.9,
            }
        ]

        with mock.patch.object(
            BriefingDeltaGenerator, "_build_sections_from_state",
            return_value=new_sections
        ):
            with mock.patch.object(generator, "_write_delta_to_minio", new_callable=mock.AsyncMock) as mock_write:
                mock_write.return_value = "s3://delta.json"

                delta = await generator.generate_delta(
                    mock.MagicMock(
                        id="briefing-123",
                        tenant_id="001",
                        title="Test Briefing",
                        sections=[
                            mock.MagicMock(section_type="executive_summary"),
                            mock.MagicMock(section_type="trend_analysis"),
                        ],
                        tenant_id="001",
                        title="Test",
                        version=1,
                    ),
                    {},
                )

        assert delta is not None
        assert delta.delta_type == "append"
        assert len(delta.sections_added) == 1

    def test_compare_sections_append(self, generator):
        """Test append delta type detection."""
        current = [
            {"section_type": "executive_summary", "title": "Summary"},
            {"section_type": "trend_analysis", "title": "Trends"},
        ]
        new = [
            {"section_type": "executive_summary", "title": "Summary", "content": "Same"},
            {"section_type": "trend_analysis", "title": "Trends", "content": "Same"},
            {"section_type": "anomaly_alerts", "title": "Anomalies", "content": "New"},
        ]

        result = generator._compare_sections(current, new)

        assert result["type"] == "append"
        assert len(result["sections_added"]) == 1
        assert result["sections_added"][0]["section_type"] == "anomaly_alerts"

    def test_compare_sections_replace_section(self, generator):
        """Test replace_section delta type detection."""
        current = [
            {"section_type": "trend_analysis", "title": "Trends", "content": "Old content"},
        ]
        new = [
            {"section_type": "trend_analysis", "title": "Trends", "content": "New content"},
        ]

        result = generator._compare_sections(current, new)

        assert result["type"] == "replace_section"
        assert len(result["sections_updated"]) == 1

    def test_compare_sections_full_regeneration_threshold(self, generator):
        """Test full_regeneration when >50% sections changed."""
        current = [
            {"section_type": "a", "title": "A"},
            {"section_type": "b", "title": "B"},
            {"section_type": "c", "title": "C"},
            {"section_type": "d", "title": "D"},
        ]
        new = [
            {"section_type": "a", "title": "A", "content": "New A"},
            {"section_type": "b", "title": "B", "content": "New B"},
            {"section_type": "e", "title": "E"},
            {"section_type": "f", "title": "F"},
        ]

        result = generator._compare_sections(current, new)

        # 3/4 changed (75%) -> full_regeneration
        assert result["type"] == "full_regeneration"

    def test_compare_sections_anomaly_regen(self, generator):
        """Test full_regeneration when 3+ new anomalies."""
        current = [
            {"section_type": "executive_summary", "title": "Summary"},
        ]
        new = [
            {"section_type": "executive_summary", "title": "Summary", "content": "Same"},
            {"section_type": "anomaly_alerts", "title": "Anomaly 1", "content": "Anomaly 1"},
            {"section_type": "anomaly_alerts", "title": "Anomaly 2", "content": "Anomaly 2"},
            {"section_type": "anomaly_alerts", "title": "Anomaly 3", "content": "Anomaly 3"},
        ]

        result = generator._compare_sections(current, new)

        # 3 new anomaly sections -> full_regeneration
        assert result["type"] == "full_regeneration"

    def test_generate_summary_append(self, generator):
        """Test summary generation for append."""
        delta = {
            "type": "append",
            "sections_added": [{"title": "Anomaly Alerts"}],
            "sections_updated": [],
            "sections_removed": [],
        }

        summary = generator._generate_summary(delta)

        assert "append" in summary.lower() or "added" in summary.lower()
        assert "Anomaly Alerts" in summary

    def test_generate_summary_replace(self, generator):
        """Test summary for replace_section."""
        delta = {
            "type": "replace_section",
            "sections_added": [],
            "sections_updated": [{"title": "Trend Analysis"}],
            "sections_removed": [],
        }

        summary = generator._generate_summary(delta)

        assert "replace" in summary.lower() or "updated" in summary.lower()
        assert "Trend Analysis" in summary

    def test_generate_summary_full_regeneration(self, generator):
        """Test summary for full_regeneration."""
        delta = {
            "type": "full_regeneration",
            "sections_added": [],
            "sections_updated": [],
            "sections_removed": [],
        }

        summary = generator._generate_summary(delta)

        assert "regenerated" in summary.lower() or "significant" in summary.lower()

    @pytest.mark.asyncio
    async def test_write_delta_to_minio(self, generator, mock_minio_client):
        """Test writing delta to MinIO."""
        delta = mock.MagicMock()
        delta.model_dump.return_value = {"test": "data"}
        delta.tenant_id = "001"
        delta.briefing_id = "briefing-123"

        mock_minio_client.upload = mock.AsyncMock(return_value="s3://stratops-briefings-001/briefing-123/delta.json")

        uri = await generator._write_delta_to_minio("001", "briefing-123", mock.MagicMock(
            model_dump=mock.MagicMock(return_value={}),
            tenant_id="001",
            briefing_id="briefing-123",
        ))

        assert uri.startswith("s3://stratops-briefings-")
        mock_minio_client.upload.assert_called_once()


class TestBriefingDeltaWorker:
    """Tests for BriefingDeltaWorker."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return mock.AsyncMock()

    @pytest.fixture
    def mock_delta_generator(self):
        """Mock delta generator."""
        return mock.AsyncMock()

    @pytest.fixture
    def mock_briefing_repo(self):
        """Mock briefing repo."""
        return mock.AsyncMock()

    @pytest.mark.asyncio
    async def test_worker_start_stop(self, mock_redis, mock_delta_generator, mock_briefing_repo):
        """Test worker start/stop lifecycle."""
        worker = BriefingDeltaWorker(
            redis=mock_redis,
            delta_generator=mock_delta_generator,
            briefing_repo=mock_briefing_repo,
            tenant_id="001",
        )

        await worker.start()
        assert worker._running is True
        assert worker._consume_task is not None

        await worker.stop()
        assert worker._running is False
        assert worker._consume_task is None
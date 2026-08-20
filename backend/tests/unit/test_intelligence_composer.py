"""Unit tests for the Briefing Composer LangGraph node."""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest import mock

import pytest

from backend.intelligence.agents.composer import (
    BriefingComposerNode,
    BriefingSection,
    Briefing,
)


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client."""
    client = mock.AsyncMock()
    client.download = mock.AsyncMock()
    client.upload = mock.AsyncMock(return_value="s3://stratops-briefings-test/briefing-id/v1.md")
    return client


@pytest.fixture
def mock_narrative_client():
    """Mock NarrativeService HTTP client."""
    client = mock.AsyncMock()
    mock_response = mock.AsyncMock()
    mock_response.status_code = 200
    mock_response.json = mock.AsyncMock(return_value={
        "narrative": "# Executive Summary\n\nApple reported strong quarterly results.",
        "key_takeaways": ["Record revenue", "Strong iPhone demand"],
        "confidence": 0.85,
    })
    client.post = mock.AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def composer(mock_minio_client, mock_narrative_client):
    """Provide a BriefingComposerNode instance."""
    return BriefingComposerNode(
        minio_client=mock_minio_client,
        narrative_client=mock_narrative_client,
    )


@pytest.fixture
def sample_state() -> dict:
    """Sample IntelligenceState for testing."""
    return {
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "trace_id": "trace-001",
        "signal_uris": [],
        "extracted_entities": [],
        "content_uris": [
            "s3://stratops-correlations-test/trace-001/correlations.json",
            "s3://stratops-trends-test/trace-001/trends.json",
            "s3://stratops-anomalies-test/trace-001/anomalies.json",
        ],
        "extracted_entities": [],
        "correlation_graph_delta": [],
        "briefing_section_uris": [],
    }


class TestBriefingSection:
    """Tests for BriefingSection model."""

    def test_section_creation(self) -> None:
        """Test section creation with required fields."""
        section = BriefingSection(
            section_type="executive_summary",
            title="Executive Summary",
            content="Test content",
            source_uris=["s3://test"],
            confidence=0.9,
        )
        assert section.section_type == "executive_summary"
        assert section.title == "Executive Summary"
        assert section.confidence == 0.9

    def test_confidence_bounds(self) -> None:
        """Test confidence validation."""
        with pytest.raises(ValueError):
            BriefingSection(
                section_type="test",
                title="Test",
                content="Test",
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            BriefingSection(
                section_type="test",
                title="Test",
                content="Test",
                confidence=-0.1,
            )


class TestBriefing:
    """Tests for Briefing model."""

    def test_briefing_creation(self) -> None:
        """Test briefing creation."""
        briefing = Briefing(
            tenant_id="001",
            title="Test Briefing",
            sections=[],
        )
        assert briefing.tenant_id == "001"
        assert briefing.title == "Test Briefing"
        assert briefing.version == 1
        assert briefing.is_current is True
        assert briefing.id  # UUID generated

    def test_metadata_default(self) -> None:
        """Test metadata defaults to empty dict."""
        briefing = Briefing(tenant_id="001", title="Test", sections=[])
        assert briefing.metadata == {}


class TestBriefingComposerNode:
    """Tests for BriefingComposerNode."""

    @pytest.fixture
    def composer(self, mock_minio_client, mock_narrative_client):
        """Provide a BriefingComposerNode instance."""
        return BriefingComposerNode(
            minio_client=mock_minio_client,
            narrative_client=mock_narrative_client,
        )

    @pytest.fixture
    def sample_state(self) -> dict:
        """Sample IntelligenceState for testing."""
        return {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": [
                "s3://stratops-correlations-test/trace-001/correlations.json",
                "s3://stratops-trends-test/trace-001/trends.json",
                "s3://stratops-anomalies-test/trace-001/anomalies.json",
            ],
            "extracted_entities": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

    @pytest.mark.asyncio
    async def test_empty_content_uris_returns_state(self, composer, sample_state) -> None:
        """No content URIs returns unchanged state."""
        state = {**sample_state, "content_uris": []}
        result = await composer(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_download_intelligence(self, composer, mock_minio_client) -> None:
        """Test downloading intelligence from URIs."""
        uris = ["s3://bucket/file1.json", "s3://bucket/file2.json"]
        mock_content1 = json.dumps({"correlations": [{"type": "pricing"}]})
        mock_content2 = json.dumps({"trends": [{"type": "pricing"}]})
        mock_minio_client.download.side_effect = [mock_content1, mock_content2]

        result = await composer._download_intelligence(uris)

        assert len(result) == 2
        assert "correlations" in result["s3://bucket/file1.json"]
        assert "trends" in result["s3://bucket/file2.json"]

    @pytest.mark.asyncio
    async def test_download_handles_errors(self, composer, mock_minio_client) -> None:
        """Download continues on individual failures."""
        mock_minio_client.download.side_effect = [
            json.dumps({"data": "ok"}),
            Exception("Network error"),
        ]

        uris = ["s3://ok", "s3://fail"]
        result = await composer._download_intelligence(uris)

        assert "s3://ok" in result
        assert "s3://fail" not in result  # Failed download skipped

    @pytest.mark.asyncio
    async def test_build_sections_from_intelligence(self, composer, sample_state) -> None:
        """Test building sections from intelligence data."""
        intelligence = {
            "s3://corr": {"correlations": [{"entity_a": {"name": "A"}, "entity_b": {"name": "B"}, "correlation_type": "pricing", "strength": 0.8}]},
            "s3://trend": {"trends": [{"entity_name": "Apple", "trend_type": "pricing", "direction": "up", "z_score": 2.5, "confidence": 0.9}]},
            "s3://anom": {"anomalies": [{"entity_name": "Test", "anomaly_score": -0.8, "severity": "high", "features": {}}]},
            "s3://narr": {"narrative": "Test narrative"},
        }

        sections = await composer._build_sections(intelligence, "001", "trace-001")

        assert len(sections) == 4
        types = [s.section_type for s in sections]
        assert "correlation_analysis" in types
        assert "trend_analysis" in types
        assert "anomaly_alerts" in types
        assert "narrative_synthesis" in types

    @pytest.mark.asyncio
    async def test_format_correlations(self, composer) -> None:
        """Test correlation formatting."""
        correlations = [
            {"entity_a": {"name": "A"}, "entity_b": {"name": "B"}, "correlation_type": "pricing", "strength": 0.8},
            {"entity_a": {"name": "C"}, "entity_b": {"name": "D"}, "correlation_type": "talent", "strength": 0.6},
        ]

        md = composer._format_correlations(correlations)

        # The output uses Unicode arrow (U+2194) within markdown bold
        assert "**A** \u2194 **B**" in md
        assert "pricing" in md
        assert "0.80" in md
        assert "**C** \u2194 **D**" in md

    def test_format_trends(self, composer) -> None:
        """Test trend formatting."""
        trends = [
            {"entity_name": "Apple", "trend_type": "pricing", "direction": "up", "z_score": 2.5, "confidence": 0.9},
            {"entity_name": "Microsoft", "trend_type": "hiring", "direction": "down", "z_score": -1.8, "confidence": 0.7},
        ]

        md = composer._format_trends(trends)

        assert "Apple" in md
        assert "up" in md
        assert "2.50" in md
        assert "Microsoft" in md

    def test_format_anomalies(self, composer) -> None:
        """Test anomaly formatting."""
        anomalies = [
            {"entity_name": "Test Co", "anomaly_score": -1.2, "severity": "high", "features": {"vol": 0.5, "mentions": 3.0}},
        ]

        md = composer._format_anomalies(anomalies)

        assert "Test Co" in md
        assert "HIGH" in md
        assert "1.20" in md
        assert "vol=0.50" in md

    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, composer, mock_narrative_client) -> None:
        """Test executive summary generation via NarrativeService."""
        from backend.intelligence.agents.composer import BriefingSection

        sections = [
            BriefingSection(
                section_type="trend_analysis",
                title="Trend",
                content="Content",
                source_uris=["s3://test"],
                confidence=0.8,
            )
        ]

        # Mock the narrative client post response
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "narrative": "# Executive Brief\n\nApple reported record revenue of $94.8B driven by iPhone and services.\n\nKey developments:\n- iPhone 15 demand strong\n- Services revenue growing\n- Guidance conservative\n\nRecommended actions:\n- Monitor iPhone demand in China\n- Invest in AI services",
                    "key_takeaways": [
                        "Apple reported record $94.8B revenue",
                        "iPhone and services driving growth",
                        "Conservative guidance for next quarter"
                    ],
                    "confidence": 0.85,
                    "model": "test-model"
                }

        mock_narrative_client.post.return_value = MockResponse()

        summary = await composer._generate_executive_summary(sections, "001", "trace-001")

        assert summary is not None
        assert summary.section_type == "executive_summary"
        assert "Apple reported record $94.8B revenue" in summary.content
        assert summary.confidence == 0.85

    @pytest.mark.asyncio

    @pytest.mark.asyncio
    async def test_generate_executive_summary_failure(self, composer, mock_narrative_client) -> None:
        """Executive summary returns None on failure."""
        mock_narrative_client.post.side_effect = Exception("Service down")

        from backend.intelligence.agents.composer import BriefingSection
        summary = await composer._generate_executive_summary([], "001", "trace-001")
        assert summary is None

    @pytest.mark.asyncio
    async def test_write_briefing_to_minio(self, composer, mock_minio_client) -> None:
        """Test writing briefing to MinIO."""
        from backend.intelligence.agents.composer import Briefing, BriefingSection

        briefing = Briefing(
            tenant_id="001",
            title="Test Briefing",
            sections=[
                BriefingSection(
                    section_type="test",
                    title="Test",
                    content="Content",
                    confidence=0.8,
                )
            ],
            metadata={"trace_id": "trace-001"},
        )

        uri = await composer._write_briefing_to_minio("001", briefing)

        assert uri.startswith("s3://stratops-briefings-")
        assert mock_minio_client.upload.call_count == 2  # markdown + metadata

    @pytest.mark.asyncio
    async def test_full_call_pointer_only(
        self, composer, sample_state, mock_minio_client, mock_narrative_client
    ) -> None:
        """Full call test - verify pointer-only state."""
        # Mock MinIO download for intelligence
        mock_minio_client.download.side_effect = [json.dumps(v) for v in [
            {"correlations": [{"entity_a": {"name": "A"}, "entity_b": {"name": "B"}, "correlation_type": "pricing", "strength": 0.8}]},
            {"trends": [{"entity_name": "Test", "trend_type": "pricing", "direction": "up", "z_score": 2.0, "confidence": 0.8}]},
        ]]

        result = await composer(sample_state)

        # Verify pointer-only: no raw content in state
        state_json = json.dumps(result, default=str)
        assert "s3://" in state_json  # Has URIs
        # Briefing content should NOT be in state
        assert "Briefing content" not in state_json

        # Verify state size < 5KB
        state_size = len(json.dumps(result, default=str).encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"

        # Verify briefing URI added
        assert len(result["briefing_section_uris"]) == 1
        assert result["briefing_section_uris"][0].startswith("s3://stratops-briefings-")

    @pytest.mark.asyncio
    async def test_state_size_under_limit(self, composer, mock_minio_client, mock_narrative_client) -> None:
        """State size stays under 5KB even with many sections."""
        state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "content_uris": [f"s3://test/{i}" for i in range(20)],
            "extracted_entities": [],
            "content_uris": [f"s3://test/{i}" for i in range(20)],
            "extracted_entities": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

        # Mock downloads
        mock_data = {}
        for i in range(20):
            mock_data[f"s3://test/{i}"] = {"correlations": [{"entity_a": {"name": f"A{i}"}, "entity_b": {"name": f"B{i}"}, "correlation_type": "pricing", "strength": 0.5}]}
        mock_minio_client.download.side_effect = [json.dumps(v) for v in mock_data.values()]

        result = await composer(state)

        state_size = len(json.dumps(result, default=str).encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"
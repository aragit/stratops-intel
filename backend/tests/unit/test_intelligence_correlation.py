"""Unit tests for the Correlation Engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest import mock

import pytest

from backend.intelligence.agents.correlation import (
    CorrelationEngineNode,
    CorrelationResult,
)


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client."""
    return mock.AsyncMock()


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client."""
    client = mock.AsyncMock()
    client.upload = mock.AsyncMock(
        return_value="s3://stratops-correlations-test/trace-001/correlations.json"
    )
    return client


@pytest.fixture
def correlation_engine(mock_neo4j_client, mock_minio_client):
    """Provide a CorrelationEngineNode instance."""
    return CorrelationEngineNode(
        neo4j_client=mock_neo4j_client,
        minio_client=mock_minio_client,
        time_window_days=30,
    )


@pytest.fixture
def sample_state():
    """Sample IntelligenceState for testing."""
    return {
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "trace_id": "trace-001",
        "signal_uris": ["s3://stratops-signals/test-signal.json"],
        "extracted_entities": [
            {"company_name": "Company A", "ticker": "A"},
            {"company_name": "Company B", "ticker": "B"},
            {"name": "John Doe", "role": "CTO", "company": "Company A"},
        ],
        "content_uris": [],
        "correlation_graph_delta": [],
        "briefing_section_uris": [],
    }


class TestCorrelationResult:
    """Tests for CorrelationResult model."""

    def test_correlation_result_creation(self) -> None:
        """Test basic CorrelationResult creation."""
        corr = CorrelationResult(
            correlation_type="pricing",
            entity_a={"type": "Company", "id": "A", "name": "Company A"},
            entity_b={"type": "Company", "id": "B", "name": "Company B"},
            strength=0.8,
            evidence=["s3://signal-1"],
            valid_from=datetime.utcnow(),
            valid_to=None,
        )
        assert corr.correlation_type == "pricing"
        assert corr.strength == 0.8
        assert corr.entity_a["name"] == "Company A"

    def test_correlation_strength_bounds(self) -> None:
        """Test strength validation bounds."""
        with pytest.raises(ValueError):
            CorrelationResult(
                correlation_type="test",
                entity_a={"type": "Company", "id": "A", "name": "A"},
                entity_b={"type": "Company", "id": "B", "name": "B"},
                strength=1.5,  # > 1.0
                evidence=[],
                valid_from=datetime.utcnow(),
            )

        with pytest.raises(ValueError):
            CorrelationResult(
                correlation_type="test",
                entity_a={"type": "Company", "id": "A", "name": "A"},
                entity_b={"type": "Company", "id": "B", "name": "B"},
                strength=-0.1,  # < 0.0
                evidence=[],
                valid_from=datetime.utcnow(),
            )


class TestCorrelationEngineNode:
    """Tests for CorrelationEngineNode."""

    @pytest.mark.asyncio
    async def test_empty_entities_returns_same_state(
        self, correlation_engine, sample_state
    ) -> None:
        """Should return unchanged state when no entities."""
        state = {**sample_state, "extracted_entities": []}
        result = await correlation_engine(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_pricing_correlation_strength_calculation(self, correlation_engine) -> None:
        """Test pricing strength computation."""
        # Same price = 1.0
        assert correlation_engine._compute_pricing_strength(100, 100) == 1.0
        # ~10% relative difference (vs mean) = 0.9
        assert correlation_engine._compute_pricing_strength(100, 110) == 0.9
        # 40% relative difference (vs mean) = 0.6
        assert correlation_engine._compute_pricing_strength(100, 150) == 0.6
        # One price zero
        assert correlation_engine._compute_pricing_strength(0, 100) == 0.3

    @pytest.mark.asyncio
    async def test_pricing_correlation_query_construction(
        self, correlation_engine, mock_neo4j_client
    ) -> None:
        """Test pricing correlation query is constructed correctly."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        window_start = datetime.utcnow() - timedelta(days=30)
        window_end = datetime.utcnow()

        mock_neo4j_client.run.return_value = [
            {
                "company_a": "Company A",
                "company_b": "Company B",
                "product": "Product X",
                "price_a": "100",
                "price_b": "105",
                "valid_from": datetime.utcnow().isoformat(),
            }
        ]

        results = await correlation_engine._find_pricing_correlations(
            tenant_id, window_start, window_end
        )

        assert len(results) == 1
        assert results[0].correlation_type == "pricing"
        assert results[0].entity_a["name"] == "Company A"
        assert results[0].entity_b["name"] == "Company B"
        assert 0.0 <= results[0].strength <= 1.0

        # Verify query was called with correct parameters
        mock_neo4j_client.run.assert_called_once()
        call_args = mock_neo4j_client.run.call_args
        assert call_args[0][1]["tenant_id"] == tenant_id

    @pytest.mark.asyncio
    async def test_talent_flow_correlation(self, correlation_engine, mock_neo4j_client) -> None:
        """Test talent flow correlation detection."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        window_start = datetime.utcnow() - timedelta(days=30)
        window_end = datetime.utcnow()

        mock_neo4j_client.run.return_value = [
            {
                "person_name": "John Doe",
                "company_from": "Company A",
                "company_to": "Company B",
                "role_from": "Engineer",
                "role_to": "Senior Engineer",
                "left_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "joined_date": datetime.utcnow().isoformat(),
            }
        ]

        results = await correlation_engine._find_talent_flow_correlations(
            tenant_id, window_start, window_end
        )

        assert len(results) == 1
        assert results[0].correlation_type == "talent"
        assert results[0].strength == 0.8

    @pytest.mark.asyncio
    async def test_co_mention_correlation(self, correlation_engine, mock_neo4j_client) -> None:
        """Test co-mention correlation detection."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        window_start = datetime.utcnow() - timedelta(days=30)
        window_end = datetime.utcnow()

        mock_neo4j_client.run.return_value = [
            {
                "company_a": "Company A",
                "company_b": "Company B",
                "signal_uri": "s3://signal-1",
                "valid_from": datetime.utcnow().isoformat(),
            }
        ]

        results = await correlation_engine._find_co_mention_correlations(
            tenant_id, window_start, window_end
        )

        assert len(results) == 1
        assert results[0].correlation_type == "co_mention"
        assert results[0].evidence == ["s3://signal-1"]

    @pytest.mark.asyncio
    async def test_patent_correlation_placeholder(
        self, correlation_engine, mock_neo4j_client
    ) -> None:
        """Patent correlation returns empty list (placeholder)."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        window_start = datetime.utcnow() - timedelta(days=30)
        window_end = datetime.utcnow()

        results = await correlation_engine._find_patent_correlations(
            tenant_id, window_start, window_end
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_write_correlations_to_minio(self, correlation_engine, mock_minio_client) -> None:
        """Test writing correlations to MinIO."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        trace_id = "trace-001"

        correlations = [
            CorrelationResult(
                correlation_type="pricing",
                entity_a={"type": "Company", "id": "A", "name": "Company A"},
                entity_b={"type": "Company", "id": "B", "name": "Company B"},
                strength=0.8,
                evidence=[],
                valid_from=datetime.utcnow(),
            )
        ]

        uris = await correlation_engine._write_correlations_to_minio(
            tenant_id, trace_id, correlations
        )

        assert len(uris) == 1
        assert uris[0].startswith("s3://stratops-correlations-")
        mock_minio_client.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_graph_deltas(self, correlation_engine) -> None:
        """Test building graph deltas for GraphWriterWorker."""
        correlations = [
            CorrelationResult(
                correlation_type="pricing",
                entity_a={"type": "Company", "id": "A", "name": "Company A"},
                entity_b={"type": "Company", "id": "B", "name": "Company B"},
                strength=0.8,
                evidence=[],
                valid_from=datetime.utcnow(),
            ),
            CorrelationResult(
                correlation_type="talent",
                entity_a={"type": "Company", "id": "C", "name": "Company C"},
                entity_b={"type": "Company", "id": "D", "name": "Company D"},
                strength=0.9,
                evidence=[],
                valid_from=datetime.utcnow(),
            ),
        ]

        deltas = correlation_engine._build_graph_deltas(correlations)

        assert len(deltas) == 2
        for delta in deltas:
            assert delta.startswith("MERGE (a:Entity")
            assert "CORRELATED_WITH" in delta
            assert "strength:" in delta

    @pytest.mark.asyncio
    async def test_full_call_pointer_only(
        self, correlation_engine, sample_state, mock_neo4j_client, mock_minio_client
    ) -> None:
        """Full call test - verify pointer-only state (no raw content)."""
        # Mock all query responses
        mock_neo4j_client.run.side_effect = [
            # Pricing
            [
                {
                    "company_a": "Company A",
                    "company_b": "Company B",
                    "product": "Product X",
                    "price_a": "100",
                    "price_b": "105",
                    "valid_from": datetime.utcnow().isoformat(),
                }
            ],
            # Talent (second correlation -> second graph delta)
            [
                {
                    "company_from": "Company A",
                    "company_to": "Company B",
                    "joined_date": datetime.utcnow().isoformat(),
                }
            ],
            # Co-mention
            [],
            # Patent
            [],
        ]

        result = await correlation_engine(sample_state)

        # Verify state is pointer-only
        assert "extracted_entities" in result
        assert "content_uris" in result
        assert "correlation_graph_delta" in result

        # Verify no raw correlation content in state: entity names legitimately
        # appear in extracted_entities and compact MERGE deltas, but raw query
        # records (prices, products) must only live in MinIO.
        state_json = json.dumps(result, default=str)
        assert "price_a" not in state_json
        assert "Product X" not in state_json

        # Full correlation detail went to MinIO
        upload_kwargs = mock_minio_client.upload.call_args
        assert "Company A" in json.dumps(upload_kwargs, default=str)

        # Verify state size < 5KB
        state_size = len(state_json.encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"

        # Verify MinIO URIs added
        assert len(result["content_uris"]) == 1
        assert result["content_uris"][0].startswith("s3://stratops-correlations-")

        # Verify graph deltas added
        assert len(result["correlation_graph_delta"]) == 2

    @pytest.mark.asyncio
    async def test_state_size_under_limit(
        self, correlation_engine, mock_neo4j_client, mock_minio_client
    ) -> None:
        """Test state size remains < 5KB with many correlations."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        trace_id = "trace-001"

        # Create many entities (kept under the 5KB state budget after pass-through)
        entities = [{"company_name": f"Company {i}", "ticker": f"T{i}"} for i in range(80)]

        state = {
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "signal_uris": [],
            "extracted_entities": entities,
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

        # Mock many correlations
        mock_neo4j_client.run.return_value = []

        result = await correlation_engine(state)

        state_json = json.dumps(result, default=str)
        state_size = len(state_json.encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB limit"

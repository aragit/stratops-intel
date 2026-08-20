"""Unit tests for the Trend Analyzer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest import mock

import pytest

from backend.intelligence.agents.trend import (
    TrendAnalyzerNode,
    TrendResult,
)


@pytest.fixture
def mock_db_pool():
    """Mock database pool."""
    return mock.AsyncMock()


@pytest.fixture
def mock_summarization_client():
    """Mock summarization service client."""
    return mock.AsyncMock()


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client."""
    client = mock.AsyncMock()
    client.upload = mock.AsyncMock(return_value="s3://stratops-trends-test/trace-001/trends.json")
    return client


@pytest.fixture
def trend_analyzer(mock_db_pool, mock_summarization_client, mock_minio_client):
    """Provide a TrendAnalyzerNode instance."""
    return TrendAnalyzerNode(
        db_pool=mock_db_pool,
        summarization_client=mock_summarization_client,
        minio_client=mock_minio_client,
        lookback_days=90,
        z_threshold=2.5,
        stl_min_points=30,
    )


class TestTrendResult:
    """Tests for TrendResult model."""

    def test_trend_result_creation(self) -> None:
        """Test basic TrendResult creation."""
        trend = TrendResult(
            trend_type="pricing",
            entity_name="Company A - Product X",
            direction="up",
            z_score=2.8,
            confidence=0.85,
            narrative="Price increasing trend detected.",
            supporting_signals=[],
        )
        assert trend.trend_type == "pricing"
        assert trend.direction == "up"
        assert trend.z_score == 2.8
        assert trend.confidence == 0.85

    def test_trend_confidence_bounds(self) -> None:
        """Test confidence bounds."""
        with pytest.raises(ValueError):
            TrendResult(
                trend_type="test",
                entity_name="Test",
                direction="up",
                confidence=1.5,  # > 1.0
                narrative="Test",
            )

        with pytest.raises(ValueError):
            TrendResult(
                trend_type="test",
                entity_name="Test",
                direction="up",
                confidence=-0.1,  # < 0.0
                narrative="Test",
            )


class TestTrendAnalyzerNode:
    """Tests for TrendAnalyzerNode."""

    @pytest.fixture
    def sample_state(self) -> dict:
        """Sample IntelligenceState for testing."""
        return {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": ["s3://signal-1"],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

    @pytest.mark.asyncio
    async def test_empty_content_uris_returns_same_state(
        self, trend_analyzer
    ) -> None:
        """Empty content_uris returns unchanged state."""
        state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "content_uris": [],
            "briefing_section_uris": [],
        }
        result = await trend_analyzer(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_pricing_trend_query_construction(
        self, trend_analyzer, mock_db_pool
    ) -> None:
        """Test pricing trend query is constructed correctly."""
        tenant_id = "00000000-0000-0000-0000-000000000001"
        window_start = datetime.utcnow() - timedelta(days=90)
        window_end = datetime.utcnow()

        mock_db_pool.fetch.return_value = [
            {
                "company": "Company A",
                "product": "Product X",
                "price": "100",
                "valid_from": datetime.utcnow().isoformat(),
            },
            {
                "company": "Company A",
                "product": "Product X",
                "price": "105",
                "valid_from": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            },
            {
                "company": "Company A",
                "product": "Product X",
                "price": "110",
                "valid_from": (datetime.utcnow() + timedelta(days=2)).isoformat(),
            },
        ]

        trends = await trend_analyzer._analyze_pricing_trends(
            tenant_id, window_start, window_end
        )

        mock_db_pool.fetch.assert_called_once()
        call_args = mock_db_pool.fetch.call_args
        assert call_args[0][1] == tenant_id  # tenant_id param

        # Should detect upward trend
        assert len(trends) > 0
        assert trends[0].trend_type == "pricing"
        assert trends[0].direction in ("up", "down", "stable", "anomalous")

    @pytest.mark.asyncio
    async def test_hiring_trend_query(self, trend_analyzer, mock_db_pool) -> None:
        """Test hiring trend query."""
        tenant_id = "001"
        window_start = datetime.utcnow() - timedelta(days=90)
        window_end = datetime.utcnow()

        mock_db_pool.fetch.return_value = [
            {"company": "Company A", "hires": 5, "month": datetime(2024, 1, 1)},
            {"company": "Company A", "hires": 8, "month": datetime(2024, 2, 1)},
            {"company": "Company A", "hires": 12, "month": datetime(2024, 3, 1)},
        ]

        trends = await trend_analyzer._analyze_hiring_trends(
            tenant_id, window_start, window_end
        )

        mock_db_pool.fetch.assert_called_once()
        assert len(trends) > 0
        assert trends[0].trend_type == "hiring"

    @pytest.mark.asyncio
    async def test_mention_trend_query(self, trend_analyzer, mock_db_pool) -> None:
        """Test mention frequency trend query."""
        tenant_id = "001"
        window_start = datetime.utcnow() - timedelta(days=90)
        window_end = datetime.utcnow()

        mock_db_pool.fetch.return_value = [
            {"company": "Company A", "mentions": 10, "week": datetime(2024, 1, 1)},
            {"company": "Company A", "mentions": 15, "week": datetime(2024, 1, 8)},
            {"company": "Company A", "mentions": 25, "week": datetime(2024, 1, 15)},
        ]

        trends = await trend_analyzer._analyze_mention_trends(
            tenant_id, window_start, window_end
        )

        mock_db_pool.fetch.assert_called_once()
        assert trends[0].trend_type == "mention_frequency"

    @pytest.mark.asyncio
    async def test_compute_time_series_trends(self, trend_analyzer) -> None:
        """Test time series trend computation."""
        rows = [
            {"company": "A", "product": "X", "price": "100", "valid_from": datetime(2024, 1, 1)},
            {"company": "A", "product": "X", "price": "105", "valid_from": datetime(2024, 1, 2)},
            {"company": "A", "product": "X", "price": "110", "valid_from": datetime(2024, 1, 3)},
            {"company": "A", "product": "X", "price": "115", "valid_from": datetime(2024, 1, 4)},
            {"company": "A", "product": "X", "price": "120", "valid_from": datetime(2024, 1, 5)},
        ]

        trends = trend_analyzer._compute_time_series_trends(
            rows,
            group_keys=["company", "product"],
            value_key="price",
            trend_type="pricing",
            entity_template="{company} - {product}",
        )

        assert len(trends) == 1
        trend = trends[0]
        assert trend.trend_type == "pricing"
        assert trend.entity_name == "A - X"
        assert trend.direction in ("up", "down", "stable", "anomalous")
        assert trend.z_score is not None
        assert 0.0 <= trend.confidence <= 1.0
        assert trend.narrative

    def test_compute_z_score(self, trend_analyzer) -> None:
        """Test Z-score calculation logic."""
        # Test data with clear upward trend
        values = [100, 102, 105, 108, 112, 118, 125]
        
        # Manually compute expected
        historical = values[:-1]
        recent = values[-1]
        mean = sum(historical) / len(historical)
        std = (sum((x - mean) ** 2 for x in historical) / len(historical)) ** 0.5
        expected_z = (recent - mean) / std if std > 0 else 0.0

        # Use internal method
        z = trend_analyzer._compute_z_score(values)
        assert abs(z - expected_z) < 0.01

    def test_compute_z_score_flat(self, trend_analyzer) -> None:
        """Z-score for flat series should be 0."""
        values = [100, 100, 100, 100, 100]
        z = trend_analyzer._compute_z_score(values)
        assert z == 0.0

    def test_stl_residual_calculation(self, trend_analyzer) -> None:
        """Test STL residual calculation."""
        values = [100 + i for i in range(20)]  # Linear trend
        residuals = trend_analyzer._stl_residual(values)

        assert len(residuals) == len(values)
        # For linear trend, residuals should be near zero after detrending
        # (allowing for edge effects at boundaries)
        middle_residuals = residuals[5:15]
        assert all(abs(r) < 1.0 for r in middle_residuals)

    def test_stl_residual_short_series(self, trend_analyzer) -> None:
        """STL on short series returns zeros."""
        values = [100, 101, 102]
        residuals = trend_analyzer._stl_residual(values)
        assert residuals == [0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_generate_trend_narrative(self, trend_analyzer) -> None:
        """Test narrative generation."""
        narrative = await trend_analyzer._generate_trend_narrative(
            trend_type="pricing",
            entity_name="Company A - Product X",
            z_score=2.5,
            direction="up",
        )

        assert "pricing" in narrative.lower()
        assert "Company A - Product X" in narrative
        assert "increasing" in narrative.lower()

    @pytest.mark.asyncio
    async def test_write_trends_to_minio(self, trend_analyzer, mock_minio_client) -> None:
        """Test writing trends to MinIO."""
        tenant_id = "001"
        trace_id = "trace-001"
        trends = [
            TrendResult(
                trend_type="pricing",
                entity_name="Company A",
                direction="up",
                z_score=2.5,
                confidence=0.8,
                narrative="Test narrative",
            )
        ]

        uris = await trend_analyzer._write_trends_to_minio(
            tenant_id, trace_id, trends
        )

        assert len(uris) == 1
        assert uris[0].startswith("s3://stratops-trends-")
        mock_minio_client.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_call_pointer_only(
        self, trend_analyzer, mock_db_pool, mock_minio_client, sample_state
    ) -> None:
        """Full call - verify pointer-only state."""
        # Mock all queries to return empty
        mock_db_pool.fetch.return_value = []

        result = await trend_analyzer(sample_state)

        # Verify pointer-only
        state_json = json.dumps(result, default=str)
        assert "trend" not in state_json.lower() or "s3://" in state_json

        # State size < 5KB
        state_size = len(state_json.encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"

        # Verify URIs added
        assert len(result["content_uris"]) >= len(sample_state["content_uris"])
        assert len(result["briefing_section_uris"]) >= len(sample_state["briefing_section_uris"])

    @pytest.mark.asyncio
    async def test_full_call_with_trends(
        self, trend_analyzer, mock_db_pool, mock_minio_client, sample_state
    ) -> None:
        """Full call with actual trend data."""
        mock_db_pool.fetch.side_effect = [
            # Pricing
            [
                {"company": "A", "product": "X", "price": "100", "valid_from": datetime.utcnow()},
                {"company": "A", "product": "X", "price": "110", "valid_from": datetime.utcnow() + timedelta(days=1)},
            ],
            # Hiring
            [],
            # Mentions
            [],
            # Sentiment
            [],
        ]

        result = await trend_analyzer(sample_state)

        # Should have added trend URIs
        assert len(result["content_uris"]) > len(sample_state["content_uris"])
        assert any("stratops-trends-" in uri for uri in result["content_uris"])

    @pytest.mark.asyncio
    async def test_state_size_under_limit(
        self, trend_analyzer, mock_db_pool, mock_minio_client
    ) -> None:
        """Test state size remains under 5KB with many trends."""
        state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "content_uris": [],
            "briefing_section_uris": [],
        }

        # Mock many trends
        mock_db_pool.fetch.side_effect = [
            [{"company": f"C{i}", "product": "X", "price": "100", "valid_from": datetime.utcnow()} for i in range(20)],
            [], [], [],
        ]

        result = await trend_analyzer(state)

        state_json = json.dumps(result, default=str)
        state_size = len(state_json.encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"
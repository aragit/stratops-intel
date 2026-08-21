"""Unit tests for the Anomaly Detector."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from backend.intelligence.agents.anomaly import (
    AnomalyDetectorNode,
    AnomalyResult,
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
    client.upload = mock.AsyncMock(return_value="s3://stratops-anomalies-test/trace-001/anomalies.json")
    return client


@pytest.fixture
def anomaly_detector(mock_db_pool, mock_summarization_client, mock_minio_client):
    """Provide an AnomalyDetectorNode instance."""
    return AnomalyDetectorNode(
        db_pool=mock_db_pool,
        summarization_client=mock_summarization_client,
        minio_client=mock_minio_client,
        contamination=0.1,
        anomaly_threshold=-0.5,
        lookback_days=90,
        retrain_interval_hours=24,
    )


class TestAnomalyResult:
    """Tests for AnomalyResult model."""

    def test_anomaly_result_creation(self) -> None:
        """Test basic AnomalyResult creation."""
        anomaly = AnomalyResult(
            entity_type="Company",
            entity_name="Test Company",
            anomaly_score=-0.8,
            features={"pricing_volatility": 0.3, "mention_frequency_delta": 2.5},
            severity="high",
            recommended_action="Investigate immediately",
        )
        assert anomaly.entity_type == "Company"
        assert anomaly.anomaly_score == -0.8
        assert anomaly.severity == "high"

    def test_severity_values(self) -> None:
        """Test severity field accepts valid values."""
        for severity in ["low", "medium", "high"]:
            anomaly = AnomalyResult(
                entity_type="Company",
                entity_name="Test",
                anomaly_score=-0.5,
                features={},
                severity=severity,
            )
            assert anomaly.severity == severity


class TestAnomalyDetectorNode:
    """Tests for AnomalyDetectorNode."""

    @pytest.fixture
    def sample_state(self) -> dict:
        """Sample IntelligenceState for testing."""
        return {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

    @pytest.mark.asyncio
    async def test_isolation_forest_training(self, anomaly_detector) -> None:
        """Test IsolationForest model training."""
        # Mock training data
        with mock.patch.object(
            anomaly_detector, "_extract_training_features"
        ) as mock_extract:
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 100
            training_data = []
            for _ in range(n_samples):
                training_data.append({
                    "pricing_volatility": np.random.normal(0.1, 0.05),
                    "mention_frequency_delta": np.random.normal(0, 1),
                    "hiring_velocity": np.random.normal(5, 2),
                    "sentiment_variance": np.random.normal(0.1, 0.05),
                })
            mock_extract.return_value = training_data

            await anomaly_detector._train_model("001")

            assert anomaly_detector._model is not None
            assert anomaly_detector._last_trained is not None
            assert anomaly_detector._model.n_estimators == 100
            assert anomaly_detector._model.contamination == 0.1

    def test_severity_classification(self, anomaly_detector) -> None:
        """Test anomaly severity classification."""
        assert anomaly_detector._classify_severity(-1.2) == "high"
        assert anomaly_detector._classify_severity(-0.8) == "medium"
        assert anomaly_detector._classify_severity(-0.5) == "low"
        assert anomaly_detector._classify_severity(-0.4) == "low"

    @pytest.mark.asyncio
    async def test_generate_recommendation(self, anomaly_detector) -> None:
        """Test LLM recommendation generation."""
        anomaly = AnomalyResult(
            entity_type="Company",
            entity_name="Test Company",
            anomaly_score=-1.2,
            features={"pricing_volatility": 0.5, "mention_frequency_delta": 3.0},
            severity="high",
        )

        # Mock summarization client
        mock_response = mock.AsyncMock()
        mock_response.status_code = 200
        mock_response.json = mock.AsyncMock(return_value=[
            {"summaries": ["Investigate pricing volatility immediately."]}
        ])
        anomaly_detector.summarization_client.post.return_value = mock_response

        recommendation = await anomaly_detector._generate_recommendation(
            AnomalyResult(
                entity_type="Company",
                entity_name="Test Company",
                anomaly_score=-1.2,
                features={},
                severity="high",
            )
        )

        assert recommendation is not None
        assert "Investigate" in recommendation

    @pytest.mark.asyncio
    async def test_generate_recommendation_fallback(self, anomaly_detector) -> None:
        """Test fallback recommendation when LLM fails."""
        anomaly_detector.summarization_client.post.side_effect = Exception("Service unavailable")

        anomaly = AnomalyResult(
            entity_type="Company",
            entity_name="Test Company",
            anomaly_score=-1.2,
            features={"feature1": 1.0, "feature2": 2.0},
            severity="high",
        )

        recommendation = await anomaly_detector._generate_recommendation(anomaly)

        # Should return fallback
        assert recommendation is not None
        assert "Test Company" in recommendation
        assert "-0.5" in recommendation or "1.2" in recommendation

    @pytest.mark.asyncio
    async def test_write_anomalies_to_minio(
        self, anomaly_detector, mock_minio_client
    ) -> None:
        """Test writing anomalies to MinIO."""
        tenant_id = "001"
        trace_id = "trace-001"
        anomalies = [
            AnomalyResult(
                entity_type="Company",
                entity_name="Test Co",
                anomaly_score=-0.8,
                features={"feature1": 1.0},
                severity="high",
            )
        ]

        uris = await anomaly_detector._write_anomalies_to_minio(
            tenant_id, trace_id, anomalies
        )

        assert len(uris) == 1
        assert uris[0].startswith("s3://stratops-anomalies-")
        mock_minio_client.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_anomalies_empty_list(
        self, anomaly_detector, mock_minio_client
    ) -> None:
        """Empty anomalies list returns empty URI list."""
        uris = await anomaly_detector._write_anomalies_to_minio(
            "001", "trace-001", []
        )
        assert uris == []
        mock_minio_client.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_call_pointer_only(
        self, anomaly_detector, mock_minio_client
    ) -> None:
        """Full call - verify pointer-only state."""
        state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "content_uris": [],
        }

        # Mock model as trained
        anomaly_detector._model = mock.MagicMock()
        anomaly_detector._model.decision_function.return_value = np.array([0.0])  # Not anomaly

        # Mock entity features to return some data
        with mock.patch.object(
            anomaly_detector, "_extract_entity_features"
        ) as mock_extract:
            mock_extract.return_value = {
                "Company A": {
                    "pricing_volatility": 0.1,
                    "mention_frequency_delta": 0.5,
                    "hiring_velocity": 5.0,
                    "sentiment_variance": 0.05,
                }
            }

            result = await anomaly_detector({"tenant_id": "001", "trace_id": "trace-001"})

        # Should return state (no anomalies detected)
        assert "content_uris" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_mock_model(
        self, anomaly_detector
    ) -> None:
        """Test anomaly detection with mocked model."""
        # Setup mock model
        mock_model = mock.MagicMock()
        # Return score below threshold for first entity (anomaly), above for second
        anomaly_detector._model = mock_model
        anomaly_detector._model.decision_function.side_effect = [
            np.array([-0.8]),  # Anomaly
            np.array([0.1]),   # Normal
        ]

        entity_features = {
            "Company A": {
                "pricing_volatility": 0.5,
                "mention_frequency_delta": 3.0,
                "hiring_velocity": 10.0,
                "sentiment_variance": 0.2,
            },
            "Company B": {
                "pricing_volatility": 0.1,
                "mention_frequency_delta": 0.5,
                "hiring_velocity": 5.0,
                "sentiment_variance": 0.05,
            },
        }

        anomalies = await anomaly_detector._detect_anomalies(entity_features)

        assert len(anomalies) == 1
        assert anomalies[0].entity_name == "Company A"
        assert anomalies[0].severity in ("low", "medium", "high")
        assert anomalies[0].anomaly_score < -0.5

    @pytest.mark.asyncio
    async def test_write_anomalies_empty_list(
        self, anomaly_detector, mock_minio_client
    ) -> None:
        """Test writing empty anomalies list."""
        uris = await anomaly_detector._write_anomalies_to_minio(
            "001", "trace-001", []
        )
        assert uris == []
        mock_minio_client.upload.assert_not_called()

    def test_severity_classification_boundaries(self, anomaly_detector) -> None:
        """Test severity classification at boundaries."""
        assert anomaly_detector._classify_severity(-1.1) == "high"
        assert anomaly_detector._classify_severity(-1.0) == "medium"
        assert anomaly_detector._classify_severity(-0.7) == "low"
        assert anomaly_detector._classify_severity(-0.69) == "low"
        assert anomaly_detector._classify_severity(-0.5) == "low"
        assert anomaly_detector._classify_severity(-0.3) == "low"

"""Anomaly Detector — Isolation Forest for Multivariate Anomaly Detection.

Detects multivariate anomalies in competitive intelligence features.
Uses sklearn.ensemble.IsolationForest for unsupervised anomaly detection.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import IsolationForest

from .extractor import IntelligenceState

logger = structlog.get_logger(__name__)


class AnomalyResult(BaseModel):
    """A detected anomaly with score, features, and optional LLM recommendation."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str = Field(..., description="Entity type: Company, Product")
    entity_name: str = Field(..., description="Entity identifier")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score (negative = more anomalous)")
    features: Dict[str, float] = Field(..., description="Feature values that triggered anomaly")
    severity: str = Field(..., description="Severity: low, medium, high")
    recommended_action: Optional[str] = Field(None, description="LLM-generated recommendation")


class AnomalyDetectorNode:
    """LangGraph node for multivariate anomaly detection using Isolation Forest.

    Features per entity:
    - Pricing volatility (std dev of price changes)
    - Mention frequency delta (recent vs historical)
    - Hiring velocity (hires per month)
    - Sentiment variance (std dev of sentiment scores)

    Process:
    1. Extract features for each entity from time-series data
    2. Fit IsolationForest on historical data (last 90 days)
    3. Predict anomaly scores for current window
    4. Flag anomalies with score < -0.5
    5. Generate LLM recommendations for high-severity anomalies
    """

    def __init__(
        self,
        db_pool: Any,
        summarization_client: Any,
        minio_client: Any,
        contamination: float = 0.1,
        anomaly_threshold: float = -0.5,
        lookback_days: int = 90,
        retrain_interval_hours: int = 24,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            db_pool: Async database connection pool
            summarization_client: HTTP client for BentoML summarization service
            minio_client: MinIO client for writing anomaly results
            contamination: Expected proportion of anomalies in training data
            anomaly_threshold: Score threshold for anomaly flagging (IsolationForest decision function)
            lookback_days: Training data window
            retrain_interval_hours: Model retraining interval
        """
        self.db_pool = db_pool
        self.summarization_client = summarization_client
        self.minio_client = minio_client
        self.contamination = contamination
        self.anomaly_threshold = anomaly_threshold
        self.lookback_days = lookback_days
        self.retrain_interval_hours = retrain_interval_hours

        self._model: Optional[IsolationForest] = None
        self._feature_names: List[str] = [
            "pricing_volatility",
            "mention_frequency_delta",
            "hiring_velocity",
            "sentiment_variance",
        ]
        self._last_trained: Optional[datetime] = None
        self._bucket_prefix = "stratops-anomalies"

    async def __call__(self, state: IntelligenceState) -> IntelligenceState:
        """Detect anomalies from entity features.

        Args:
            state: IntelligenceState with tenant_id, trace_id

        Returns:
            Updated IntelligenceState with anomaly URIs appended
        """
        start_time = time.time()
        tenant_id = state["tenant_id"]
        trace_id = state["trace_id"]

        logger = structlog.get_logger().bind(
            trace_id=trace_id,
            tenant_id=tenant_id,
        )

        logger.info("anomaly_detector_started", trace_id=trace_id)

        # Ensure model is trained
        await self._ensure_model_trained(tenant_id)

        if not self._model:
            logger.warning("model_not_available", trace_id=trace_id)
            return state

        # Extract features for all entities
        entity_features = await self._extract_entity_features(state["tenant_id"])

        if not entity_features:
            logger.info("no_entities_for_anomaly_detection", trace_id=trace_id)
            return state

        # Predict anomaly scores
        anomalies = await self._detect_anomalies(entity_features)

        # Generate LLM recommendations for high-severity anomalies
        for anomaly in anomalies:
            if anomaly.severity == "high":
                anomaly.recommended_action = await self._generate_recommendation(anomaly)

        # Write anomalies to MinIO
        anomaly_uris = await self._write_anomalies_to_minio(
            state["tenant_id"], state["trace_id"], anomalies
        )

        # Build updated state - POINTER ONLY
        new_state: IntelligenceState = {
            **state,
            "content_uris": state.get("content_uris", []) + anomaly_uris,
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "anomaly_detector_completed",
            trace_id=trace_id,
            entity_count=len(entity_features),
            anomaly_count=len(anomalies),
            duration_ms=round(duration_ms, 2),
        )

        return new_state

    async def _ensure_model_trained(self, tenant_id: str) -> None:
        """Ensure IsolationForest model is trained and up to date."""
        now = datetime.utcnow()

        if (self._model is None or
            self._last_trained is None or
            (now - self._last_trained).total_seconds() > self.retrain_interval_hours * 3600):
            
            await self._train_model(tenant_id)

    async def _train_model(self, tenant_id: str) -> None:
        """Train IsolationForest on historical entity features."""
        logger.info("training_anomaly_model", tenant_id=tenant_id)

        # Extract training features
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=self.lookback_days)

        training_features = await self._extract_training_features(
            tenant_id, window_start, window_end
        )

        if len(training_features) < 10:
            structlog.get_logger().warning(
                "insufficient_training_data",
                tenant_id=tenant_id,
                samples=len(training_features),
            )
            return

        # Prepare feature matrix
        X = np.array([list(f.values()) for f in training_features])

        # Train IsolationForest
        self._model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._last_trained = datetime.utcnow()

        logger.info(
            "anomaly_model_trained",
            tenant_id=tenant_id,
            samples=len(X),
            features=len(self._feature_names),
        )

    async def _extract_training_features(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, float]]:
        """Extract feature vectors for all entities in training window."""
        # This would query the database for historical features
        # For now, return mock data
        # In production, query actual time-series data and compute features
        return []

    async def _extract_entity_features(self, tenant_id: str) -> Dict[str, Dict[str, float]]:
        """Extract current feature vectors for all entities.

        Returns dict of entity_name -> {feature_name: value}
        """
        # In production, this would query the database for recent feature values
        # For now, return mock data
        return {}

    async def _detect_anomalies(
        self,
        entity_features: Dict[str, Dict[str, float]],
    ) -> List[AnomalyResult]:
        """Detect anomalies using trained IsolationForest model."""
        if not self._model:
            return []

        anomalies: List[AnomalyResult] = []

        for entity_name, features in entity_features.items():
            # Ensure feature order matches training
            feature_vector = [features.get(name, 0.0) for name in self._feature_names]
            X = np.array([feature_vector])

            # Get anomaly score (lower = more anomalous)
            score = self._model.decision_function(X)[0]
            is_anomaly = score < self.anomaly_threshold

            if is_anomaly:
                severity = self._classify_severity(score)
                anomaly = AnomalyResult(
                    entity_type="Company",  # Would determine from entity
                    entity_name=entity_name,
                    anomaly_score=round(float(score), 4),
                    features=features,
                    severity=severity,
                    recommended_action=None,  # Will be filled for high severity
                )
                anomalies.append(anomaly)

        return anomalies

    def _classify_severity(self, score: float) -> str:
        """Classify anomaly severity based on score.

        Lower score = more anomalous.
        """
        if score < -1.0:
            return "high"
        elif score < -0.7:
            return "medium"
        else:
            return "low"

    async def _generate_recommendation(self, anomaly: AnomalyResult) -> str:
        """Generate LLM recommendation for high-severity anomaly."""
        try:
            prompt = (
                f"A significant competitive anomaly has been detected for {anomaly.entity_name}. "
                f"Score: {anomaly.anomaly_score:.4f}. "
                f"Features: {json.dumps(anomaly.features, default=str)}. "
                f"Severity: {anomaly.severity}. "
                f"Recommend immediate strategic actions."
            )

            # Call summarization service for recommendation
            # This is a placeholder - would call actual summarization service
            response = await self.summarization_client.post(
                "/summarize",
                json={
                    "texts": [prompt],
                    "style": "executive",
                    "tenant_id": "system",
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data[0]["summaries"][0] if data[0]["summaries"] else None

        except Exception as e:
            structlog.get_logger().error(
                "recommendation_generation_failed",
                error=str(e),
                entity=anomaly.entity_name,
            )

        return f"Investigate {anomaly.entity_name} anomaly (score: {anomaly.anomaly_score:.4f}). " \
               f"Key features: {', '.join(f'{k}={v:.2f}' for k, v in anomaly.features.items())}"

    async def _write_anomalies_to_minio(
        self,
        tenant_id: str,
        trace_id: str,
        anomalies: List[AnomalyResult],
    ) -> List[str]:
        """Write anomaly results to MinIO as JSON."""
        if not anomalies:
            return []

        bucket = f"{self._bucket_prefix}-{tenant_id}"
        key = f"{trace_id}/anomalies.json"

        data = {
            "trace_id": trace_id,
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),
            "anomalies": [a.model_dump(mode="json") for a in anomalies],
        }

        json_data = json.dumps(data, default=str)

        try:
            uri = await self.minio_client.upload(
                bucket=bucket,
                key=key,
                data=json_data.encode("utf-8"),
                content_type="application/json",
            )
            return [uri]
        except Exception as e:
            structlog.get_logger().error(
                "anomaly_minio_write_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []
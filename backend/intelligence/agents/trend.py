"""Trend Analyzer — Z-score/STL + LLM Narrative Generation.

Statistical trend detection on time-series signals with LLM narrative generation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .extractor import IntelligenceState

logger = structlog.get_logger(__name__)


class TrendResult(BaseModel):
    """A detected trend with statistical metrics and LLM narrative."""

    model_config = ConfigDict(extra="forbid")

    trend_type: str = Field(..., description="Type: pricing, hiring, mention_frequency, sentiment")
    entity_name: str = Field(..., description="Entity name (company, product, etc.)")
    direction: str = Field(..., description="Direction: up, down, stable, anomalous")
    z_score: float | None = Field(None, description="Z-score of recent vs historical")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    narrative: str = Field(..., description="LLM-generated narrative explanation")
    supporting_signals: list[str] = Field(default_factory=list, description="Signal URIs")


class TrendAnalyzerNode:
    """LangGraph node for statistical trend detection and narrative generation.

    1. Queries PostgreSQL for time-series signals (pricing, hiring, mentions)
    2. Computes Z-scores for recent values vs historical mean
    3. Applies STL decomposition if enough data points (>30)
    4. Detects anomalies: |z| > 2.5 or STL residual spike
    4. Generates LLM narrative for each trend via SummarizationService
    """

    def __init__(
        self,
        db_pool: Any,
        summarization_client: Any,
        minio_client: Any,
        lookback_days: int = 90,
        z_threshold: float = 2.5,
        stl_min_points: int = 30,
    ) -> None:
        """Initialize the trend analyzer.

        Args:
            db_pool: Async database connection pool
            summarization_client: HTTP client for BentoML summarization service
            minio_client: MinIO client for writing trend results
            lookback_days: Historical window for baseline
            z_threshold: Z-score threshold for anomaly detection
            stl_min_points: Minimum data points for STL decomposition
        """
        self.db_pool = db_pool
        self.summarization_client = summarization_client
        self.minio_client = minio_client
        self.lookback_days = lookback_days
        self.z_threshold = z_threshold
        self.stl_min_points = stl_min_points
        self._bucket_prefix = "stratops-trends"

    async def __call__(self, state: IntelligenceState) -> IntelligenceState:
        """Analyze trends from time-series data.

        Args:
            state: IntelligenceState with tenant_id, trace_id, content_uris

        Returns:
            Updated IntelligenceState with trend URIs appended
        """
        start_time = time.time()
        tenant_id = state["tenant_id"]
        trace_id = state["trace_id"]

        logger = structlog.get_logger().bind(
            trace_id=trace_id,
            tenant_id=tenant_id,
        )

        logger.info("trend_analyzer_started", trace_id=trace_id)

        if not state.get("content_uris"):
            logger.warning("no_content_uris_for_trends", trace_id=trace_id)
            return state

        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=self.lookback_days)

        all_trends: list[TrendResult] = []

        # 1. Pricing trends
        pricing_trends = await self._analyze_pricing_trends(tenant_id, window_start, window_end)
        all_trends.extend(pricing_trends)

        # 2. Hiring trends
        hiring_trends = await self._analyze_hiring_trends(tenant_id, window_start, window_end)
        all_trends.extend(hiring_trends)

        # 3. Mention frequency trends
        mention_trends = await self._analyze_mention_trends(tenant_id, window_start, window_end)
        all_trends.extend(mention_trends)

        # 4. Sentiment trends
        sentiment_trends = await self._analyze_sentiment_trends(tenant_id, window_start, window_end)
        all_trends.extend(sentiment_trends)

        # Write trends to MinIO
        trend_uris = await self._write_trends_to_minio(tenant_id, trace_id, all_trends)

        # Build updated state - POINTER ONLY
        new_state: IntelligenceState = {
            **state,
            "content_uris": state.get("content_uris", []) + trend_uris,
            "briefing_section_uris": state.get("briefing_section_uris", []) + trend_uris,
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "trend_analyzer_completed",
            trace_id=trace_id,
            trend_count=len(all_trends),
            duration_ms=round(duration_ms, 2),
            trend_types=[t.trend_type for t in all_trends],
        )

        return new_state

    async def _analyze_pricing_trends(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[TrendResult]:
        """Analyze pricing trends from PRICED_AT relationships."""
        query = """
        SELECT
            c.name as company,
            p.name as product,
            r.price,
            r.valid_from
        FROM priced_at r
        JOIN company c ON c.id = r.company_id
        JOIN product p ON p.id = r.product_id
        WHERE c.tenant_id = $1
          AND r.valid_from >= $2
          AND r.valid_from <= $3
        ORDER BY c.name, p.name, r.valid_from
        """

        try:
            rows = await self.db_pool.fetch(
                query,
                tenant_id,
                window_start,
                window_end,
            )

            # Group by company+product and compute trends
            trends = await self._compute_time_series_trends(
                rows,
                group_keys=["company", "product"],
                value_key="price",
                trend_type="pricing",
                entity_template="{company} - {product}",
            )
            return trends

        except Exception as e:
            structlog.get_logger().error(
                "pricing_trend_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _analyze_hiring_trends(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[TrendResult]:
        """Analyze hiring velocity trends from EMPLOYED_AT relationships."""
        query = """
        SELECT
            c.name as company,
            COUNT(*) as hires,
            DATE_TRUNC('month', r.valid_from) as month
        FROM employed_at r
        JOIN company c ON c.id = r.company_id
        WHERE c.tenant_id = $1
          AND r.valid_from >= $2
          AND r.valid_from <= $3
          AND r.valid_to IS NULL  -- Current employees
        GROUP BY c.name, month
        ORDER BY c.name, month
        """

        try:
            rows = await self.db_pool.fetch(
                query,
                tenant_id,
                window_start,
                window_end,
            )

            trends = await self._compute_time_series_trends(
                rows,
                group_keys=["company"],
                value_key="hires",
                trend_type="hiring",
                entity_template="{company}",
            )
            return trends

        except Exception as e:
            structlog.get_logger().error(
                "hiring_trend_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _analyze_mention_trends(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[TrendResult]:
        """Analyze mention frequency trends from MENTIONED_IN relationships."""
        query = """
        SELECT
            c.name as company,
            COUNT(*) as mentions,
            DATE_TRUNC('week', r.valid_from) as week
        FROM mentioned_in r
        JOIN company c ON c.id = r.company_id
        WHERE c.tenant_id = $1
          AND r.valid_from >= $2
          AND r.valid_from <= $3
        GROUP BY c.name, week
        ORDER BY c.name, week
        """

        try:
            rows = await self.db_pool.fetch(
                query,
                tenant_id,
                window_start,
                window_end,
            )

            trends = await self._compute_time_series_trends(
                rows,
                group_keys=["company"],
                value_key="mentions",
                trend_type="mention_frequency",
                entity_template="{company}",
            )
            return trends

        except Exception as e:
            structlog.get_logger().error(
                "mention_trend_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _analyze_sentiment_trends(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[TrendResult]:
        """Analyze sentiment trends from earnings call analysis."""
        query = """
        SELECT
            c.name as company,
            AVG(s.score) as avg_sentiment,
            DATE_TRUNC('month', r.valid_from) as month
        FROM sentiment_analysis s
        JOIN company c ON c.id = s.company_id
        JOIN signal r ON r.id = s.signal_id
        WHERE c.tenant_id = $1
          AND r.valid_from >= $2
          AND r.valid_from <= $3
        GROUP BY c.name, month
        ORDER BY c.name, month
        """

        try:
            rows = await self.db_pool.fetch(
                query,
                tenant_id,
                window_start,
                window_end,
            )

            trends = await self._compute_time_series_trends(
                rows,
                group_keys=["company"],
                value_key="avg_sentiment",
                trend_type="sentiment",
                entity_template="{company}",
            )
            return trends

        except Exception as e:
            structlog.get_logger().error(
                "sentiment_trend_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    @staticmethod
    def _compute_z_score(values: list[float]) -> float:
        """Compute Z-score of the most recent value vs the historical segment.

        The historical segment is ``values[:-1]`` (population std). Returns
        0.0 for flat or single-value series.
        """
        if len(values) < 2:
            return 0.0
        recent = values[-1]
        historical = values[:-1]
        mean = sum(historical) / len(historical)
        std = (sum((x - mean) ** 2 for x in historical) / len(historical)) ** 0.5
        if std == 0:
            return 0.0
        return float((recent - mean) / std)

    async def _compute_time_series_trends(
        self,
        rows: list[Any],
        group_keys: list[str],
        value_key: str,
        trend_type: str,
        entity_template: str,
    ) -> list[TrendResult]:
        """Compute Z-scores and detect trends from time-series data.

        Returns list of TrendResult objects.
        """
        if not rows:
            return []

        # Group by entity
        from collections import defaultdict

        groups = defaultdict(list)

        for row in rows:
            key = tuple(row[k] for k in group_keys)
            groups[key].append(row)

        trends = []

        for key, series in groups.items():
            if len(series) < 3:
                continue

            # Sort by time
            series.sort(key=lambda r: r.get("valid_from") or r.get("month") or r.get("week"))

            values = [float(r[value_key]) for r in series if value_key in r]
            if len(values) < 3:
                continue

            # Compute Z-score for most recent value
            z_score = self._compute_z_score(values)

            # Determine direction
            if z_score > self.z_threshold:
                direction = "up"
            elif z_score < -self.z_threshold:
                direction = "down"
            elif abs(z_score) > 1.0:
                direction = "up" if z_score > 0 else "down"
            else:
                direction = "stable"

            # Try STL decomposition if enough points
            _stl_detected = False
            if len(values) >= self.stl_min_points:
                try:
                    stl_residual = self._stl_residual(values)
                    if (
                        abs(stl_residual[-1])
                        > 2 * (sum(r**2 for r in stl_residual) / len(stl_residual)) ** 0.5
                    ):
                        direction = "anomalous"
                        _stl_detected = True
                except Exception:
                    pass

            # Build entity name
            entity_name = entity_template.format(**dict(zip(group_keys, key, strict=False)))

            # Build narrative via LLM (placeholder for now)
            narrative = await self._generate_trend_narrative(
                trend_type=trend_type,
                entity_name=entity_name,
                z_score=z_score,
                direction=direction,
            )

            confidence = min(0.9, max(0.3, abs(z_score) / 3.0))

            trend = TrendResult(
                trend_type=trend_type,
                entity_name=entity_name,
                direction=direction,
                z_score=round(z_score, 2),
                confidence=confidence,
                narrative=narrative,
                supporting_signals=[],
            )

            trends.append(trend)

        return trends

    def _stl_residual(self, values: list[float]) -> list[float]:
        """Simple STL-like residual calculation (placeholder).

        Returns residuals after removing trend and seasonal components.
        """
        if len(values) < 12:
            return [0.0] * len(values)

        # Simple moving average as trend
        window = 3
        trend = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            trend.append(sum(values[start:end]) / (end - start))

        # Residual = value - trend
        residual = [v - t for v, t in zip(values, trend, strict=False)]
        return residual

    async def _generate_trend_narrative(
        self,
        trend_type: str,
        entity_name: str,
        z_score: float,
        direction: str,
    ) -> str:
        """Generate narrative via LLM (SummarizationService).

        For now returns template narrative.
        """
        direction_text = {
            "up": "increasing",
            "down": "decreasing",
            "stable": "stable",
            "anomalous": "showing anomalous behavior",
        }.get(direction, "changing")

        narrative = f"{entity_name} is {direction_text} in {trend_type}. Z-score: {z_score:.2f}. "

        if direction == "up":
            narrative += "This upward trend may indicate competitive pressure or market expansion."
        elif direction == "down":
            narrative += "This downward trend warrants monitoring for potential market contraction."
        elif direction == "anomalous":
            narrative += "Anomalous behavior detected - investigate underlying causes."

        return narrative

    async def _write_trends_to_minio(
        self,
        tenant_id: str,
        trace_id: str,
        trends: list[TrendResult],
    ) -> list[str]:
        """Write trend results to MinIO as JSON."""
        if not trends:
            return []

        bucket = f"{self._bucket_prefix}-{tenant_id}"
        key = f"{trace_id}/trends.json"

        data = {
            "trace_id": trace_id,
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),
            "trends": [t.model_dump(mode="json") for t in trends],
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
                "trend_minio_write_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

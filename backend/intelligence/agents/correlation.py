"""Correlation Engine — Temporal Cypher Queries for Competitive Intelligence.

Implements temporal correlation detection across companies, products, people, and signals.
All Neo4j writes go through the micro-batching GraphWriterWorker (Constraint #4).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel, Field

from .extractor import IntelligenceState

logger = structlog.get_logger(__name__)


class CorrelationResult(BaseModel):
    """A detected temporal correlation between entities."""

    correlation_type: str = Field(..., description="Type: pricing, talent, co_mention, patent")
    entity_a: dict[str, Any] = Field(..., description="First entity: type, id, name")
    entity_b: dict[str, Any] = Field(..., description="Second entity: type, id, name")
    strength: float = Field(..., ge=0.0, le=1.0, description="Correlation strength 0.0-1.0")
    evidence: list[str] = Field(default_factory=list, description="Signal URIs supporting this correlation")
    valid_from: datetime = Field(..., description="Correlation start timestamp")
    valid_to: datetime | None = Field(None, description="Correlation end timestamp")


class CorrelationEngineNode:
    """LangGraph node that detects temporal correlations from extracted entities.

    Queries Neo4j for temporal patterns:
    - Competitor pricing correlation (same product, different companies)
    - Talent flow correlation (person moves between companies within 90 days)
    - Co-mention correlation (companies mentioned in same signal within 7 days)
    - Patent citation correlation (prepared for future patent adapter)
    """

    def __init__(
        self,
        neo4j_client: Any,
        minio_client: Any,
        time_window_days: int = 30,
    ) -> None:
        """Initialize the correlation engine.

        Args:
            neo4j_client: Neo4jClient instance for queries
            minio_client: MinIO client for writing correlation results
            time_window_days: Historical window for correlation queries
        """
        self.neo4j_client = neo4j_client
        self.minio_client = minio_client
        self.time_window_days = time_window_days
        self._bucket_prefix = "stratops-correlations"

    async def __call__(self, state: IntelligenceState) -> IntelligenceState:
        """Detect correlations from extracted entities and graph deltas.

        Args:
            state: IntelligenceState with extracted_entities and correlation_graph_delta

        Returns:
            Updated IntelligenceState with correlation URIs appended
        """
        start_time = time.time()
        tenant_id = state["tenant_id"]
        trace_id = state["trace_id"]

        logger = structlog.get_logger().bind(
            trace_id=trace_id,
            tenant_id=tenant_id,
        )

        logger.info("correlation_engine_started", trace_id=trace_id)

        if not state.get("extracted_entities"):
            logger.warning("no_entities_for_correlation", trace_id=trace_id)
            return state

        # Build time window
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=self.time_window_days)

        # Run correlation queries
        all_correlations: list[CorrelationResult] = []

        # 1. Pricing correlation
        pricing_correlations = await self._find_pricing_correlations(
            tenant_id, window_start, window_end
        )
        all_correlations.extend(pricing_correlations)

        # 2. Talent flow correlation
        talent_correlations = await self._find_talent_flow_correlations(
            tenant_id, window_start, window_end
        )
        all_correlations.extend(talent_correlations)

        # 3. Co-mention correlation
        mention_correlations = await self._find_co_mention_correlations(
            tenant_id, window_start, window_end
        )
        all_correlations.extend(mention_correlations)

        # 4. Patent correlation (placeholder for future)
        patent_correlations = await self._find_patent_correlations(
            tenant_id, window_start, window_end
        )
        all_correlations.extend(patent_correlations)

        # Write correlations to MinIO
        correlation_uris = await self._write_correlations_to_minio(
            tenant_id, trace_id, all_correlations
        )

        # Build correlation graph deltas for GraphWriterWorker
        graph_deltas = self._build_graph_deltas(all_correlations)

        # Build updated state - POINTER ONLY
        new_state: IntelligenceState = {
            **state,
            "content_uris": state.get("content_uris", []) + correlation_uris,
            "correlation_graph_delta": state.get("correlation_graph_delta", []) + graph_deltas,
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "correlation_engine_completed",
            trace_id=trace_id,
            correlation_count=len(all_correlations),
            duration_ms=round(duration_ms, 2),
            correlation_types=[c.correlation_type for c in all_correlations],
        )

        return new_state

    async def _find_pricing_correlations(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[CorrelationResult]:
        """Find competitor pricing correlations for same product.

        Cypher pattern:
        MATCH (c1:Company)-[r1:PRICED_AT]->(p:Product)<-[r2:PRICED_AT]-(c2:Company)
        WHERE c1.tenant_id = $tenant_id AND r1.valid_from >= $window_start
        RETURN c1.name, c2.name, p.name, r1.price, r2.price, r1.valid_from
        """
        query = """
        MATCH (c1:Company)-[r1:PRICED_AT]->(p:Product)<-[r2:PRICED_AT]-(c2:Company)
        WHERE c1.tenant_id = $tenant_id
          AND c2.tenant_id = $tenant_id
          AND c1.name < c2.name
          AND r1.valid_from >= $window_start
          AND r1.valid_from <= $window_end
        RETURN c1.name AS company_a, c2.name AS company_b,
               p.name AS product, r1.price AS price_a, r2.price AS price_b,
               r1.valid_from AS valid_from
        """

        try:
            results = await self.neo4j_client.run(
                query,
                {
                    "tenant_id": tenant_id,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                },
            )

            correlations = []
            for row in results:
                price_a = float(row.get("price_a", 0))
                price_b = float(row.get("price_b", 0))
                # Strength based on price proximity and recency
                strength = self._compute_pricing_strength(price_a, price_b)

                corr = CorrelationResult(
                    correlation_type="pricing",
                    entity_a={"type": "Company", "id": row["company_a"], "name": row["company_a"]},
                    entity_b={"type": "Company", "id": row["company_b"], "name": row["company_b"]},
                    strength=strength,
                    evidence=[],  # Would add signal URIs from PRICED_AT relationships
                    valid_from=datetime.fromisoformat(row["valid_from"]),
                    valid_to=None,
                )
                correlations.append(corr)

            return correlations

        except Exception as e:
            structlog.get_logger().error(
                "pricing_correlation_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _find_talent_flow_correlations(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[CorrelationResult]:
        """Find talent flow between companies (person moves within 90 days).

        Cypher pattern:
        MATCH (person:Person)-[r1:EMPLOYED_AT]->(c1:Company),
              (person)-[r2:EMPLOYED_AT]->(c2:Company)
        WHERE c1.tenant_id = $tenant_id AND r1.valid_to IS NOT NULL
          AND r2.valid_from >= r1.valid_to
          AND duration.between(r1.valid_to, r2.valid_from).days <= 90
        """
        query = """
        MATCH (person:Person)-[r1:EMPLOYED_AT]->(c1:Company),
              (person)-[r2:EMPLOYED_AT]->(c2:Company)
        WHERE c1.tenant_id = $tenant_id
          AND c2.tenant_id = $tenant_id
          AND c1.name <> c2.name
          AND r1.valid_to IS NOT NULL
          AND r2.valid_from >= r1.valid_to
          AND r2.valid_from <= $window_end
          AND duration.between(r1.valid_to, r2.valid_from).days <= 90
        RETURN person.name AS person_name, c1.name AS company_from,
               c2.name AS company_to, r1.role AS role_from, r2.role AS role_to,
               r1.valid_to AS left_date, r2.valid_from AS joined_date
        """

        try:
            results = await self.neo4j_client.run(
                query,
                {
                    "tenant_id": tenant_id,
                    "window_end": window_end.isoformat(),
                },
            )

            correlations = []
            for row in results:
                corr = CorrelationResult(
                    correlation_type="talent",
                    entity_a={"type": "Company", "id": row["company_from"], "name": row["company_from"]},
                    entity_b={"type": "Company", "id": row["company_to"], "name": row["company_to"]},
                    strength=0.8,  # Talent flow is strong signal
                    evidence=[],
                    valid_from=datetime.fromisoformat(row["joined_date"]),
                    valid_to=None,
                )
                correlations.append(corr)

            return correlations

        except Exception as e:
            structlog.get_logger().error(
                "talent_flow_correlation_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _find_co_mention_correlations(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[CorrelationResult]:
        """Find co-mention correlations (companies in same signal within 7 days).

        Cypher pattern:
        MATCH (c1:Company)-[r1:MENTIONED_IN]->(s:Signal)<-[r2:MENTIONED_IN]-(c2:Company)
        WHERE c1.tenant_id = $tenant_id AND r1.valid_from >= $window_start
        """
        query = """
        MATCH (c1:Company)-[r1:MENTIONED_IN]->(s:Signal)<-[r2:MENTIONED_IN]-(c2:Company)
        WHERE c1.tenant_id = $tenant_id
          AND c2.tenant_id = $tenant_id
          AND c1.name < c2.name
          AND r1.valid_from >= $window_start
          AND r1.valid_from <= $window_end
        RETURN c1.name AS company_a, c2.name AS company_b,
               s.source_url AS signal_uri, r1.valid_from AS valid_from
        """

        try:
            results = await self.neo4j_client.run(
                query,
                {
                    "tenant_id": tenant_id,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                },
            )

            correlations = []
            for row in results:
                corr = CorrelationResult(
                    correlation_type="co_mention",
                    entity_a={"type": "Company", "id": row["company_a"], "name": row["company_a"]},
                    entity_b={"type": "Company", "id": row["company_b"], "name": row["company_b"]},
                    strength=0.5,  # Co-mention is moderate signal
                    evidence=[row["signal_uri"]] if row.get("signal_uri") else [],
                    valid_from=datetime.fromisoformat(row["valid_from"]),
                    valid_to=None,
                )
                correlations.append(corr)

            return correlations

        except Exception as e:
            structlog.get_logger().error(
                "co_mention_correlation_query_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    async def _find_patent_correlations(
        self,
        tenant_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[CorrelationResult]:
        """Find patent citation correlations (placeholder for future patent adapter).

        This is a placeholder - the patent adapter will create CITES relationships
        between Patent nodes. For now, return empty list.
        """
        return []

    def _compute_pricing_strength(self, price_a: float, price_b: float) -> float:
        """Compute pricing correlation strength based on price proximity.

        Returns 0.0-1.0 where 1.0 = identical pricing.
        """
        if price_a == 0 and price_b == 0:
            return 0.5
        if price_a == 0 or price_b == 0:
            return 0.3

        diff = abs(price_a - price_b)
        avg = (price_a + price_b) / 2
        relative_diff = diff / avg if avg > 0 else 1.0

        # Strength decreases with relative difference
        strength = max(0.0, 1.0 - relative_diff)
        return round(min(1.0, max(0.0, strength)), 2)

    async def _write_correlations_to_minio(
        self,
        tenant_id: str,
        trace_id: str,
        correlations: list[CorrelationResult],
    ) -> list[str]:
        """Write correlation results to MinIO as JSON.

        Returns list of S3 URIs.
        """
        if not correlations:
            return []

        bucket = f"{self._bucket_prefix}-{tenant_id}"
        key = f"{trace_id}/correlations.json"

        # Serialize correlations
        data = {
            "trace_id": trace_id,
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),
            "correlations": [c.model_dump(mode="json") for c in correlations],
        }

        json_data = json.dumps(data, default=str)

        try:
            # Use async minio client to write
            uri = await self.minio_client.upload(
                bucket=bucket,
                key=key,
                data=json_data.encode("utf-8"),
                content_type="application/json",
            )
            return [uri]
        except Exception as e:
            structlog.get_logger().error(
                "correlation_minio_write_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            return []

    def _build_graph_deltas(self, correlations: list[CorrelationResult]) -> list[str]:
        """Build correlation graph deltas for GraphWriterWorker.

        Returns list of Cypher MERGE statements as strings.
        """
        deltas = []
        for corr in correlations:
            entity_a_id = corr.entity_a.get("id", corr.entity_a.get("name"))
            entity_b_id = corr.entity_b.get("id", corr.entity_b.get("name"))
            delta = f"MERGE (a:Entity {{id: '{entity_a_id}'}})-[r:CORRELATED_WITH {{type: '{corr.correlation_type}', strength: {corr.strength}, valid_from: '{corr.valid_from.isoformat()}'}}]->(b:Entity {{id: '{entity_b_id}'}})"
            deltas.append(delta)
        return deltas

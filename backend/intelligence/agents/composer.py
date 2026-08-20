"""Briefing Composer — Assembles intelligence into structured briefings.

LangGraph node that reads intelligence URIs from state, downloads content from MinIO,
calls NarrativeService for synthesis, assembles structured briefings, and writes
to MinIO with pointer-only state.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .extractor import IntelligenceState

logger = structlog.get_logger(__name__)


class BriefingSection(BaseModel):
    """A section within a briefing.

    Attributes:
        section_type: Type of section (executive_summary, competitive_landscape, etc.)
        title: Human-readable section title
        content: Markdown content of the section
        source_uris: List of MinIO URIs pointing to evidence
        generated_at: Timestamp when section was generated
        confidence: Confidence score 0.0-1.0
    """

    model_config = ConfigDict(extra="forbid")

    section_type: str = Field(..., description="Section type: executive_summary, competitive_landscape, threat_assessment, trend_analysis, anomaly_alerts")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Markdown content")
    source_uris: List[str] = Field(default_factory=list, description="Evidence URIs")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Briefing(BaseModel):
    """Complete briefing document.

    Attributes:
        id: Unique briefing identifier
        tenant_id: Tenant identifier
        title: Briefing title
        sections: List of briefing sections
        generated_at: Generation timestamp
        version: Briefing version (increments on updates)
        is_current: Whether this is the current version
        metadata: Generation parameters and trace info
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    title: str = Field(..., description="Briefing title")
    sections: List[BriefingSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, ge=1)
    is_current: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BriefingComposerNode:
    """LangGraph node that composes intelligence into structured briefings.

    Reads intelligence URIs from state, downloads content from MinIO,
    calls NarrativeService for synthesis, assembles briefing, writes to MinIO.
    """

    def __init__(
        self,
        minio_client: Any,
        narrative_client: Any,
        default_title: str = "Competitive Intelligence Briefing",
    ) -> None:
        """Initialize the briefing composer.

        Args:
            minio_client: MinIO client for downloading/uploading
            narrative_client: HTTP client for NarrativeService
            default_title: Default briefing title
        """
        self.minio_client = minio_client
        self.narrative_client = narrative_client
        self.default_title = default_title
        self._bucket_prefix = "stratops-briefings"

    async def __call__(self, state: IntelligenceState) -> IntelligenceState:
        """Compose briefing from intelligence URIs in state.

        Args:
            state: IntelligenceState with content_uris containing intelligence URIs

        Returns:
            Updated IntelligenceState with briefing URI appended
        """
        start_time = time.time()
        tenant_id = state["tenant_id"]
        trace_id = state["trace_id"]

        logger = structlog.get_logger().bind(
            trace_id=trace_id,
            tenant_id=tenant_id,
        )

        logger.info("briefing_composer_started", trace_id=trace_id)

        # Check if we have intelligence URIs to work with
        content_uris = state.get("content_uris", [])
        if not content_uris:
            logger.warning("no_intelligence_uris_for_briefing", trace_id=trace_id)
            return state

        # Download and parse intelligence from URIs
        intelligence_data = await self._download_intelligence(content_uris)

        # Build briefing sections from intelligence
        sections = await self._build_sections(intelligence_data, tenant_id, trace_id)

        # Generate executive summary via NarrativeService
        executive_summary = await self._generate_executive_summary(sections, tenant_id, trace_id)
        if executive_summary:
            sections.insert(0, executive_summary)

        # Assemble briefing
        briefing = Briefing(
            tenant_id=tenant_id,
            title=self.default_title,
            sections=sections,
            metadata={
                "trace_id": trace_id,
                "source_uris": content_uris,
                "section_count": len(sections),
                "generated_by": "BriefingComposerNode",
            },
        )

        # Write briefing to MinIO
        briefing_uri = await self._write_briefing_to_minio(tenant_id, briefing)

        # Build updated state - POINTER ONLY
        new_state: IntelligenceState = {
            **state,
            "briefing_section_uris": state.get("briefing_section_uris", []) + [briefing_uri],
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "briefing_composer_completed",
            trace_id=trace_id,
            section_count=len(sections),
            briefing_uri=briefing_uri,
            duration_ms=round(duration_ms, 2),
        )

        return new_state

    async def _download_intelligence(self, uris: List[str]) -> Dict[str, Any]:
        """Download and parse intelligence JSON from MinIO URIs.

        Args:
            uris: List of S3 URIs pointing to intelligence JSON

        Returns:
            Dict mapping URI to parsed intelligence data
        """
        intelligence = {}

        for uri in uris:
            try:
                content = await self.minio_client.download(uri)
                data = json.loads(content)
                intelligence[uri] = data
                structlog.get_logger().debug("intelligence_downloaded", uri=uri)
            except Exception as e:
                structlog.get_logger().warning(
                    "intelligence_download_failed",
                    uri=uri,
                    error=str(e),
                )

        return intelligence

    async def _build_sections(
        self,
        intelligence: Dict[str, Any],
        tenant_id: str,
        trace_id: str,
    ) -> List[BriefingSection]:
        """Build briefing sections from downloaded intelligence.

        Args:
            intelligence: Downloaded intelligence data keyed by URI
            tenant_id: Tenant identifier
            trace_id: Trace identifier

        Returns:
            List of BriefingSection objects
        """
        sections = []

        # Categorize intelligence by type
        correlations = []
        trends = []
        anomalies = []
        narratives = []

        for uri, data in intelligence.items():
            if "correlations" in data:
                correlations.extend(data["correlations"])
            if "trends" in data:
                trends.extend(data["trends"])
            if "anomalies" in data:
                anomalies.extend(data["anomalies"])
            if "narrative" in data:
                narratives.append({"uri": uri, "content": data["narrative"]})

        # Correlation section
        if correlations:
            sections.append(BriefingSection(
                section_type="correlation_analysis",
                title="Competitive Correlations",
                content=self._format_correlations(correlations),
                source_uris=[uri for uri in intelligence if "correlations" in intelligence[uri]],
                confidence=0.85,
            ))

        # Trend analysis section
        if trends:
            sections.append(BriefingSection(
                section_type="trend_analysis",
                title="Trend Analysis",
                content=self._format_trends(trends),
                source_uris=[uri for uri in intelligence if "trends" in intelligence[uri]],
                confidence=0.8,
            ))

        # Anomaly alerts section
        if anomalies:
            sections.append(BriefingSection(
                section_type="anomaly_alerts",
                title="Anomaly Alerts",
                content=self._format_anomalies(anomalies),
                source_uris=[uri for uri in intelligence if "anomalies" in intelligence[uri]],
                confidence=0.9,
            ))

        # Narrative synthesis section
        if narratives:
            sections.append(BriefingSection(
                section_type="narrative_synthesis",
                title="Narrative Synthesis",
                content="\n\n".join(n["content"] for n in narratives),
                source_uris=[n["uri"] for n in narratives],
                confidence=0.85,
            ))

        return sections

    def _format_correlations(self, correlations: List[Dict[str, Any]]) -> str:
        """Format correlations into markdown."""
        if not correlations:
            return "No correlations detected."

        lines = ["## Competitive Correlations\n"]
        for corr in correlations[:10]:  # Limit to top 10
            entity_a = corr.get("entity_a", {}).get("name", "Unknown")
            entity_b = corr.get("entity_b", {}).get("name", "Unknown")
            corr_type = corr.get("correlation_type", "unknown")
            strength = corr.get("strength", 0)
            lines.append(f"- **{entity_a}** ↔ **{entity_b}** ({corr_type}): strength={strength:.2f}")

        return "\n".join(lines)

    def _format_trends(self, trends: List[Dict[str, Any]]) -> str:
        """Format trends into markdown."""
        if not trends:
            return "No trends detected."

        lines = ["## Trend Analysis\n"]
        for trend in trends[:10]:
            entity = trend.get("entity_name", "Unknown")
            trend_type = trend.get("trend_type", "unknown")
            direction = trend.get("direction", "stable")
            z_score = trend.get("z_score", 0)
            confidence = trend.get("confidence", 0)
            lines.append(f"- **{entity}** ({trend_type}): {direction} (z={z_score:.2f}, conf={confidence:.2f})")

        return "\n".join(lines)

    def _format_anomalies(self, anomalies: List[Dict[str, Any]]) -> str:
        """Format anomalies into markdown."""
        if not anomalies:
            return "No anomalies detected."

        lines = ["## Anomaly Alerts\n"]
        for anomaly in anomalies[:10]:
            entity = anomaly.get("entity_name", "Unknown")
            score = anomaly.get("anomaly_score", 0)
            severity = anomaly.get("severity", "low")
            features = anomaly.get("features", {})
            feat_str = ", ".join(f"{k}={v:.2f}" for k, v in list(features.items())[:3])
            lines.append(f"- **{entity}** [{severity.upper()}]: score={score:.2f} ({feat_str})")

        return "\n".join(lines)

    async def _generate_executive_summary(
        self,
        sections: List[BriefingSection],
        tenant_id: str,
        trace_id: str,
    ) -> Optional[BriefingSection]:
        """Generate executive summary via NarrativeService.

        Args:
            sections: Briefing sections to synthesize
            tenant_id: Tenant identifier
            trace_id: Trace identifier

        Returns:
            Executive summary section or None if generation fails
        """
        if not sections:
            return None

        try:
            # Prepare sections for NarrativeService
            sections_payload = [
                {
                    "title": s.title,
                    "content_uri": s.source_uris[0] if s.source_uris else f"s3://placeholder/{uuid.uuid4()}",
                    "source_type": s.section_type,
                }
                for s in sections
            ]

            # Call NarrativeService
            response = await self.narrative_client.post(
                "http://bentoml-narrative:3000/generate",
                json={
                    "sections": sections_payload,
                    "narrative_type": "executive_brief",
                    "tenant_id": tenant_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                narrative = data.get("narrative", "")
                key_takeaways = data.get("key_takeaways", [])
                confidence = data.get("confidence", 0.7)

                # Combine narrative with key takeaways
                content = narrative
                if key_takeaways:
                    content += "\n\n**Key Takeaways:**\n" + "\n".join(f"- {t}" for t in key_takeaways)

                return BriefingSection(
                    section_type="executive_summary",
                    title="Executive Summary",
                    content=content,
                    source_uris=[],
                    confidence=confidence,
                )

        except Exception as e:
            structlog.get_logger().error(
                "executive_summary_generation_failed",
                error=str(e),
                trace_id=trace_id,
            )

        return None

    async def _write_briefing_to_minio(self, tenant_id: str, briefing: Briefing) -> str:
        """Write briefing to MinIO as markdown and JSON metadata.

        Args:
            tenant_id: Tenant identifier
            briefing: Briefing object to write

        Returns:
            S3 URI of the briefing markdown
        """
        bucket = f"{self._bucket_prefix}-{tenant_id}"
        briefing_id = briefing.id
        version = briefing.version

        # Build markdown content
        md_parts = [
            f"# {briefing.title}\n",
            f"**Generated:** {briefing.generated_at.isoformat()}\n",
            f"**Version:** {briefing.version}\n",
            f"**Tenant:** {tenant_id}\n",
            f"**Trace ID:** {briefing.metadata.get('trace_id', 'N/A')}\n",
            f"**Sections:** {len(briefing.sections)}\n",
            "---\n",
        ]

        for section in briefing.sections:
            md_parts.append(f"## {section.title}\n")
            md_parts.append(f"*Confidence: {section.confidence:.0%}*\n\n")
            md_parts.append(f"{section.content}\n\n")

        markdown = "\n".join(md_parts)

        # Write markdown
        md_key = f"{briefing_id}/v{version}.md"
        md_uri = await self.minio_client.upload(
            bucket=f"{self._bucket_prefix}-{tenant_id}",
            key=md_key,
            data=markdown.encode("utf-8"),
            content_type="text/markdown",
        )

        # Write metadata JSON
        meta_key = f"{briefing_id}/metadata.json"
        meta_data = {
            "briefing_id": briefing_id,
            "tenant_id": tenant_id,
            "title": briefing.title,
            "version": version,
            "is_current": briefing.is_current,
            "generated_at": briefing.generated_at.isoformat(),
            "sections": [s.model_dump(mode="json") for s in briefing.sections],
            "metadata": briefing.metadata,
        }
        await self.minio_client.upload(
            bucket=f"{self._bucket_prefix}-{tenant_id}",
            key=meta_key,
            data=json.dumps(meta_data, default=str).encode("utf-8"),
            content_type="application/json",
        )

        return md_uri
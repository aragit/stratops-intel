"""Briefing Delta Generator — Computes incremental updates to existing briefings.

Compares the current (historical baseline) briefing with a new intelligence
state and generates minimal delta updates: ``append``,
``replace_section``, or ``full_regeneration``.

Thresholds:
- >50% of sections changed vs. the historical baseline → full_regeneration
- 3+ new anomaly sections → full_regeneration
- Content changed on existing sections only → replace_section
- New sections added → append
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ..intelligence.agents.composer import Briefing
from ..intelligence.agents.extractor import IntelligenceState

logger = structlog.get_logger(__name__)


class BriefingDelta(BaseModel):
    """Represents a delta update to a briefing.

    Attributes:
        briefing_id: Briefing identifier
        tenant_id: Tenant identifier
        delta_type: Type of delta (append, replace_section, full_regeneration)
        sections_added: New sections to add
        sections_updated: Existing sections to update
        sections_removed: Section titles to remove
        summary: LLM-generated summary of changes
        generated_at: Generation timestamp
    """

    model_config = ConfigDict(extra="forbid")

    briefing_id: str = Field(..., description="Briefing identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    delta_type: str = Field(..., description="Type: append, replace_section, full_regeneration")
    sections_added: list[dict[str, Any]] = Field(
        default_factory=list, description="New sections to add"
    )
    sections_updated: list[dict[str, Any]] = Field(
        default_factory=list, description="Existing sections to update"
    )
    sections_removed: list[str] = Field(default_factory=list, description="Section titles removed")
    summary: str = Field(..., description="LLM-generated summary of changes")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


def _section_attr(section: Any, attr: str, default: Any = None) -> Any:
    """Read *attr* from a section that is either a model or a plain dict."""
    if hasattr(section, attr):
        return getattr(section, attr)
    if isinstance(section, dict):
        return section.get(attr, default)
    return default


class BriefingDeltaGenerator:
    """Generates incremental delta updates for briefings.

    Compares the current briefing (the historical intelligence baseline)
    against sections rebuilt from the new intelligence state and determines
    the minimal delta needed.
    """

    def __init__(
        self,
        minio_client: Any,
        narrative_client: Any,
        briefing_repo: Any,
        full_regen_threshold: float = 0.5,
        anomaly_regen_threshold: int = 3,
    ) -> None:
        """Initialize the delta generator.

        Args:
            minio_client: MinIO client for reading/writing briefings
            narrative_client: NarrativeService client for generating summaries
            briefing_repo: BriefingRepository for DB operations
            full_regen_threshold: Fraction of sections changed to trigger full regen (0.0-1.0)
            anomaly_regen_threshold: Number of new anomalies to trigger full regen
        """
        self.minio_client = minio_client
        self.narrative_client = narrative_client
        self.briefing_repo = briefing_repo
        self.full_regen_threshold = full_regen_threshold
        self.anomaly_regen_threshold = anomaly_regen_threshold
        self._bucket_prefix = "stratops-briefings"

    async def generate_delta(
        self,
        current_briefing: Briefing,
        new_state: IntelligenceState,
    ) -> BriefingDelta | None:
        """Generate delta between current briefing and new intelligence state.

        Args:
            current_briefing: Current Briefing object (historical baseline)
            new_state: New IntelligenceState with updated intelligence

        Returns:
            BriefingDelta if changes detected, None if no changes
        """
        start_time = time.time()
        tenant_id = current_briefing.tenant_id
        briefing_id = current_briefing.id

        bound_logger = structlog.get_logger().bind(
            briefing_id=briefing_id,
            tenant_id=tenant_id,
        )
        bound_logger.info("delta_generation_started", briefing_id=briefing_id)

        # Historical baseline: sections of the current briefing
        current_sections = current_briefing.sections

        # Build new sections from the incoming intelligence state
        new_sections = await self._build_sections_from_state(new_state)

        # Compare against the historical baseline
        delta = self._compare_sections(
            current_sections=current_sections,
            new_sections=new_sections,
        )

        if delta is None:
            bound_logger.info("no_changes_detected", briefing_id=briefing_id)
            return None

        # Generate LLM summary of the shift
        summary = await self._generate_delta_summary(delta)

        delta_obj = BriefingDelta(
            briefing_id=briefing_id,
            tenant_id=tenant_id,
            delta_type=delta["type"],
            sections_added=delta["sections_added"],
            sections_updated=delta["sections_updated"],
            sections_removed=delta["sections_removed"],
            summary=summary,
        )

        # Persist the delta document
        await self._write_delta_to_minio(tenant_id, briefing_id, delta_obj)

        # Bump the stored version for structural rewrites
        if delta["type"] != "append":
            await self.briefing_repo.set_current_version(
                tenant_id,
                current_briefing.title,
                current_briefing.version + 1,
            )

        duration_ms = (time.time() - start_time) * 1000
        bound_logger.info(
            "delta_generation_completed",
            briefing_id=briefing_id,
            delta_type=delta["type"],
            sections_added=len(delta["sections_added"]),
            sections_updated=len(delta["sections_updated"]),
            sections_removed=len(delta["sections_removed"]),
            duration_ms=round(duration_ms, 2),
        )

        return delta_obj

    async def _build_sections_from_state(self, state: IntelligenceState) -> list[dict[str, Any]]:
        """Build section dicts from the intelligence state.

        Downloads intelligence JSON documents referenced by
        ``state["content_uris"]`` from MinIO and folds them into briefing
        section dicts. Download or parse failures for individual URIs are
        logged and skipped so one corrupt document cannot stall the delta.

        Args:
            state: IntelligenceState with content_uris

        Returns:
            List of section dicts with keys: section_type, title, content,
            source_uris, confidence
        """
        sections: list[dict[str, Any]] = []

        correlations: list[dict[str, Any]] = []
        trends: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []
        narratives: list[str] = []
        correlation_uris: list[str] = []
        trend_uris: list[str] = []
        anomaly_uris: list[str] = []
        narrative_uris: list[str] = []

        content_uris = state.get("content_uris", []) if isinstance(state, dict) else []

        for uri in content_uris:
            try:
                raw = await self.minio_client.download(uri)
                data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                if not isinstance(data, dict):
                    raise ValueError(f"expected JSON object, got {type(data).__name__}")
            except Exception as e:
                structlog.get_logger().warning(
                    "intelligence_download_failed",
                    uri=uri,
                    error=str(e),
                )
                continue

            doc_correlations = data.get("correlations")
            if isinstance(doc_correlations, list) and doc_correlations:
                correlations.extend(doc_correlations)
                correlation_uris.append(uri)

            doc_trends = data.get("trends")
            if isinstance(doc_trends, list) and doc_trends:
                trends.extend(doc_trends)
                trend_uris.append(uri)

            doc_anomalies = data.get("anomalies")
            if isinstance(doc_anomalies, list) and doc_anomalies:
                anomalies.extend(doc_anomalies)
                anomaly_uris.append(uri)

            narrative = data.get("narrative")
            if isinstance(narrative, str) and narrative:
                narratives.append(narrative)
                narrative_uris.append(uri)

        if correlations:
            sections.append(
                {
                    "section_type": "correlation_analysis",
                    "title": "Competitive Correlations",
                    "content": self._format_correlations(correlations),
                    "source_uris": correlation_uris,
                    "confidence": 0.85,
                }
            )

        if trends:
            sections.append(
                {
                    "section_type": "trend_analysis",
                    "title": "Trend Analysis",
                    "content": self._format_trends(trends),
                    "source_uris": trend_uris,
                    "confidence": 0.8,
                }
            )

        if anomalies:
            sections.append(
                {
                    "section_type": "anomaly_alerts",
                    "title": "Anomaly Alerts",
                    "content": self._format_anomalies(anomalies),
                    "source_uris": anomaly_uris,
                    "confidence": 0.9,
                }
            )

        if narratives:
            sections.append(
                {
                    "section_type": "narrative_synthesis",
                    "title": "Narrative Synthesis",
                    "content": "\n\n".join(narratives),
                    "source_uris": narrative_uris,
                    "confidence": 0.85,
                }
            )

        return sections

    @staticmethod
    def _format_correlations(correlations: list[dict[str, Any]]) -> str:
        """Format correlations into markdown."""
        lines = ["## Competitive Correlations", ""]
        for corr in correlations[:10]:
            entity_a = corr.get("entity_a", {})
            entity_a_name = (
                entity_a.get("name", "Unknown") if isinstance(entity_a, dict) else str(entity_a)
            )
            entity_b = corr.get("entity_b", {})
            entity_b_name = (
                entity_b.get("name", "Unknown") if isinstance(entity_b, dict) else str(entity_b)
            )
            strength = BriefingDeltaGenerator._coerce_float(corr.get("strength"), 0.0)
            lines.append(
                f"- **{entity_a_name}** ↔ **{entity_b_name}** "
                f"({corr.get('correlation_type', 'unknown')}): strength={strength:.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_trends(trends: list[dict[str, Any]]) -> str:
        """Format trends into markdown."""
        lines = ["## Trend Analysis", ""]
        for trend in trends[:10]:
            z_score = BriefingDeltaGenerator._coerce_float(trend.get("z_score"), 0.0)
            confidence = BriefingDeltaGenerator._coerce_float(trend.get("confidence"), 0.0)
            lines.append(
                f"- **{trend.get('entity_name', 'Unknown')}** "
                f"({trend.get('trend_type', 'unknown')}): {trend.get('direction', 'stable')} "
                f"(z={z_score:.2f}, conf={confidence:.2f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_anomalies(anomalies: list[dict[str, Any]]) -> str:
        """Format anomalies into markdown."""
        lines = ["## Anomaly Alerts", ""]
        for anomaly in anomalies[:10]:
            score = BriefingDeltaGenerator._coerce_float(anomaly.get("anomaly_score"), 0.0)
            features = anomaly.get("features", {})
            feature_items = list(features.items())[:3] if isinstance(features, dict) else []
            feat_str = ", ".join(
                f"{k}={BriefingDeltaGenerator._coerce_float(v, 0.0):.2f}" for k, v in feature_items
            )
            lines.append(
                f"- **{anomaly.get('entity_name', 'Unknown')}** "
                f"[{str(anomaly.get('severity', 'low')).upper()}]: score={score:.2f} ({feat_str})"
            )
        return "\n".join(lines)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Convert *value* to float, falling back to *default*."""
        if isinstance(value, bool) or value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _compare_sections(
        self,
        current_sections: list[Any],
        new_sections: list[Any],
    ) -> dict[str, Any] | None:
        """Compare new sections against the historical baseline.

        Args:
            current_sections: Sections of the current briefing (baseline).
                May be BriefingSection models or dicts.
            new_sections: Sections rebuilt from the new intelligence state.

        Returns:
            Delta dict describing the shift, or None when nothing changed.
        """
        if not new_sections:
            return None

        if not current_sections:
            return {
                "type": "full_regeneration",
                "sections_added": [self._section_dict(s) for s in new_sections],
                "sections_updated": [],
                "sections_removed": [],
                "summary": self._generate_summary(
                    "full_regeneration",
                    [self._section_dict(s) for s in new_sections],
                    [],
                    [],
                ),
            }

        current_by_type: dict[str, Any] = {}
        for section in current_sections:
            current_by_type[_section_attr(section, "section_type", "")] = section

        new_by_type: dict[str, Any] = {}
        for section in new_sections:
            new_by_type[_section_attr(section, "section_type", "")] = section

        current_types: set[str] = set(current_by_type.keys())
        new_types: set[str] = set(new_by_type.keys())

        added_types = new_types - current_types
        removed_types = current_types - new_types
        common_types = current_types & new_types

        sections_added = [self._section_dict(new_by_type[t]) for t in sorted(added_types)]
        sections_updated: list[dict[str, Any]] = []
        sections_removed = [
            str(current_by_type[t].title) if hasattr(current_by_type[t], "title") else str(t)
            for t in sorted(removed_types)
        ]

        for sec_type in common_types:
            current_content = _section_attr(current_by_type[sec_type], "content", "")
            new_content = _section_attr(new_by_type[sec_type], "content", "")
            if current_content != new_content:
                sections_updated.append(self._section_dict(new_by_type[sec_type]))

        # Shift magnitude relative to the historical baseline
        total_sections = len(current_sections) + len(new_sections)
        changed_count = len(sections_added) + len(sections_updated) + len(sections_removed)
        change_ratio = changed_count / max(1, total_sections)

        new_anomaly_count = sum(
            1 for s in new_sections if _section_attr(s, "section_type", "") == "anomaly_alerts"
        )

        if change_ratio > self.full_regen_threshold:
            delta_type = "full_regeneration"
        elif new_anomaly_count >= self.anomaly_regen_threshold:
            delta_type = "full_regeneration"
        elif sections_updated and not sections_added:
            delta_type = "replace_section"
        elif sections_removed and not sections_added and not sections_updated:
            delta_type = "replace_section"
        elif sections_added or sections_updated:
            delta_type = "append"
        else:
            return None

        summary = self._generate_summary(
            delta_type, sections_added, sections_updated, sections_removed
        )

        return {
            "type": delta_type,
            "sections_added": sections_added,
            "sections_updated": sections_updated,
            "sections_removed": sections_removed,
            "summary": summary,
        }

    @staticmethod
    def _section_dict(section: Any) -> dict[str, Any]:
        """Normalize a section (model or dict) into a section dict."""
        return {
            "section_type": _section_attr(section, "section_type", ""),
            "title": _section_attr(section, "title", ""),
            "content": _section_attr(section, "content", ""),
            "source_uris": _section_attr(section, "source_uris", []),
            "confidence": _section_attr(section, "confidence", 0.7),
        }

    def _generate_summary(
        self,
        delta_type: str,
        sections_added: list[dict[str, Any]],
        sections_updated: list[dict[str, Any]],
        sections_removed: list[str],
    ) -> str:
        """Generate human-readable summary of changes."""
        parts: list[str] = []

        if delta_type == "full_regeneration":
            parts.append("Briefing fully regenerated due to significant changes.")
        elif delta_type == "replace_section":
            parts.append("Updated existing sections with new data.")
        elif delta_type == "append":
            parts.append("Added new sections to briefing.")

        if sections_added:
            added_titles = [str(s.get("title", "Unknown")) for s in sections_added]
            parts.append(f"Added sections: {', '.join(added_titles)}.")

        if sections_updated:
            updated_titles = [str(s.get("title", "Unknown")) for s in sections_updated]
            parts.append(f"Updated sections: {', '.join(updated_titles)}.")

        if sections_removed:
            parts.append(f"Removed sections: {', '.join(sections_removed)}.")

        return " ".join(parts)

    async def _generate_delta_summary(self, delta: dict[str, Any]) -> str:
        """Generate an LLM summary of the delta via the narrative service.

        Falls back to the deterministic template summary when the narrative
        client is unavailable or fails.
        """
        if self.narrative_client is not None:
            try:
                response = await self.narrative_client.post(
                    "http://bentoml-narrative:3000/generate",
                    json={
                        "narrative_type": "briefing_delta",
                        "delta_type": delta["type"],
                        "sections_added": delta["sections_added"],
                        "sections_updated": delta["sections_updated"],
                        "sections_removed": delta["sections_removed"],
                    },
                )
                if getattr(response, "status_code", None) == 200:
                    data = response.json()
                    narrative = data.get("narrative")
                    if isinstance(narrative, str) and narrative:
                        return narrative
            except Exception as e:
                logger.warning("delta_summary_llm_failed", error=str(e))

        return self._generate_summary(
            delta["type"],
            delta["sections_added"],
            delta["sections_updated"],
            delta["sections_removed"],
        )

    async def _write_delta_to_minio(
        self,
        tenant_id: str,
        briefing_id: str,
        delta: BriefingDelta,
    ) -> str:
        """Write the delta document to MinIO under the tenant bucket."""
        key = f"{briefing_id}/delta_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        data = delta.model_dump(mode="json")
        json_data = json.dumps(data, default=str)

        uri = await self.minio_client.upload(
            bucket=f"{self._bucket_prefix}-{tenant_id}",
            key=key,
            data=json_data.encode("utf-8"),
            content_type="application/json",
        )
        return str(uri)


class BriefingDeltaWorker:
    """Stream consumer that generates briefing deltas.

    Consumes from ``stratops:tenant:{tenant_id}:intelligence`` with consumer
    group ``cg:delta_generator``, loads the current briefing, and generates
    a delta when the new intelligence state shifts it.
    """

    def __init__(
        self,
        redis: Any,
        delta_generator: BriefingDeltaGenerator,
        briefing_repo: Any,
        tenant_id: str,
    ) -> None:
        """Initialize the delta worker.

        Args:
            redis: Redis async client
            delta_generator: BriefingDeltaGenerator instance
            briefing_repo: BriefingRepository for DB operations
            tenant_id: Tenant identifier
        """
        self.redis = redis
        self.delta_generator = delta_generator
        self.briefing_repo = briefing_repo
        self.tenant_id = tenant_id
        self._running = False
        self._consume_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("delta_worker_started", tenant_id=self.tenant_id)

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
            self._consume_task = None
        logger.info("delta_worker_stopped", tenant_id=self.tenant_id)

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        stream_key = f"stratops:tenant:{self.tenant_id}:intelligence"
        consumer_group = "cg:delta_generator"
        consumer_name = f"delta_worker_{self.tenant_id}"

        while True:
            try:
                result = await self.redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_key: ">"},
                    block=5000,
                    count=10,
                )

                if result:
                    for _stream, messages in result:
                        for message_id, message_data in messages:
                            await self._process_message(message_id, message_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("delta_worker_consume_error", error=str(e))
                await asyncio.sleep(0.1)

    async def _process_message(self, message_id: str, message_data: dict[str, Any]) -> None:
        """Process an intelligence update message and generate a delta if needed."""
        stream_key = f"stratops:tenant:{self.tenant_id}:intelligence"
        try:
            briefing = await self.briefing_repo.get_current(
                self.tenant_id, "Competitive Intelligence Briefing"
            )

            if not briefing:
                logger.debug("no_current_briefing", tenant_id=self.tenant_id)
                return

            new_state = message_data.get("state", {})

            delta = await self.delta_generator.generate_delta(
                Briefing(
                    id=briefing.id,
                    tenant_id=briefing.tenant_id,
                    title=briefing.title,
                    sections=briefing.sections,
                    version=briefing.version,
                    metadata=briefing.generated_by or {},
                ),
                new_state,
            )

            if delta:
                logger.info(
                    "delta_generated",
                    briefing_id=briefing.id,
                    delta_type=delta.delta_type,
                    sections_added=len(delta.sections_added),
                )

            await self.redis.xack(stream_key, "cg:delta_generator", message_id)

        except Exception as e:
            logger.error("delta_processing_failed", error=str(e), message_id=message_id)
            await self.redis.xack(stream_key, "cg:delta_generator", message_id)

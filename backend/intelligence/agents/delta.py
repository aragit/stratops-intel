"""Briefing Delta Generator — Computes incremental updates to existing briefings.

Compares current briefing with new intelligence state and generates
incremental delta updates (append, replace_section, full_regeneration).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .composer import Briefing

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
    sections_added: list[dict[str, Any]] = Field(default_factory=list, description="New sections to add")
    sections_updated: list[dict[str, Any]] = Field(default_factory=list, description="Existing sections to update")
    sections_removed: list[str] = Field(default_factory=list, description="Section titles removed")
    summary: str = Field(..., description="LLM-generated summary of changes")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BriefingDeltaGenerator:
    """Generates incremental delta updates for briefings.

    Compares current briefing with new intelligence state and determines
    the minimal delta needed (append, replace_section, or full_regeneration).

    Thresholds:
    - >50% sections changed → full_regeneration
    - 3+ new anomalies → full_regeneration
    - 1-2 new sections → append
    - Section content changed → replace_section
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

    def _get_section_attr(self, section: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from section (handles both BriefingSection and dict)."""
        if hasattr(section, attr):
            return getattr(section, attr)
        return section.get(attr, default)

    async def generate_delta(
        self,
        current_briefing: Briefing,
        new_state: Any,
    ) -> Any | None:
        """Generate delta between current briefing and new intelligence state.

        Args:
            current_briefing: Current Briefing object
            new_state: New IntelligenceState with updated intelligence

        Returns:
            BriefingDelta if changes detected, None if no changes
        """
        start_time = time.time()
        tenant_id = current_briefing.tenant_id
        briefing_id = current_briefing.id

        logger = structlog.get_logger().bind(
            briefing_id=briefing_id,
            tenant_id=current_briefing.tenant_id,
        )

        logger.info("delta_generation_started", briefing_id=briefing_id)

        # Download current briefing content from MinIO
        current_sections = current_briefing.sections

        # Build new sections from current intelligence state
        new_sections = await self._build_sections_from_state(new_state)

        # Compare sections
        delta = self._compare_sections(
            current_sections=current_briefing.sections,
            new_sections=new_sections,
        )

        if delta is None:
            logger.info("no_changes_detected", briefing_id=briefing_id)
            return None

        # Generate summary via LLM
        summary = await self._generate_delta_summary(delta)

        # Build delta object
        delta_obj = BriefingDelta(
            briefing_id=briefing_id,
            tenant_id=current_briefing.tenant_id,
            delta_type=delta["type"],
            sections_added=delta["sections_added"],
            sections_updated=delta["sections_updated"],
            sections_removed=delta["sections_removed"],
            summary=delta["summary"],
        )

        # Write delta to MinIO
        delta_uri = await self._write_delta_to_minio(
            current_briefing.tenant_id, briefing_id, delta_obj
        )

        # Update briefing version in DB if needed
        if delta["type"] != "append":
            await self.briefing_repo.set_current_version(
                current_briefing.tenant_id,
                current_briefing.title,
                current_briefing.version + 1,
            )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "delta_generation_completed",
            briefing_id=briefing_id,
            delta_type=delta["type"],
            sections_added=len(delta["sections_added"]),
            sections_updated=len(delta["sections_updated"]),
            sections_removed=len(delta["sections_removed"]),
            duration_ms=round(time.time() - start_time, 2),
        )

        return delta_obj

    def _get_section_attr(self, section: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from section (handles both BriefingSection and dict)."""
        if hasattr(section, attr):
            return getattr(section, attr)
        return section.get(attr, default)

    async def _build_sections_from_state(self, state: Any) -> list[dict[str, Any]]:
        """Build section dicts from intelligence state.

        Args:
            state: IntelligenceState with content_uris, etc.

        Returns:
            List of section dicts with keys: section_type, title, content, source_uris, confidence
        """
        sections = []

        # This would read from MinIO URIs in state["content_uris"]
        # For now, return empty - actual implementation would download and parse
        return []

    def _compare_sections(
        self,
        current_sections: list[Any],
        new_sections: list[Any],
    ) -> dict[str, Any] | None:
        """Compare current and new sections to determine delta type.

        Args:
            current_sections: Current briefing sections (BriefingSection objects)
            new_sections: New sections from intelligence state (dicts)

        Returns:
            Delta dict or None if no changes
        """
        if not current_sections and not new_sections:
            return None

        # Map sections by type for comparison
        def get_section_type(s):
            if hasattr(s, 'section_type'):
                return s.section_type
            return s.get("section_type", "")

        def get_attr(s, attr, default=None):
            if hasattr(s, attr):
                return getattr(s, attr)
            return s.get(attr, None)

        current_by_type = {self._get_section_attr(s, "section_type"): s for s in current_sections}
        new_by_type = {self._get_section_attr(s, "section_type"): s for s in new_sections}

        current_types: set[str] = set(current_by_type.keys())
        new_types: set[str] = set(self._get_section_attr(s, "section_type") for s in new_sections)

        added_types = new_types - current_types
        removed_types = set(current_types) - new_types
        common_types = set(current_by_type.keys()) & set(new_types)

        sections_added = []
        sections_updated = []
        sections_removed = []

        # Check added sections
        for sec_type in (set(self._get_section_attr(s, "section_type") for s in new_sections) - set(s.section_type for s in current_sections)):
            new_sec = next(s for s in new_sections if self._get_section_attr(s, "section_type") == sec_type)
            sections_added.append({
                "section_type": sec_type,
                "title": self._get_section_attr(new_sec, "title", sec_type),
                "content": self._get_section_attr(new_sec, "content", ""),
                "source_uris": self._get_section_attr(new_sec, "source_uris", []),
                "confidence": self._get_section_attr(new_sec, "confidence", 0.7),
            })

        # Check updated sections (content changed)
        common_types = set(self._get_section_attr(s, "section_type") for s in current_sections) & set(self._get_section_attr(s, "section_type") for s in new_sections)
        for sec_type in set(self._get_section_attr(s, "section_type") for s in current_sections) & set(self._get_section_attr(s, "section_type") for s in new_sections):
            current_sec = next(s for s in current_sections if self._get_section_attr(s, "section_type") == sec_type)
            new_sec = next(s for s in new_sections if self._get_section_attr(s, "section_type") == sec_type)

            # Compare content (simplified - could use diff)
            if self._get_section_attr(current_sec, "content") != self._get_section_attr(new_sec, "content", ""):
                sections_updated.append({
                    "section_type": sec_type,
                    "title": self._get_section_attr(new_sec, "title", self._get_section_attr(current_sec, "title")),
                    "content": self._get_section_attr(new_sec, "content", ""),
                    "source_uris": self._get_section_attr(new_sec, "source_uris", []),
                    "confidence": self._get_section_attr(new_sec, "confidence", current_sec.confidence),
                })

        # Determine delta type
        total_sections = len(current_sections) + len(new_sections)
        changed_count = len(sections_added) + len(sections_updated)
        change_ratio = changed_count / max(1, total_sections)

        # Determine delta type based on changes
        # Count new anomaly sections (count individual sections, not grouped by type)
        anomaly_sections = [s for s in new_sections if self._get_section_attr(s, "section_type") == "anomaly_alerts"]

        # Determine delta type based on changes
        if total_sections == 0 and len(new_sections) > 0:
            delta_type = "full_regeneration"
        elif len(sections_updated) > 0 and len(sections_added) == 0:
            delta_type = "replace_section"
        elif change_ratio >= self.full_regen_threshold:
            delta_type = "full_regeneration"
        elif len(sections_added) >= 3:
            delta_type = "full_regeneration"
        elif len(anomaly_sections) >= self.anomaly_regen_threshold:
            delta_type = "full_regeneration"
        elif len(sections_removed) > 0 and len(sections_added) == 0 and len(sections_updated) == 0:
            # Only sections removed, no additions or updates
            delta_type = "replace_section"
        elif len(sections_added) > 0 or len(sections_updated) > 0:
            delta_type = "append"
        else:
            # No changes at all
            return None

        # Generate summary
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

    def _generate_summary(
        self,
        delta_type: str,
        sections_added: list[dict[str, Any]],
        sections_updated: list[dict[str, Any]],
        sections_removed: list[str],
    ) -> str:
        """Generate human-readable summary of changes."""
        parts = []

        if delta_type == "full_regeneration":
            parts.append("Briefing fully regenerated due to significant changes.")
        elif delta_type == "replace_section":
            parts.append("Updated existing sections with new data.")
        elif delta_type == "append":
            parts.append("Added new sections to briefing.")

        if sections_added:
            added_titles = [s.get("title", "Unknown") for s in sections_added]
            parts.append(f"Added sections: {', '.join(added_titles)}.")

        if sections_updated:
            updated_titles = [s.get("title", "Unknown") for s in sections_updated]
            parts.append(f"Updated sections: {', '.join(updated_titles)}.")

        if sections_removed:
            parts.append(f"Removed sections: {', '.join(sections_removed)}.")

        return " ".join(parts)

    async def _generate_delta_summary(self, delta: dict[str, Any]) -> str:
        """Generate LLM summary of delta changes (placeholder)."""
        # In production, would call NarrativeService or SummarizationService
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
        delta: Any,
    ) -> str:
        """Write delta to MinIO."""
        bucket = f"{self._bucket_prefix}-{tenant_id}"
        key = f"{briefing_id}/delta_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        data = delta.model_dump(mode="json")
        json_data = json.dumps(data, default=str)

        uri = await self.minio_client.upload(
            bucket=f"{self._bucket_prefix}-{tenant_id}",
            key=key,
            data=json_data.encode("utf-8"),
            content_type="application/json",
        )
        return uri


class BriefingDeltaWorker:
    """Stream consumer that generates briefing deltas.

    Consumes from intelligence stream, loads current briefing,
    generates delta if changes detected.
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
        if self._consume_task is not None:
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
                    for stream, messages in result:
                        for message_id, message_data in messages:
                            await self._process_message(message_id, message_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("delta_worker_consume_error", error=str(e))
                await asyncio.sleep(0.1)

    async def _process_message(self, message_id: str, message_data: dict[str, Any]) -> None:
        """Process intelligence update message and generate delta if needed."""
        try:
            # Get current briefing from DB
            briefing = await self.briefing_repo.get_current(
                self.tenant_id, "Competitive Intelligence Briefing"
            )

            if not briefing:
                logger.debug("no_current_briefing", tenant_id=self.tenant_id)
                return

            # Get new intelligence state from message
            new_state = message_data.get("state", {})

            # Generate delta
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

            await self.redis.xack(
                f"stratops:tenant:{self.tenant_id}:intelligence",
                "cg:delta_generator",
                message_id,
            )

        except Exception as e:
            logger.error("delta_processing_failed", error=str(e), message_id=message_id)
            await self.redis.xack(
                f"stratops:tenant:{self.tenant_id}:intelligence",
                "cg:delta_generator",
                message_id,
            )

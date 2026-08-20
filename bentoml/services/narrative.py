"""Narrative Service — Qwen2.5-14B-AWQ + vLLM Backend.

BentoML service for generating coherent narratives from multiple intelligence sections.
Uses larger context window (16384 tokens) for synthesizing multiple signals.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import bentoml
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class NarrativeSection(BaseModel):
    """A section of intelligence to include in narrative synthesis.

    Attributes:
        title: Section title/heading.
        content_uri: S3 URI to full section content.
        source_type: Type of source (correlation, trend, anomaly, signal).
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Section title")
    content_uri: str = Field(..., description="S3 URI to full content")
    source_type: str = Field(..., description="Type: correlation, trend, anomaly, signal")


class NarrativeRequest(BaseModel):
    """Request for narrative generation.

    Attributes:
        sections: List of intelligence sections to synthesize.
        narrative_type: Type of narrative (executive_brief, competitive_update, threat_assessment).
        tenant_id: Tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(extra="forbid")

    sections: List[NarrativeSection] = Field(..., min_length=1, max_length=20, description="Sections to synthesize")
    narrative_type: str = Field(
        default="executive_brief",
        description="Type: executive_brief, competitive_update, threat_assessment"
    )
    tenant_id: str = Field(..., description="Tenant identifier")


class NarrativeResponse(BaseModel):
    """Response containing generated narrative.

    Attributes:
        narrative: Full markdown narrative.
        key_takeaways: Extracted key takeaways as bullet points.
        confidence: Confidence score 0.0-1.0.
        model: Model identifier used.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(..., description="Full markdown narrative")
    key_takeaways: List[str] = Field(default_factory=list, description="Key takeaway bullet points")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    model: str = Field(..., description="Model identifier")


NARRATIVE_PROMPTS = {
    "executive_brief": (
        "Synthesize the following competitive intelligence into an executive briefing. "
        "Include:\n"
        "1) Situation overview - What is happening in the competitive landscape?\n"
        "2) Key developments - Most important changes and events\n"
        "3) Competitive implications - What this means for our competitive position\n"
        "4) Recommended actions - Specific, actionable next steps\n\n"
        "Sections:\n{sections}\n\n"
        "Executive Briefing:"
    ),
    "competitive_update": (
        "Provide a competitive landscape update focusing on:\n"
        "1) Market position changes - Who gained/lost ground\n"
        "2) Product launches - New offerings and capabilities\n"
        "3) Pricing moves - Competitive pricing dynamics\n"
        "4) Talent movements - Key hires and departures\n\n"
        "Sections:\n{sections}\n\n"
        "Competitive Update:"
    ),
    "threat_assessment": (
        "Assess competitive threats based on:\n"
        "1) New entrants - Emerging competitors or technologies\n"
        "2) Pricing pressure - Margin compression risks\n"
        "3) Technology shifts - Disruptive innovations\n"
        "4) Talent poaching - Key personnel losses\n\n"
        "Sections:\n{sections}\n\n"
        "Threat Assessment:"
    ),
}


@bentoml.service(
    name="narrative-service",
    resources={"gpu": 1, "memory": "24Gi"},
    traffic={"timeout": 120, "concurrency": 4},
)
class NarrativeService:
    """BentoML service for narrative generation using Qwen2.5-14B-AWQ with vLLM.

    Model: Qwen/Qwen2.5-14B-Instruct-AWQ
    Max context: 16384 tokens for multi-section synthesis
    Non-batchable: Each narrative is unique synthesis
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("NARRATIVE_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        self.llm: Any = None
        self.sampling_params: Any = None
        self._initialized = False
        self._request_count = 0
        self._total_duration_ms = 0.0

    async def _load_model(self) -> None:
        """Lazy-load vLLM model on first request."""
        if self._initialized:
            return

        logger.info("loading_narrative_model", model=self.model_id)

        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=self.model_id,
                quantization="awq",
                max_model_len=16384,
                gpu_memory_utilization=0.90,
                enable_prefix_caching=True,
                trust_remote_code=True,
                dtype="half",
            )

            self.sampling_params = SamplingParams(
                temperature=0.4,
                max_tokens=2048,
                stop=["</end>", "<|endoftext|>"],
            )

            # Warmup
            logger.info("warming_up_narrative_model")
            dummy_prompt = "Summarize: Apple and Microsoft compete in cloud."
            self.llm.generate([dummy_prompt], self.sampling_params)

            self._initialized = True
            logger.info("narrative_model_loaded", model=self.model_id)

        except Exception as e:
            logger.error("narrative_model_load_failed", error=str(e))
            raise

    def _build_prompt(self, request: Any) -> str:
        """Build synthesis prompt from sections."""
        template = NARRATIVE_PROMPTS.get(
            request.narrative_type,
            NARRATIVE_PROMPTS["executive_brief"]
        )

        # Build sections text from content URIs (would download in production)
        sections_text = []
        for i, section in enumerate(request.sections, 1):
            # In production, would download content from URI
            # For now, use title as placeholder
            sections_text.append(f"Section {i}: {section.title} [Content at {section.content_uri}]")

        sections_combined = "\n\n".join(sections_text)
        return template.format(sections=sections_combined)

    def _extract_key_takeaways(self, narrative: str) -> List[str]:
        """Extract key takeaways from narrative markdown.

        Looks for bullet points or numbered lists.
        """
        takeaways = []
        lines = narrative.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                takeaways.append(line[2:].strip())
            elif line.startswith("• "):
                takeaways.append(line[2:].strip())
            elif line and line[0].isdigit() and ". " in line[:3]:
                takeaways.append(line.split(". ", 1)[1].strip())

        # Fallback: take first few sentences if no bullets
        if not takeaways:
            sentences = narrative.split(". ")
            takeaways = [s.strip() + "." for s in sentences[:3] if s.strip()]

        return takeaways[:5]  # Max 5 takeaways

    @bentoml.api(batchable=False, max_batch_size=1)
    async def generate(self, request: NarrativeRequest) -> NarrativeResponse:
        """Generate narrative from intelligence sections.

        Args:
            request: NarrativeRequest with sections and type.

        Returns:
            NarrativeResponse with markdown narrative and key takeaways.
        """
        start_time = time.time()

        await self._load_model()

        if not self.llm or not self.sampling_params:
            raise RuntimeError("Narrative model not initialized")

        # Build prompt
        prompt = self._build_prompt(request)

        # Generate narrative
        try:
            generate_start = time.time()
            outputs = self.llm.generate([prompt], self.sampling_params)
            generate_duration_ms = (time.time() - generate_start) * 1000
        except Exception as e:
            logger.error("vllm_narrative_generate_failed", error=str(e))
            return NarrativeResponse(
                narrative="Error generating narrative.",
                key_takeaways=[],
                confidence=0.0,
                model=self.model_id,
            )

        narrative = outputs[0].outputs[0].text.strip()

        # Extract key takeaways
        key_takeaways = self._extract_key_takeaways(narrative)

        # Compute confidence based on narrative length and structure
        confidence = self._compute_confidence(narrative, request.sections)

        total_duration_ms = (time.time() - start_time) * 1000

        self._request_count += 1

        logger.info(
            "narrative_generated",
            narrative_type=request.narrative_type,
            sections_count=len(request.sections),
            duration_ms=round(total_duration_ms, 2),
            narrative_length=len(narrative),
        )

        return NarrativeResponse(
            narrative=narrative,
            key_takeaways=key_takeaways,
            confidence=confidence,
            model=self.model_id,
        )

    def _compute_confidence(self, narrative: str, sections: List[Any]) -> float:
        """Compute confidence score based on narrative quality."""
        if not narrative:
            return 0.0

        base_confidence = 0.5

        # Length factor
        length_score = min(1.0, len(narrative) / 2000)

        # Structure score (has sections, bullet points, etc.)
        structure_score = 0.0
        if "\n\n" in narrative:
            structure_score += 0.2
        if "-" in narrative or "•" in narrative:
            structure_score += 0.2
        if any(c.isdigit() for c in narrative[:100]):  # Numbered sections
            structure_score += 0.1

        # Source coverage
        coverage_score = min(1.0, len(sections) / 5)

        confidence = base_confidence + length_score * 0.2 + structure_score + coverage_score * 0.1
        return min(1.0, max(0.1, confidence))

    @bentoml.api(route="/metrics")
    async def metrics(self) -> dict[str, str]:
        """Prometheus-format metrics endpoint."""
        avg_duration = self._total_duration_ms / max(self._request_count, 1)

        return {
            "metrics": f"""# HELP narrative_requests_total Total narrative requests
# TYPE narrative_requests_total counter
narrative_requests_total {self._request_count}
# HELP narrative_duration_seconds Average duration in seconds
# TYPE narrative_duration_seconds gauge
narrative_duration_seconds {avg_duration / 1000}
"""
        }

    @bentoml.api(route="/health")
    async def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self._initialized else "loading",
            "model": self.model_id,
            "backend": "vllm",
            "max_model_len": 16384,
            "initialized": self._initialized,
        }
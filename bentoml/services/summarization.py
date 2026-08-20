"""Summarization Service — Llama-3.1-8B + vLLM Backend.

BentoML service for document summarization with multiple styles.
Uses vLLM backend (NOT raw transformers) for high-throughput batched inference.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import bentoml
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class SummarizationRequest(BaseModel):
    """Request for document summarization.

    Attributes:
        texts: List of document texts to summarize.
        max_length: Maximum summary length in tokens.
        style: Summary style (executive, technical, bullet_points).
        tenant_id: Tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(extra="forbid")

    texts: List[str] = Field(..., min_length=1, max_length=16, description="Documents to summarize")
    max_length: int = Field(default=256, ge=64, le=1024, description="Max summary length in tokens")
    style: str = Field(default="executive", description="Summary style: executive, technical, bullet_points")
    tenant_id: str = Field(..., description="Tenant identifier")


class SummarizationResponse(BaseModel):
    """Response containing generated summaries.

    Attributes:
        summaries: List of generated summary texts.
        model: Model identifier used.
        batch_size: Number of documents processed.
        total_tokens: Total tokens generated across all summaries.
    """

    model_config = ConfigDict(extra="forbid")

    summaries: List[str] = Field(..., description="Generated summaries")
    model: str = Field(..., description="Model identifier")
    batch_size: int = Field(..., description="Number of documents in batch")
    total_tokens: int = Field(..., description="Total tokens across all summaries")


STYLE_PROMPTS = {
    "executive": (
        "Summarize the following competitive intelligence for an executive audience. "
        "Focus on strategic implications and actionable insights:\n\n{text}\n\nSummary:"
    ),
    "technical": (
        "Summarize the following technical competitive intelligence. "
        "Focus on architecture decisions, tech stack changes, and engineering implications:\n\n{text}\n\nSummary:"
    ),
    "bullet_points": (
        "Summarize the following as 3-5 bullet points highlighting key competitive changes:\n\n{text}\n\nSummary:"
    ),
}


@bentoml.service(
    name="summarization-service",
    resources={"gpu": 1, "memory": "16Gi"},
    traffic={"timeout": 60, "concurrency": 8},
    batching={"max_batch_size": 16, "max_latency_ms": 100, "batch_dim": 0},
)
class SummarizationService:
    """BentoML service for document summarization using Llama-3.1-8B with vLLM.

    Model: meta-llama/Llama-3.1-8B-Instruct (or AWQ variant)
    Max context: 8192 tokens
    Batch size: up to 16
    Styles: executive, technical, bullet_points
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("SUMMARIZATION_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.llm: Optional[Any] = None
        self.sampling_params: Optional[Any] = None
        self._initialized = False
        self._request_count = 0
        self._total_duration_ms = 0.0
        self._total_tokens = 0

    async def _load_model(self) -> None:
        """Lazy-load vLLM model on first request."""
        if self._initialized:
            return

        logger.info("loading_vllm_summarization_model", model=self.model_id)

        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=self.model_id,
                quantization="awq" if "awq" in self.model_id.lower() else None,
                max_model_len=8192,
                gpu_memory_utilization=0.85,
                enable_prefix_caching=True,
                trust_remote_code=True,
                dtype="half",
            )

            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=1024,
                stop=["</s>", "<|endoftext|>"],
            )

            # Warmup
            logger.info("warming_up_summarization_model")
            dummy_prompt = "Summarize: Apple released a new product."
            self.llm.generate([dummy_prompt], self.sampling_params)

            self._initialized = True
            logger.info("summarization_model_loaded", model=self.model_id)

        except Exception as e:
            logger.error("summarization_model_load_failed", error=str(e))
            raise

    def _build_prompt(self, text: str, style: str) -> str:
        """Build prompt for given style."""
        template = STYLE_PROMPTS.get(style, STYLE_PROMPTS["executive"])
        return template.format(text=text)

    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4 + 1

    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=100)
    async def summarize(self, requests: List[SummarizationRequest]) -> List[SummarizationResponse]:
        """Generate summaries for a batch of documents.

        Args:
            requests: List of SummarizationRequest (batched by BentoML).

        Returns:
            List of SummarizationResponse with generated summaries.
        """
        start_time = time.time()

        await self._load_model()

        if not self.llm or not self.sampling_params:
            raise RuntimeError("Summarization model not initialized")

        all_responses: List[SummarizationResponse] = [None] * len(requests)

        # Group requests by style for efficient processing
        style_groups: Dict[str, List[int]] = {}
        for idx, req in enumerate(requests):
            if req.style not in style_groups:
                style_groups[req.style] = []
            style_groups[req.style].append(idx)

        for style, req_indices in style_groups.items():
            template = STYLE_PROMPTS.get(style, STYLE_PROMPTS["executive"])

            # Build all prompts for this style
            prompts = []
            text_token_counts = []
            for idx in req_indices:
                req = requests[idx]
                for text in req.texts:
                    prompt = template.format(text=text)
                    prompts.append(prompt)
                    text_token_counts.append(self._count_tokens(text))

            # Generate summaries
            try:
                generate_start = time.time()
                outputs = self.llm.generate(prompts, self.sampling_params)
                generate_duration_ms = (time.time() - generate_start) * 1000
            except Exception as e:
                logger.error("vllm_summarize_failed", error=str(e), style=style)
                # Return error summaries
                for idx in req_indices:
                    all_responses[idx] = SummarizationResponse(
                        summaries=["Error generating summary"] * len(requests[idx].texts),
                        model=self.model_id,
                        batch_size=len(requests[idx].texts),
                        total_tokens=0,
                    )
                continue

            # Parse outputs and group by request
            output_idx = 0
            for req_idx in req_indices:
                req = requests[req_idx]
                req_summaries = []
                req_tokens = 0

                for _ in req.texts:
                    output = outputs[output_idx]
                    output_idx += 1
                    summary = output.outputs[0].text.strip()
                    req_summaries.append(summary)
                    req_tokens += self._count_tokens(summary)

                all_responses[req_idx] = SummarizationResponse(
                    summaries=req_summaries,
                    model=self.model_id,
                    batch_size=len(req.texts),
                    total_tokens=req_tokens,
                )

        # Metrics
        total_duration_ms = (time.time() - start_time) * 1000
        self._request_count += 1
        self._total_duration_ms += total_duration_ms

        return all_responses

    @bentoml.api(route="/metrics")
    async def metrics(self) -> dict[str, str]:
        """Prometheus-format metrics endpoint."""
        avg_duration = self._total_duration_ms / max(self._request_count, 1)

        return {
            "metrics": f"""# HELP summarization_requests_total Total summarization requests
# TYPE summarization_requests_total counter
summarization_requests_total {self._request_count}
# HELP summarization_duration_seconds Average duration in seconds
# TYPE summarization_duration_seconds gauge
summarization_duration_seconds {avg_duration / 1000}
"""
        }

    @bentoml.api(route="/health")
    async def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self._initialized else "loading",
            "model": self.model_id,
            "backend": "vllm",
            "max_model_len": 8192,
            "initialized": self._initialized,
        }
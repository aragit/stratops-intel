"""Fallback Service — Lightweight Qwen2.5-3B-GGUF + vLLM.

Lightweight fallback for when primary services are overloaded.
Provides basic summarization, extraction, and classification.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import bentoml
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class FallbackRequest(BaseModel):
    """Request for fallback processing.

    Attributes:
        text: Input text to process
        task_type: Type of task (summarize, extract, classify)
        tenant_id: Tenant identifier
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=4096, description="Text to process")
    task_type: str = Field(..., description="Task type: summarize, extract, classify")
    tenant_id: str = Field(..., description="Tenant identifier")


class FallbackResponse(BaseModel):
    """Response from fallback processing.

    Attributes:
        result: Generated result text
        task_type: Task type that was processed
        model: Model identifier used
    """

    model_config = ConfigDict(extra="forbid")

    result: str = Field(..., description="Generated result")
    task_type: str = Field(..., description="Task type")
    model: str = Field(..., description="Model identifier")


TASK_PROMPTS = {
    "summarize": (
        "Summarize the following text in 2-3 sentences. "
        "Focus on key facts and actionable insights:\n\n{text}\n\nSummary:"
    ),
    "extract": (
        "Extract key entities (companies, people, products, metrics) from the following text. "
        "Return as a JSON list of objects with 'type' and 'value' fields:\n\n{text}\n\nEntities:"
    ),
    "classify": (
        "Classify the sentiment of the following text as positive, negative, or neutral. "
        "Return only the classification:\n\n{text}\n\nClassification:"
    ),
}


@bentoml.service(
    name="fallback-service",
    resources={"gpu": 1, "memory": "8Gi"},
    traffic={"timeout": 30, "concurrency": 32},
    batching={"max_batch_size": 64, "max_latency_ms": 30, "batch_dim": 0},
)
class FallbackService:
    """BentoML fallback service using Qwen2.5-3B with vLLM.

    Model: Qwen/Qwen2.5-3B-Instruct-GGUF (or AWQ)
    Max context: 4096 tokens
    Fast, lightweight fallback for when primary services are overloaded
    Supports: summarize, extract, classify tasks
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("FALLBACK_MODEL", "Qwen/Qwen2.5-3B-Instruct-GGUF")
        self.llm: Optional[Any] = None
        self.sampling_params: Optional[Any] = None
        self._initialized = False
        self._request_count = 0
        self._total_duration_ms = 0.0

    async def _load_model(self) -> None:
        """Lazy-load vLLM model on first request."""
        if self._initialized:
            return

        logger.info("loading_fallback_model", model=self.model_id)

        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=self.model_id,
                quantization="gguf" if "gguf" in self.model_id.lower() else "awq",
                max_model_len=4096,
                gpu_memory_utilization=0.60,
                enable_prefix_caching=True,
                trust_remote_code=True,
                dtype="half",
            )

            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=512,
                stop=["</s>", "<|endoftext|>"],
            )

            # Warmup
            logger.info("warming_up_fallback_model")
            dummy_prompt = "Summarize: Apple announced new product."
            self.llm.generate([dummy_prompt], self.sampling_params)

            self._initialized = True
            logger.info("fallback_model_loaded", model=self.model_id)

        except Exception as e:
            logger.error("fallback_model_load_failed", error=str(e))
            raise

    def _build_prompt(self, text: str, task_type: str) -> str:
        """Build prompt for given task type."""
        template = TASK_PROMPTS.get(task_type, TASK_PROMPTS["summarize"])
        return template.format(text=text)

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=30)
    async def process(self, requests: List[FallbackRequest]) -> List[FallbackResponse]:
        """Process fallback requests.

        Args:
            requests: List of FallbackRequest objects

        Returns:
            List of FallbackResponse objects
        """
        start_time = time.time()

        await self._load_model()

        if not self.llm or not self.sampling_params:
            raise RuntimeError("Fallback model not initialized")

        # Group requests by task type for efficiency
        task_groups: Dict[str, List[int]] = {}
        for idx, req in enumerate(requests):
            if req.task_type not in task_groups:
                task_groups[req.task_type] = []
            task_groups[req.task_type].append(idx)

        all_responses: List[FallbackResponse] = [None] * len(requests)

        for task_type, req_indices in task_groups.items():
            template = TASK_PROMPTS.get(task_type, TASK_PROMPTS["summarize"])

            # Build prompts for this task type
            prompts = []
            for idx in req_indices:
                req = requests[idx]
                prompt = template.format(text=req.text)
                prompts.append(prompt)

            # Generate responses
            try:
                outputs = self.llm.generate(prompts, self.sampling_params)
            except Exception as e:
                logger.error("fallback_generate_failed", error=str(e), task_type=task_type)
                # Return error responses
                for idx in req_indices:
                    all_responses[idx] = FallbackResponse(
                        result="Error generating response",
                        task_type=task_type,
                        model=self.model_id,
                    )
                continue

            # Parse outputs and assign to requests
            output_idx = 0
            for req_idx in req_indices:
                output = outputs[output_idx]
                output_idx += 1
                result = output.outputs[0].text.strip()

                all_responses[req_idx] = FallbackResponse(
                    result=result,
                    task_type=task_type,
                    model=self.model_id,
                )

        # Metrics
        total_duration_ms = (time.time() - start_time) * 1000
        self._request_count += 1
        self._total_duration_ms += total_duration_ms

        return all_responses

    @bentoml.api(route="/metrics")
    async def metrics(self) -> Dict[str, str]:
        """Prometheus-format metrics endpoint."""
        avg_duration = self._total_duration_ms / max(self._request_count, 1)

        return {
            "metrics": f"""# HELP fallback_requests_total Total fallback requests
# TYPE fallback_requests_total counter
fallback_requests_total {self._request_count}
# HELP fallback_duration_seconds Average duration in seconds
# TYPE fallback_duration_seconds gauge
fallback_duration_seconds {avg_duration / 1000}
"""
        }

    @bentoml.api(route="/health")
    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self._initialized else "loading",
            "model": self.model_id,
            "backend": "vllm",
            "max_model_len": 4096,
            "initialized": self._initialized,
        }
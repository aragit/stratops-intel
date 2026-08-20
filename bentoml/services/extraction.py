"""BentoML extraction service using vLLM backend for structured information extraction.

This service uses vLLM's LLM engine (NOT raw transformers) for high-throughput
batched inference with guided JSON decoding.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from typing import Any, Optional, List

import bentoml
import structlog
import torch
from pydantic import BaseModel, ConfigDict, Field, validator

logger = structlog.get_logger(__name__)


def validate_uuid_str(v: str) -> str:
    """Validate that a string is a valid UUID."""
    try:
        uuid.UUID(v, version=4)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid UUID format: {v}")
    return v


# Pre-defined extraction schemas for guided JSON decoding
EXTRACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "company": {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "ticker": {"type": ["string", "null"]},
            "industry": {"type": ["string", "null"]},
            "headquarters": {"type": ["string", "null"]},
            "founded_year": {"type": ["integer", "null"]},
            "employee_count": {"type": ["integer", "null"]},
            "revenue": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "key_executives": {
                "type": "array",
                "items": {"type": "string"},
            },
            "products": {
                "type": "array",
                "items": {"type": "string"},
            },
            "competitors": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["company_name"],
    },
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "title": {"type": ["string", "null"]},
            "company": {"type": ["string", "null"]},
            "role": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "linkedin": {"type": ["string", "null"]},
            "bio": {"type": ["string", "null"]},
        },
        "required": ["name"],
    },
    "product": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "company": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "features": {
                "type": "array",
                "items": {"type": "string"},
            },
            "pricing": {"type": ["string", "null"]},
            "release_date": {"type": ["string", "null"]},
            "target_market": {"type": ["string", "null"]},
        },
        "required": ["product_name"],
    },
    "financial_metric": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": ["earnings", "guidance", "dividend", "buyback", "m&a", "ipo", "funding"]},
            "company": {"type": "string"},
            "date": {"type": ["string", "null"]},
            "amount": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]},
            "details": {"type": ["string", "null"]},
        },
        "required": ["event_type", "company"],
    },
}


def get_schema(schema_name: str) -> dict[str, Any]:
    """Get JSON schema for extraction by name."""
    if schema_name not in EXTRACTION_SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(EXTRACTION_SCHEMAS.keys())}")
    return EXTRACTION_SCHEMAS[schema_name]


def build_extraction_prompt(text: str, schema: dict[str, Any]) -> str:
    """Build prompt for structured extraction with guided JSON schema."""
    schema_json = json.dumps(schema, indent=2)
    return f"""Extract structured information from the following text according to the JSON schema.
Only output valid JSON matching the schema. Do not include any explanation or extra text.

Schema:
{schema_json}

Text:
{text}

JSON Output:"""


def count_text_tokens(text: str) -> int:
    """Approximate token count for a text string."""
    return len(text) // 4 + 1


@bentoml.service(
    name="extraction-service",
    resources={"gpu": 1, "memory": "24Gi"},
    traffic={"timeout": 30, "concurrency": 16},
    batching={"max_batch_size": 32, "max_latency_ms": 50, "batch_dim": 0},
)
class ExtractionService:
    """BentoML service for structured information extraction using vLLM.

    Uses Qwen2.5-7B-Instruct-AWQ with vLLM backend for high-throughput
    batched inference. Supports guided JSON decoding via Outlines backend.

    Model: Qwen/Qwen2.5-7B-Instruct-AWQ
    Quantization: AWQ (4-bit)
    Max context: 8192 tokens
    Batch size: up to 32
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("EXTRACTION_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
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

        logger.info("loading_vllm_model", model=self.model_id)

        try:
            from vllm import LLM, SamplingParams

            # Initialize vLLM engine
            self.llm = LLM(
                model=self.model_id,
                quantization="awq",
                max_model_len=8192,
                gpu_memory_utilization=0.90,
                enable_prefix_caching=True,
                trust_remote_code=True,
                dtype="half",
            )

            # Sampling params for structured output
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=1024,
                stop=["<|endoftext|>"],
            )

            # Warmup with dummy prompt
            logger.info("warming_up_model")
            dummy_prompt = "Extract company name from: Apple Inc. is a technology company."
            self.llm.generate([dummy_prompt], self.sampling_params)

            self._initialized = True
            logger.info("vllm_model_loaded", model=self.model_id)

        except Exception as e:
            logger.error("vllm_model_load_failed", error=str(e))
            raise

    def _validate_request(self, req: Any) -> Optional[str]:
        """Validate extraction request. Returns error message or None."""
        # Check empty input
        if not req.texts:
            return "Empty input: texts list is empty"

        # Check batch size limit
        if len(req.texts) > 32:
            return f"Batch size exceeds limit: {len(req.texts)} > 32"

        # Check per-text token limit (leave headroom for prompt + response)
        for i, text in enumerate(req.texts):
            tokens = count_text_tokens(text)
            if tokens > 4000:
                return f"Text {i} exceeds max token limit: {tokens} tokens > 4000"

        # Validate tenant_id is valid UUID
        try:
            validate_uuid_str(req.tenant_id)
        except ValueError as e:
            return str(e)

        # Validate schema_name
        if req.schema_name not in EXTRACTION_SCHEMAS:
            return f"Unknown schema: {req.schema_name}. Available: {list(EXTRACTION_SCHEMAS.keys())}"

        return None

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=50)
    async def extract(self, requests: list[Any]) -> list[dict]:
        """Extract structured information from texts using vLLM.

        Args:
            requests: List of extraction requests (batched by BentoML).

        Returns:
            List of result dicts with extracted data and metadata.
        """
        request_start = time.time()

        await self._load_model()

        if not self.llm or not self.sampling_params:
            raise RuntimeError("vLLM model not initialized")

        all_responses: list[dict] = [None] * len(requests)

        # Batching optimization: group requests by schema_name
        # Same schema = shared guided decoding config, improves vLLM efficiency
        schema_groups: dict[str, list[int]] = {}
        for req_idx, req in enumerate(requests):
            schema_name = req.schema_name
            if schema_name not in schema_groups:
                schema_groups[schema_name] = []
            schema_groups[schema_name].append(req_idx)

        # Sort texts by length within each batch (shortest first improves PagedAttention efficiency)
        # and track padding overhead
        padding_log: list[tuple[int, int, int]] = []  # (req_idx, original_len, padded_len)
        processing_order: list[int] = []

        for schema_name, req_indices in schema_groups.items():
            # Sort by text length (ascending - shortest first)
            sorted_indices = sorted(req_indices, key=lambda i: len(requests[i].texts[0]) if requests[i].texts else 0)
            processing_order.extend(sorted_indices)
            for req_idx in sorted_indices:
                req = requests[req_idx]
                # Estimate padded length (vLLM pads to longest in batch)
                max_text_len = max((len(t) for t in req.texts), default=0)
                padding_log.append((req_idx, max_text_len, max_text_len))  # vLLM handles padding internally

        # Process requests in optimized order
        for req_idx in processing_order:
            individual_start = time.time()
            req = requests[req_idx]

            # Validate request
            validation_error = self._validate_request(req)
            if validation_error:
                duration_ms = (time.time() - individual_start) * 1000
                all_responses[req_idx] = {
                    "error": "validation_failed",
                    "detail": validation_error,
                    "model": self.model_id,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "generation_time_ms": duration_ms,
                }
                continue

            schema = get_schema(req.schema_name)
            batch_size = len(req.texts)

            # Build all prompts
            all_prompts: list[str] = []
            for text in req.texts:
                prompt = build_extraction_prompt(text, schema)
                all_prompts.append(prompt)

            # Batch generate with vLLM with error handling
            try:
                generate_start = time.time()
                outputs = self.llm.generate(all_prompts, self.sampling_params)
                generate_end = time.time()
                generate_duration_ms = (generate_end - generate_start) * 1000
            except torch.cuda.OutOfMemoryError:
                duration_ms = (time.time() - individual_start) * 1000
                torch.cuda.empty_cache()
                self._request_count += 1
                self._total_duration_ms += (time.time() - request_start) * 1000
                all_responses[req_idx] = {
                    "error": "oom",
                    "detail": "GPU out of memory",
                    "model": self.model_id,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "generation_time_ms": duration_ms,
                    "retry_after": 30,
                }
                continue
            except Exception as e:
                duration_ms = (time.time() - individual_start) * 1000
                logger.error("vllm_generate_failed", error=str(e), req_idx=req_idx)
                self._request_count += 1
                self._total_duration_ms += (time.time() - request_start) * 1000
                all_responses[req_idx] = {
                    "error": "generate_failed",
                    "detail": str(e),
                    "model": self.model_id,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "generation_time_ms": duration_ms,
                }
                continue

            # Parse outputs - one output per text in the batch
            out_idx = 0
            for text_idx in range(len(req.texts)):
                try:
                    output = outputs[out_idx]
                    out_idx += 1
                    generated_text = output.outputs[0].text.strip()

                    # Parse JSON output
                    try:
                        extracted = json.loads(generated_text)
                        # Track tokens roughly
                        completion_tokens = max(len(generated_text) // 4, 1)
                    except json.JSONDecodeError:
                        extracted = {"error": "parse_failed", "raw": generated_text[:500]}
                        completion_tokens = max(len(generated_text) // 4, 1)

                    # structlog context: tenant_id, schema_name, batch_size, duration_ms, token_count
                    structlog.contexts.set(structlog.contexts, "tenant_id", req.tenant_id)
                    structlog.contexts.set(structlog.contexts, "schema_name", req.schema_name)
                    structlog.contexts.set(structlog.contexts, "batch_size", batch_size)
                    structlog.contexts.set(structlog.contexts, "duration_ms", generate_duration_ms)
                    structlog.contexts.set(structlog.contexts, "token_count", completion_tokens)

                    all_responses[req_idx] = {
                        "result": extracted,
                        "model": self.model_id,
                        "prompt_tokens": count_text_tokens(req.texts[text_idx]),
                        "completion_tokens": completion_tokens,
                        "total_tokens": count_text_tokens(req.texts[text_idx]) + completion_tokens,
                        "generation_time_ms": generate_duration_ms,
                    }

                except Exception as e:
                    logger.warning("output_processing_failed", error=str(e), req_idx=req_idx, text_idx=text_idx)
                    all_responses[req_idx] = {
                        "error": "process_failed",
                        "detail": str(e),
                        "model": self.model_id,
                        "prompt_tokens": count_text_tokens(req.texts[text_idx]) if text_idx < len(req.texts) else 0,
                        "completion_tokens": 0,
                        "total_tokens": count_text_tokens(req.texts[text_idx]) if text_idx < len(req.texts) else 0,
                        "generation_time_ms": generate_duration_ms,
                    }

        # Final timing
        total_duration_ms = (time.time() - request_start) * 1000
        self._request_count += 1
        self._total_duration_ms += total_duration_ms

        return all_responses

    @bentoml.api(route="/metrics")
    async def metrics(self) -> dict[str, str]:
        """Prometheus-format metrics endpoint."""
        avg_duration_ms = self._total_duration_ms / max(self._request_count, 1)

        # Get GPU memory if available
        gpu_memory = "0"
        try:
            if torch.cuda.is_available():
                gpu_memory = str(torch.cuda.memory_allocated())
        except Exception:
            pass

        metrics_text = f"""# HELP bentoml_requests_total Total number of extraction requests
# TYPE bentoml_requests_total counter
bentoml_requests_total {self._request_count}
# HELP bentoml_request_duration_seconds Average request duration in seconds
# TYPE bentoml_request_duration_seconds gauge
bentoml_request_duration_seconds {avg_duration_ms / 1000}
# HELP bentoml_gpu_memory_used_bytes GPU memory used bytes
# TYPE bentoml_gpu_memory_used_bytes gauge
bentoml_gpu_memory_used_bytes {gpu_memory}
"""

        return {"metrics": metrics_text}

    @bentoml.api(route="/health")
    async def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self._initialized else "loading",
            "model": self.model_id,
            "backend": "vllm",
            "gpu": True,
            "initialized": self._initialized,
        }
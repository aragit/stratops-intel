"""BentoML extraction service using vLLM backend for structured information extraction.

This service uses vLLM's LLM engine (NOT raw transformers) for high-throughput
batched inference with guided JSON decoding.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import bentoml
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class ExtractionRequest(BaseModel):
    """Request for structured extraction from text.

    Attributes:
        texts: List of text chunks to extract from.
        schema_name: Name of the extraction schema (e.g., "company", "person", "product").
        tenant_id: Tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(..., min_length=1, max_length=32, description="Texts to extract from")
    schema_name: str = Field(..., description="Extraction schema name")
    tenant_id: str = Field(..., description="Tenant identifier")


class ExtractionResponse(BaseModel):
    """Response containing extracted structured data.

    Attributes:
        results: List of extracted objects matching the schema.
        model: Model identifier used for extraction.
        batch_size: Number of inputs processed.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]] = Field(..., description="Extracted structured data")
    model: str = Field(..., description="Model identifier")
    batch_size: int = Field(..., description="Number of inputs in batch")


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
    "financial_event": {
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


@bentoml.service(
    name="extraction-service",
    resources={"gpu": 1},
    traffic={"timeout": 300},
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
                stop=["</s>", "<|endoftext|>"],
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

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=50)
    async def extract(self, requests: list[ExtractionRequest]) -> list[ExtractionResponse]:
        """Extract structured information from texts using vLLM.

        Args:
            requests: List of ExtractionRequest (batched by BentoML).

        Returns:
            List of ExtractionResponse with extracted structured data.
        """
        await self._load_model()

        if not self.llm or not self.sampling_params:
            raise RuntimeError("vLLM model not initialized")

        responses: list[ExtractionResponse] = []

        # Build all prompts
        all_prompts: list[str] = []
        request_indices: list[int] = []  # Track which request each prompt belongs to

        for req_idx, req in enumerate(requests):
            schema = get_schema(req.schema_name)
            for text in req.texts:
                prompt = build_extraction_prompt(text, schema)
                all_prompts.append(prompt)
                request_indices.append(req_idx)

        # Batch generate with vLLM
        try:
            outputs = self.llm.generate(all_prompts, self.sampling_params)
        except Exception as e:
            logger.error("vllm_generate_failed", error=str(e))
            raise

        # Parse outputs and group by request
        results_by_request: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(requests))}

        for output, req_idx in zip(outputs, request_indices):
            generated_text = output.outputs[0].text.strip()

            # Parse JSON output
            try:
                extracted = json.loads(generated_text)
                results_by_request[req_idx].append(extracted)
            except json.JSONDecodeError as e:
                logger.warning("json_parse_failed", output=generated_text[:200], error=str(e))
                results_by_request[req_idx].append({"error": "JSON parse failed", "raw": generated_text[:500]})

        # Build responses
        for req_idx, req in enumerate(requests):
            responses.append(ExtractionResponse(
                results=results_by_request[req_idx],
                model=self.model_id,
                batch_size=len(req.texts),
            ))

        return responses

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
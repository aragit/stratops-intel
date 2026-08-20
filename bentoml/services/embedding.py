"""Embedding service in BentoML.

Uses sentence-transformers (NOT vLLM — embedding models don't need LLM engine)
for text embeddings with 1024 dimensions.

Constraints:
- Shares GPU with extraction service (gpu: 0.5) or runs on CPU
- Batch processing with max_batch_size=128
- Normalized embeddings (magnitude ≈ 1.0)
"""

from __future__ import annotations

import bentoml
import os
import numpy as np
from typing import Any, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field

import numpy as np

logger = structlog.get_logger(__name__)


class EmbeddingRequest(BaseModel):
    """Request for text embeddings.

    Attributes:
        texts: List of text strings to embed.
        tenant_id: Tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(extra="forbid")

    texts: List[str] = Field(..., min_length=1, description="Texts to embed")
    tenant_id: str = Field(..., description="Tenant identifier")


class EmbeddingResponse(BaseModel):
    """Response containing text embeddings.

    Attributes:
        embeddings: List of embedding vectors (1024-dim each).
        model: Model identifier used for embedding.
        batch_size: Number of texts processed in the batch.
    """

    model_config = ConfigDict(extra="forbid")

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model identifier")
    batch_size: int = Field(..., description="Number of texts in batch")


class EmbeddingService:
    """BentoML service for text embeddings using sentence-transformers.

    Model: BAAI/bge-large-en-v1.5 (1024-dimensional embeddings)
    Uses sentence-transformers (NOT vLLM — embedding models don't need LLM engine).
    Shares GPU with extraction service if needed, otherwise CPU.

    Embedding properties:
    - 1024-dimensional vectors
    - L2-normalized (magnitude ≈ 1.0)
    - Consistent across runs
    """

    def __init__(self) -> None:
        self.model_id = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self._model: Optional[Any] = None
        self._device = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cpu" or "cuda:0"
        self._gpu_share = float(os.getenv("EMBEDDING_GPU_SHARE", "0.5"))
        self._initialized = False

    async def _load_model(self) -> None:
        """Lazy-load sentence-transformers model on first request."""
        if self._initialized:
            return

        logger.info("loading_embedding_model", model=self.model_id, device=self._device)

        try:
            from sentence_transformers import SentenceTransformer

            # Load the model - will use GPU if available, otherwise CPU
            self._model = SentenceTransformer(self.model_id, device=self._device)

            self._initialized = True
            logger.info("embedding_model_loaded", model=self.model_id, device=self._device)

        except Exception as e:
            logger.error("embedding_model_load_failed", error=str(e))
            raise

    @bentoml.api(batchable=True, max_batch_size=128, max_latency_ms=20)
    async def embed(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """Generate embeddings for a batch of texts.

        Args:
            requests: List of EmbeddingRequest objects.

        Returns:
            List of EmbeddingResponse objects with embeddings.
        """
        await self._load_model()

        if not self._model:
            raise RuntimeError("Embedding model not initialized")

        start_time = time.time()

        # Concatenate all texts from all requests
        all_texts: List[str] = []
        request_text_counts: List[int] = []

        for req in requests:
            count = len(req.texts) if req.texts else 0
            request_text_counts.append(count)
            all_texts.extend(req.texts or [])

        if not all_texts:
            # Return empty responses for each request
            return [
                EmbeddingResponse(
                    embeddings=[],
                    model=self.model_id,
                    batch_size=len(request_text_counts) if request_text_counts else 0,
                )
                for _ in requests
            ]

        # Encode all texts at once using the model
        embeddings_array = self._model.encode(
            all_texts,
            normalize_embeddings=True,
            convert_to_tensor=False,
        )

        # Convert to list of lists
        embeddings_list = embeddings_array.tolist()

        # Split results back per request
        result_embeddings: List[List[float]] = []
        start_idx = 0
        for count in request_text_counts:
            end_idx = start_idx + count
            result_embeddings.append(embeddings_list[start_idx:end_idx])
            start_idx = end_idx

        total_duration_ms = (time.time() - start_time) * 1000

        # Verify embedding dimensionality
        for i, emb in enumerate(result_embeddings):
            if emb:
                magnitude = np.linalg.norm(emb)
                logger.debug(
                    "embedding_magnitude_check",
                    index=i,
                    magnitude=round(magnitude, 4),
                    expected_approx=1.0,
                )

        # Return one response per original request
        responses: List[EmbeddingResponse] = []
        start_idx = 0
        for i, count in enumerate(request_text_counts):
            response = EmbeddingResponse(
                embeddings=result_embeddings[i],
                model=self.model_id,
                batch_size=count,
            )
            responses.append(response)

        logger.info(
            "embedding_generated",
            total_texts=len(all_texts),
            batch_size=sum(request_text_counts),
            duration_ms=round(total_duration_ms, 2),
            model=self.model_id,
        )

        return responses

    @bentoml.api(route="/health")
    async def health(self) -> dict:
        """Health check endpoint.

        Returns:
            Model status, GPU/CPU mode, and initialization status.
        """
        await self._load_model()

        return {
            "status": "healthy" if self._initialized else "loading",
            "model": self.model_id,
            "device": self._device,
            "gpu_share": self._gpu_share,
            "initialized": self._initialized,
            "embedding_dim": 1024,
        }


# Global service instance for BentoML
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
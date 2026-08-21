"""Tenant-scoped semantic execution cache using Redis + sentence-transformers.

Provides similarity-based caching ahead of LiteLLM/vLLM calls, keyed by
tenant_id + prompt hash with cosine-similarity verification. Enables
repeated prompts (e.g. recurring financial entity extractions) to return
cached responses without re-invoking the LLM.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


#: Default TTL for cached entries (24 hours)
DEFAULT_TTL = 86400

#: Cosine similarity threshold for considering two embeddings a match
SIMILARITY_THRESHOLD = 0.92

#: Number of least-significant bits to use for prompt hash (64-bit)
HASH_BITS = 64


class SemanticCache:
    """Tenant-scoped semantic cache using Redis + sentence-transformers embeddings.

    Stores prompt embeddings in Redis under keys patterned as:
        cache:semantic:{tenant_id}:{hash}

    On `get()`, the incoming prompt is embedded and cosine-similarity-checked
    against the stored embedding. If similarity >= threshold, the cached
    response is returned; otherwise None is returned and the caller should
    invoke the LLM.

    On `set()`, the prompt + response are embedded and stored with a TTL.
    """

    def __init__(self, redis: Any, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the semantic cache.

        Args:
            redis: An async Redis client (redis.asyncio.Redis).
            model_name: Sentence-Transformer model name for embedding prompts.
        """
        self._redis = redis
        self._model_name = model_name
        self._model: Any | None = None  # lazily loaded

    @property
    def model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            logger.info("loading_semantic_cache_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _prompt_hash(self, prompt: str) -> str:
        """Deterministic 64-bit hash of a prompt string for Redis key construction."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:HASH_BITS]

    async def get(
        self, tenant_id: str, prompt: str, threshold: float = SIMILARITY_THRESHOLD
    ) -> dict[str, Any] | None:
        """Retrieve a cached response for a tenant+prompt if similar enough.

        Args:
            tenant_id: Tenant identifier for multi-tenant partitioning.
            prompt: The user prompt/query to check against the cache.
            threshold: Minimum cosine similarity to consider a match (default 0.92).

        Returns:
            The cached response dict if a match is found, otherwise None.
        """
        prompt_hash = self._prompt_hash(prompt)
        key = f"cache:semantic:{tenant_id}:{prompt_hash}"

        # Fetch stored embedding from Redis
        stored = await self._redis.get(key)
        if stored is None:
            logger.debug("cache_miss_no_entry", key=key, tenant_id=tenant_id)
            return None

        try:
            stored_data = json.loads(stored)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("cache_corrupt_stored_data", key=key, exc=exc)
            return None

        stored_embedding = stored_data.get("embedding")
        if stored_embedding is None:
            logger.warning("cache_missing_embedding", key=key)
            return None

        # Embed the incoming prompt and compute cosine similarity
        incoming_embedding = await self._ainvoke_model(prompt)

        similarity = self._cosine_similarity(incoming_embedding, stored_embedding)

        if similarity >= threshold:
            logger.debug(
                "cache_hit",
                tenant_id=tenant_id,
                similarity=round(similarity, 4),
                threshold=threshold,
            )
            response: dict[str, Any] | None = stored_data.get("response")
            return response

        logger.debug(
            "cache_similarity_too_low",
            tenant_id=tenant_id,
            similarity=round(similarity, 4),
            threshold=threshold,
        )
        return None

    async def set(
        self,
        tenant_id: str,
        prompt: str,
        response: dict[str, Any],
        ttl: int = DEFAULT_TTL,
    ) -> None:
        """Store a prompt-response pair in the semantic cache.

        Args:
            tenant_id: Tenant identifier for multi-tenant partitioning.
            prompt: The user prompt/query to cache.
            response: The LLM response dict to cache.
            ttl: Time-to-live in seconds (default 86400 / 24 hours).
        """
        prompt_hash = self._prompt_hash(prompt)
        key = f"cache:semantic:{tenant_id}:{prompt_hash}"

        embedding = await self._ainvoke_model(prompt)

        payload = json.dumps({"embedding": embedding, "response": response})

        await self._redis.set(key, payload, ex=ttl)
        logger.debug(
            "cache_set",
            tenant_id=tenant_id,
            key=key,
            ttl=ttl,
            embedding_dims=len(embedding),
        )

    async def _ainvoke_model(self, prompt: str) -> list[float]:
        """Run the sentence-transformer model on *prompt* in a thread executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.model.encode, prompt)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors: cos(theta) = a·b / (|a||b|)."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot / (norm_a * norm_b))


__all__ = ["SemanticCache", "DEFAULT_TTL", "SIMILARITY_THRESHOLD"]

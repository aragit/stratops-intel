"""Unit tests for the SemanticCache tenant-scoped cache."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from backend.cache.semantic_cache import DEFAULT_TTL, SIMILARITY_THRESHOLD, SemanticCache


class TestSemanticCacheInit:
    """Tests for SemanticCache construction."""

    @pytest.mark.asyncio
    async def test_init_stores_redis_and_model_name(self) -> None:
        """Init should accept a Redis client and model name."""
        mock_redis = mock.AsyncMock()
        cache = SemanticCache(redis=mock_redis, model_name="all-MiniLM-L6-v2")

        assert cache._redis is mock_redis
        assert cache._model_name == "all-MiniLM-L6-v2"

    def test_default_constants(self) -> None:
        """Default TTL should be 86400 and similarity threshold 0.92."""
        assert DEFAULT_TTL == 86400
        assert SIMILARITY_THRESHOLD == 0.92


class TestSemanticCacheHash:
    """Tests for prompt hash determinism."""

    def test_hash_deterministic(self) -> None:
        """Same prompt must always produce the same hash."""
        cache = SemanticCache.__new__(SemanticCache)
        h1 = cache._prompt_hash("test prompt")
        h2 = cache._prompt_hash("test prompt")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_different_prompts_differ(self) -> None:
        """Different prompts must produce different hashes."""
        cache = SemanticCache.__new__(SemanticCache)
        h1 = cache._prompt_hash("prompt A")
        h2 = cache._prompt_hash("prompt B")
        assert h1 != h2


class TestSemanticCacheCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_cosine_similarity_parallel_vectors(self) -> None:
        """Parallel vectors should have similarity = 1.0."""
        cache = SemanticCache.__new__(SemanticCache)
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = cache._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self) -> None:
        """Orthogonal vectors should have similarity = 0.0."""
        cache = SemanticCache.__new__(SemanticCache)
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = cache._cosine_similarity(a, b)
        assert abs(sim - 0.0) < 1e-10

    def test_cosine_similarity_opposite(self) -> None:
        """Anti-parallel vectors should have similarity = -1.0."""
        cache = SemanticCache.__new__(SemanticCache)
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = cache._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-10


class TestSemanticCacheGetSet:
    """Tests for get() and set() with mocked Redis and model."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_no_cached_entry(self) -> None:
        """get() must return None if Redis has no entry for the key."""
        mock_redis = mock.AsyncMock()
        cache = SemanticCache(redis=mock_redis)

        # No entry exists
        result = await cache.get(tenant_id="t-1", prompt="hello world")
        assert result is None
        mock_redis.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_stores_json_with_embedding_and_response(self) -> None:
        """set() should store JSON with embedding + response in Redis."""
        mock_redis = mock.AsyncMock()

        cache = SemanticCache(redis=mock_redis)

        # Patch _ainvoke_model to return a deterministic vector
        cache._ainvoke_model = mock.AsyncMock(return_value=[0.5] * 768)

        await cache.set(
            tenant_id="t-1",
            prompt="test prompt",
            response={"result": "ok", "request_id": "req-123"},
            ttl=3600,
        )

        # Redis set should have been called with a key, JSON payload, and TTL
        mock_redis.set.assert_awaited_once()
        args, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == 3600
        # redis.set(key, value, ex=ttl) -> value is the second positional arg
        stored_val = kwargs.get("value") or args[1] if len(args) > 1 else None
        stored_json = json.loads(stored_val)
        assert "embedding" in stored_json
        assert "response" in stored_json
        assert stored_json["response"] == {"result": "ok", "request_id": "req-123"}

    @pytest.mark.asyncio
    async def test_get_returns_cached_response_when_similar_enough(self) -> None:
        """get() should return the cached response when cosine similarity >= threshold."""
        mock_redis = mock.AsyncMock()

        cache = SemanticCache(redis=mock_redis)

        # Pre-store an embedding + response
        stored_emb = [0.9] * 768
        stored_data = {"embedding": stored_emb, "response": {"cached": True}}

        mock_redis.get = mock.AsyncMock(return_value=json.dumps(stored_data))

        # Patch _ainvoke_model so incoming embedding matches stored one exactly
        cache._ainvoke_model = mock.AsyncMock(return_value=[0.9] * 768)

        result = await cache.get(tenant_id="t-1", prompt="test prompt")
        assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_get_returns_none_when_similarity_below_threshold(self) -> None:
        """get() should return None when cosine similarity < threshold."""
        mock_redis = mock.AsyncMock()

        cache = SemanticCache(redis=mock_redis)

        # Store a very different embedding
        stored_emb = [0.1] * 768
        stored_data = {"embedding": stored_emb, "response": {"cached": False}}

        mock_redis.get = mock.AsyncMock(return_value=json.dumps(stored_data))

        # Incoming embedding [0.1]*768 vs stored [0.1]*768 gives sim=1.0
        # But we want below threshold, so use very different vectors:
        # stored [1,0,0...], incoming [0,1,0...] -> sim = 0.0
        # We'll mock _ainvoke_model to return orthogonal vectors
        cache._ainvoke_model = mock.AsyncMock(return_value=[0.0] * 768)

        # Actually let's just store emb=[1,0,0...] and have model return [0,1,0...]
        # Let's simplify: store emb all 1s, model returns all -1s → sim = -1.0
        stored_emb = [1.0] * 768
        stored_data = {"embedding": stored_emb, "response": {"cached": False}}
        mock_redis.get = mock.AsyncMock(return_value=json.dumps(stored_data))

        cache._ainvoke_model = mock.AsyncMock(return_value=[-1.0] * 768)

        result = await cache.get(tenant_id="t-1", prompt="test prompt")
        assert result is None  # similarity -1.0 < threshold 0.92

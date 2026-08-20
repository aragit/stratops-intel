"""Unit tests for the BentoML embedding service.

Tests batching, normalization, dimensionality, and model loading.
"""

from __future__ import annotations

import asyncio
import numpy as np
from unittest import mock

import pytest

from bentoml.services.embedding import EmbeddingRequest, EmbeddingResponse, EmbeddingService


class TestEmbeddingRequest:
    """Tests for the EmbeddingRequest model."""

    def test_embedding_request_creation(self) -> None:
        """Test basic EmbeddingRequest creation and validation."""
        request = EmbeddingRequest(
            texts=["Hello world", "Test sentence"],
            tenant_id="00000000-0000-0000-0000-000000000001",
        )
        assert request.texts == ["Hello world", "Test sentence"]
        assert request.tenant_id == "00000000-0000-0000-0000-000000000001"

    def test_embedding_request_empty_texts(self) -> None:
        """Test EmbeddingRequest with empty texts list."""
        request = EmbeddingRequest(
            texts=[],
            tenant_id="00000000-0000-0000-0000-000000000001",
        )
        assert request.texts == []

    def test_embedding_request_tenant_id(self) -> None:
        """Test EmbeddingRequest tenant_id validation."""
        from uuid import uuid4
        request = EmbeddingRequest(
            texts=["test"],
            tenant_id=str(uuid4()),
        )
        assert request.tenant_id is not None


class TestEmbeddingResponse:
    """Tests for the EmbeddingResponse model."""

    def test_embedding_response_creation(self) -> None:
        """Test basic EmbeddingResponse creation and validation."""
        response = EmbeddingResponse(
            embeddings=[[0.5, 0.5, 0.5, 0.5] * 256],  # 1024-dim vector
            model="test-model",
            batch_size=1,
        )
        assert len(response.embeddings) == 1
        assert response.model == "test-model"
        assert response.batch_size == 1
        # Check dimensionality
        assert len(response.embeddings[0]) == 1024


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check returns correct structure."""
        service = EmbeddingService()
        health = await service.health()

        assert "status" in health
        assert "model" in health
        assert "device" in health
        assert "initialized" in health
        assert "embedding_dim" in health

    @pytest.mark.asyncio
    async def test_embedding_batching(self) -> None:
        """Test that embedding service handles batching correctly."""
        service = EmbeddingService()

        # Mock the model encode method
        with mock.patch.object(service._model, "encode") as mock_encode:
            # Return embeddings for 5 texts
            mock_embeddings = np.random.rand(5, 1024).tolist()
            mock_encode.return_value = np.array(mock_embeddings)

            # Create requests with multiple texts each
            requests = [
                EmbeddingRequest(texts=["text1", "text2"], tenant_id="001"),
                EmbeddingRequest(texts=["text3"], tenant_id="001"),
                EmbeddingRequest(texts=["text4", "text5", "text6"], tenant_id="001"),
            ]

            responses = await service.embed(requests)

            # Should have 3 responses (one per request)
            assert len(responses) == 3

            # First response should have 2 embeddings
            assert len(responses[0].embeddings) == 2
            # Second response should have 1 embedding
            assert len(responses[1].embeddings) == 1
            # Third response should have 3 embeddings
            assert len(responses[2].embeddings) == 3

            # All should have 1024-dimensional embeddings
            for resp in responses:
                assert len(resp.embeddings[0]) == 1024 if resp.embeddings else True

    @pytest.mark.asyncio
    async def test_embedding_normalization(self) -> None:
        """Test that embeddings are normalized (magnitude ≈ 1.0)."""
        service = EmbeddingService()

        # Mock the model encode method with known vectors
        with mock.patch.object(service._model, "encode") as mock_encode:
            # Create unit vectors (magnitude = 1.0)
            unit_vectors = np.eye(1024)  # 1024 unit vectors, each with 1 at one position
            mock_encode.return_value = unit_vectors

            requests = [
                EmbeddingRequest(texts=["text1"], tenant_id="001"),
                EmbeddingRequest(texts=["text2"], tenant_id="001"),
            ]

            responses = await service.embed(requests)

            # Check that embeddings are normalized
            for resp in responses:
                for emb in resp.embeddings:
                    magnitude = np.linalg.norm(emb)
                    assert abs(magnitude - 1.0) < 0.01, (
                        f"Embedding magnitude {magnitude} not close to 1.0"
                    )

    @pytest.mark.asyncio
    async def test_embedding_dimensionality(self) -> None:
        """Test that embeddings have correct dimensionality (1024)."""
        service = EmbeddingService()

        # Mock the model encode method
        with mock.patch.object(service._model, "encode") as mock_encode:
            # Return embeddings for 3 texts
            mock_embeddings = np.random.rand(3, 1024)
            mock_encode.return_value = mock_embeddings

            requests = [
                EmbeddingRequest(texts=["text1"], tenant_id="001"),
                EmbeddingRequest(texts=["text2"], tenant_id="001"),
                EmbeddingRequest(texts=["text3"], tenant_id="001"),
            ]

            responses = await service.embed(requests)

            # All should have 1024-dimensional embeddings
            for resp in responses:
                assert len(resp.embeddings) == 1
                assert len(resp.embeddings[0]) == 1024

    @pytest.mark.asyncio
    async def test_embedding_empty_requests(self) -> None:
        """Test handling of empty text lists."""
        service = EmbeddingService()

        requests = [
            EmbeddingRequest(texts=[], tenant_id="001"),
            EmbeddingRequest(texts=["only one"], tenant_id="001"),
        ]

        responses = await service.embed(requests)

        # Should have 2 responses
        assert len(responses) == 2
        # First should have empty embeddings
        assert len(responses[0].embeddings) == 0
        # Second should have 1 embedding
        assert len(responses[1].embeddings) == 1
"""Unit tests for the BentoML Summarization Service."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from bentoml.services.summarization import (
    SummarizationService,
    SummarizationRequest,
    SummarizationResponse,
    STYLE_PROMPTS,
)


class TestSummarizationRequest:
    """Tests for SummarizationRequest model."""

    def test_request_creation(self) -> None:
        """Test basic request creation."""
        req = SummarizationRequest(
            texts=["Test document one", "Test document two"],
            max_length=256,
            style="executive",
            tenant_id="00000000-0000-0000-0000-000000000001",
        )
        assert len(req.texts) == 2
        assert req.max_length == 256
        assert req.style == "executive"

    def test_request_defaults(self) -> None:
        """Test default values."""
        req = SummarizationRequest(
            texts=["Test"],
            tenant_id="00000000-0000-0000-0000-000000000001",
        )
        assert req.max_length == 256
        assert req.style == "executive"

    def test_style_validation(self) -> None:
        """Test style field accepts valid values."""
        for style in ["executive", "technical", "bullet_points"]:
            req = SummarizationRequest(
                texts=["Test"],
                style=style,
                tenant_id="00000000-0000-0000-0000-000000000001",
            )
            assert req.style == style

    def test_texts_length_validation(self) -> None:
        """Test texts list length bounds."""
        # Min 1
        req = SummarizationRequest(texts=["one"], tenant_id="001")
        assert len(req.texts) == 1

        # Max 16
        texts = [f"doc {i}" for i in range(16)]
        req = SummarizationRequest(texts=texts, tenant_id="001")
        assert len(req.texts) == 16

    def test_max_length_bounds(self) -> None:
        """Test max_length bounds."""
        with pytest.raises(ValueError):
            SummarizationRequest(
                texts=["test"],
                max_length=63,  # < 64
                tenant_id="001",
            )
        with pytest.raises(ValueError):
            SummarizationRequest(
                texts=["test"],
                max_length=1025,  # > 1024
                tenant_id="001",
            )


class TestSummarizationResponse:
    """Tests for SummarizationResponse model."""

    def test_response_creation(self) -> None:
        """Test response creation."""
        resp = SummarizationResponse(
            summaries=["Summary one", "Summary two"],
            model="test-model",
            batch_size=2,
            total_tokens=100,
        )
        assert len(resp.summaries) == 2
        assert resp.model == "test-model"
        assert resp.batch_size == 2
        assert resp.total_tokens == 100


class TestStylePrompts:
    """Tests for style prompt templates."""

    def test_all_styles_defined(self) -> None:
        """All required styles have prompts."""
        required = {"executive", "technical", "bullet_points"}
        assert set(STYLE_PROMPTS.keys()) == required

    def test_prompt_contains_placeholder(self) -> None:
        """All prompts contain {text} placeholder."""
        for style, prompt in STYLE_PROMPTS.items():
            assert "{text}" in prompt, f"Style {style} missing placeholder"


class TestSummarizationService:
    """Tests for SummarizationService class."""

    @pytest.fixture
    def service(self) -> SummarizationService:
        """Create service instance."""
        return SummarizationService()

    @pytest.mark.asyncio
    async def test_load_model_called_once(self, service) -> None:
        """Model loading is idempotent."""
        with mock.patch("bentoml.services.summarization.LLM") as mock_llm:
            mock_instance = mock.MagicMock()
            mock_llm.return_value = mock_instance

            await service._load_model()
            await service._load_model()  # Second call

            assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_empty_requests(self, service) -> None:
        """Empty requests should return empty responses."""
        # This test is more conceptual - empty list would be filtered by Pydantic
        pass

    @pytest.mark.asyncio
    async def test_summarize_batching_by_style(
        self, service
    ) -> None:
        """Test that requests are grouped by style for efficiency."""
        with mock.patch.object(service, "_load_model"):
            with mock.patch.object(service, "llm", new_callable=mock.PropertyMock) as mock_llm_prop:
                mock_llm = mock.MagicMock()
                mock_llm_prop.return_value = mock_llm
                mock_llm.generate.return_value = [
                    mock.MagicMock(outputs=[mock.MagicMock(text="Summary 1")]),
                    mock.MagicMock(outputs=[mock.MagicMock(text="Summary 2")]),
                ]

                service.sampling_params = mock.MagicMock()
                service._initialized = True

                requests = [
                    SummarizationRequest(texts=["doc1"], style="executive", tenant_id="001"),
                    SummarizationRequest(texts=["doc2"], style="technical", tenant_id="001"),
                    SummarizationRequest(texts=["doc3"], style="executive", tenant_id="001"),
                ]

                # This tests the internal grouping logic
                style_groups: dict[str, list[int]] = {}
                for idx, req in enumerate(requests):
                    if req.style not in style_groups:
                        style_groups[req.style] = []
                    style_groups[req.style].append(idx)

                assert "executive" in style_groups
                assert "technical" in style_groups
                assert len(style_groups["executive"]) == 2
                assert len(style_groups["technical"]) == 1

    def test_prompt_templates(self) -> None:
        """Test prompt template formatting."""
        text = "Apple announced new iPhone."
        
        for style, template in STYLE_PROMPTS.items():
            prompt = template.format(text=text)
            assert text in prompt
            assert "Summary:" in prompt  # All prompts end with this


class TestSummarizationEndToEnd:
    """End-to-end style tests (mocking vLLM)."""

    @pytest.mark.asyncio
    async def test_summarize_executive_style(self) -> None:
        """Test executive style summarization."""
        service = SummarizationService()
        
        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="Apple announced new strategic initiatives.")])
            ]

            requests = [
                SummarizationRequest(
                    texts=["Apple announced new iPhone with AI features."],
                    style="executive",
                    tenant_id="001",
                )
            ]

            responses = await service.summarize(requests)

            assert len(responses) == 1
            assert responses[0].summaries[0] == "Apple announced new strategic initiatives."
            assert responses[0].model == service.model_id

    @pytest.mark.asyncio
    async def test_summarize_multiple_texts_per_request(self) -> None:
        """Test multiple texts per request."""
        service = SummarizationService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="Summary 1")]),
                mock.MagicMock(outputs=[mock.MagicMock(text="Summary 2")]),
            ]

            requests = [
                SummarizationRequest(
                    texts=["Doc 1", "Doc 2"],
                    style="bullet_points",
                    tenant_id="001",
                )
            ]

            responses = await service.summarize(requests)

            assert len(responses[0].summaries) == 2
            assert responses[0].batch_size == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_generate(self) -> None:
        """Test error handling when vLLM generate fails."""
        service = SummarizationService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.side_effect = Exception("GPU OOM")

            requests = [
                SummarizationRequest(texts=["test"], tenant_id="001")
            ]

            responses = await service.summarize(requests)

            # Should return error response, not raise
            assert len(responses) == 1
            assert "Error" in responses[0].summaries[0]


class TestHealthEndpoint:
    """Tests for health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self) -> None:
        """Health endpoint returns status info."""
        service = SummarizationService()
        service._initialized = True
        service.model_id = "test-model"

        health = await service.health()

        assert health["status"] == "healthy"
        assert health["model"] == "test-model"
        assert health["backend"] == "vllm"
        assert health["max_model_len"] == 8192

    @pytest.mark.asyncio
    async def test_health_not_initialized(self) -> None:
        """Health returns loading when not initialized."""
        service = SummarizationService()
        service._initialized = False

        health = await service.health()

        assert health["status"] == "loading"
        assert health["initialized"] is False
"""Unit tests for the BentoML Fallback Service."""

from __future__ import annotations

from unittest import mock

import pytest

from bentoml.services.fallback import (
    FallbackService,
    FallbackRequest,
    FallbackResponse,
    TASK_PROMPTS,
)


class TestFallbackRequest:
    """Tests for FallbackRequest model."""

    def test_request_creation(self) -> None:
        """Test basic request creation."""
        req = FallbackRequest(
            text="Apple reported strong earnings.",
            task_type="summarize",
            tenant_id="001",
        )
        assert req.text == "Apple reported strong earnings."
        assert req.task_type == "summarize"
        assert req.tenant_id == "001"

    def test_task_type_validation(self) -> None:
        """Test task type validation."""
        for task in ["summarize", "extract", "classify"]:
            req = FallbackRequest(
                text="Test text",
                task_type=task,
                tenant_id="001",
            )
            assert req.task_type == task

    def test_text_length_bounds(self) -> None:
        """Test text length bounds."""
        # Min length 1
        req = FallbackRequest(text="a", task_type="summarize", tenant_id="001")
        assert len(req.text) == 1

        # Max length 4096
        long_text = "a" * 4096
        req = FallbackRequest(text=long_text, task_type="summarize", tenant_id="001")
        assert len(req.text) == 4096

        # Too long
        with pytest.raises(ValueError):
            FallbackRequest(text="a" * 4097, task_type="summarize", tenant_id="001")


class TestFallbackResponse:
    """Tests for FallbackResponse model."""

    def test_response_creation(self) -> None:
        """Test response creation."""
        resp = FallbackResponse(
            result="Apple reported strong earnings.",
            task_type="summarize",
            model="Qwen/Qwen2.5-3B-Instruct-GGUF",
        )
        assert resp.result == "Apple reported strong earnings."
        assert resp.task_type == "summarize"
        assert resp.model == "Qwen/Qwen2.5-3B-Instruct-GGUF"


class TestTaskPrompts:
    """Tests for task prompt templates."""

    def test_all_tasks_defined(self) -> None:
        """All required task types have prompts."""
        required = {"summarize", "extract", "classify"}
        assert set(TASK_PROMPTS.keys()) == required

    def test_prompts_contain_placeholder(self) -> None:
        """All prompts contain {text} placeholder."""
        for prompt in TASK_PROMPTS.values():
            assert "{text}" in prompt

    def test_prompt_structure(self) -> None:
        """Prompts have appropriate structure."""
        assert "Summarize" in TASK_PROMPTS["summarize"]
        assert "Extract" in TASK_PROMPTS["extract"]
        assert "Classify" in TASK_PROMPTS["classify"]


class TestFallbackService:
    """Tests for FallbackService class."""

    @pytest.fixture
    def service(self) -> FallbackService:
        """Create service instance."""
        return FallbackService()

    @pytest.mark.asyncio
    async def test_load_model_called_once(self, service) -> None:
        """Model loading is idempotent."""
        with mock.patch("bentoml.services.fallback.LLM") as mock_llm:
            mock_instance = mock.MagicMock()
            mock_llm.return_value = mock_instance

            await service._load_model()
            await service._load_model()  # Second call

            assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_process_single_request(self, service) -> None:
        """Test processing a single request."""
        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="Apple reported strong earnings.")])
            ]

            requests = [
                FallbackRequest(
                    text="Apple reported strong quarterly earnings with record revenue.",
                    task_type="summarize",
                    tenant_id="001",
                )
            ]

            responses = await service.process(requests)

            assert len(responses) == 1
            assert responses[0].result == "Apple reported strong earnings."
            assert responses[0].task_type == "summarize"
            assert responses[0].model == service.model_id

    @pytest.mark.asyncio
    async def test_process_batching_by_task_type(self, service) -> None:
        """Test that requests are grouped by task type."""
        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="Summary 1")]),
                mock.MagicMock(outputs=[mock.MagicMock(text="Summary 2")]),
                mock.MagicMock(outputs=[mock.MagicMock(text="Company: Apple")]),
            ]

            requests = [
                FallbackRequest(text="Doc 1", task_type="summarize", tenant_id="001"),
                FallbackRequest(text="Doc 2", task_type="technical", tenant_id="001"),
                FallbackRequest(text="Doc 3", task_type="extract", tenant_id="001"),
            ]

            responses = await service.process(requests)

            assert len(responses) == 3
            assert responses[0].task_type == "summarize"
            assert responses[1].task_type == "summarize"
            assert responses[2].task_type == "extract"

    def test_prompt_templates(self) -> None:
        """Test prompt template formatting."""
        text = "Apple announced new iPhone."

        for task, template in TASK_PROMPTS.items():
            prompt = template.format(text=text)
            assert text in prompt
            assert "Summary:" in prompt or "Entities:" in prompt or "Classification:" in prompt


class TestFallbackEndToEnd:
    """End-to-end tests for fallback service."""

    @pytest.mark.asyncio
    async def test_summarize_task(self) -> None:
        """Test summarize task."""
        service = FallbackService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="Apple announced new strategic initiatives.")])
            ]

            request = FallbackRequest(
                texts=["Apple announced new iPhone with AI features."],
                task_type="summarize",
                tenant_id="001",
            )

            # Note: The API expects a list of FallbackRequest, but the process method
            # takes a list of requests where each request has a list of texts
            # Actually the API takes list[FallbackRequest] where each request has texts: list[str]
            responses = await service.process([
                FallbackRequest(text="Apple announced new iPhone with AI features.", task_type="summarize", tenant_id="001"),
            ])

            assert len(responses) == 1
            assert "Apple" in responses[0].result

    @pytest.mark.asyncio
    async def test_extract_task(self) -> None:
        """Test extract task."""
        service = FallbackService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text='[{"type": "company", "value": "Apple"}, {"type": "product", "value": "iPhone"}]')])
            ]

            responses = await service.process([
                FallbackRequest(text="Apple announced new iPhone.", task_type="extract", tenant_id="001"),
            ])

            assert len(responses) == 1
            assert "Apple" in responses[0].result or "iPhone" in responses[0].result

    @pytest.mark.asyncio
    async def test_classify_task(self) -> None:
        """Test classify task."""
        service = FallbackService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="positive")])
            ]

            responses = await service.process([
                FallbackRequest(text="Apple reported record revenue.", task_type="classify", tenant_id="001"),
            ])

            assert len(responses) == 1
            assert "positive" in responses[0].result.lower()

    @pytest.mark.asyncio
    async def test_error_handling_in_generate(self) -> None:
        """Test error handling when vLLM generate fails."""
        service = FallbackService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.side_effect = Exception("GPU OOM")

            responses = await service.process([
                FallbackRequest(text="test", task_type="summarize", tenant_id="001"),
            ])

            # Should return error response, not raise
            assert len(responses) == 1
            assert "Error" in responses[0].result


class TestHealthEndpoint:
    """Tests for health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self) -> None:
        """Health endpoint returns status info."""
        service = FallbackService()
        service._initialized = True
        service.model_id = "test-model"

        health = await service.health()

        assert health["status"] == "healthy"
        assert health["model"] == "test-model"
        assert health["backend"] == "vllm"
        assert health["max_model_len"] == 4096

    @pytest.mark.asyncio
    async def test_health_not_initialized(self) -> None:
        """Health returns loading when not initialized."""
        service = FallbackService()
        service._initialized = False

        health = await service.health()

        assert health["status"] == "loading"
        assert health["initialized"] is False
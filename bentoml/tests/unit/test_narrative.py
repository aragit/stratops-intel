"""Unit tests for the BentoML Narrative Service."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from bentoml.services.narrative import (
    NarrativeRequest,
    NarrativeResponse,
    NarrativeSection,
    NarrativeService,
    NARRATIVE_PROMPTS,
)


class TestNarrativeSection:
    """Tests for NarrativeSection model."""

    def test_section_creation(self) -> None:
        """Test section creation."""
        section = NarrativeSection(
            title="Pricing Trends",
            content_uri="s3://bucket/trends.json",
            source_type="trend",
        )
        assert section.title == "Pricing Trends"
        assert section.content_uri == "s3://bucket/trends.json"
        assert section.source_type == "trend"

    def test_section_required_fields(self) -> None:
        """Test all required fields."""
        section = NarrativeSection(
            title="Test",
            content_uri="s3://test",
            source_type="test",
        )
        assert section.title == "Test"
        assert section.content_uri == "s3://test"
        assert section.source_type == "test"


class TestNarrativeRequest:
    """Tests for NarrativeRequest model."""

    def test_request_creation(self) -> None:
        """Test basic request creation."""
        req = NarrativeRequest(
            sections=[
                NarrativeSection(title="Test", content_uri="s3://test", source_type="test")
            ],
            narrative_type="executive_brief",
            tenant_id="001",
        )
        assert len(req.sections) == 1
        assert req.narrative_type == "executive_brief"

    def test_request_default_narrative_type(self) -> None:
        """Test default narrative type."""
        req = NarrativeRequest(
            sections=[NarrativeSection(title="T", content_uri="s3://t", source_type="t")],
            tenant_id="001",
        )
        assert req.narrative_type == "executive_brief"

    def test_section_count_validation(self) -> None:
        """Test section count bounds."""
        sections = [NarrativeSection(title=f"S{i}", content_uri=f"s3://{i}", source_type="t") for i in range(21)]
        
        with pytest.raises(ValueError):
            NarrativeRequest(sections=sections, tenant_id="001")

    def test_narrative_type_validation(self) -> None:
        """Test narrative type validation."""
        for ntype in ["executive_brief", "competitive_update", "threat_assessment"]:
            req = NarrativeRequest(
                sections=[NarrativeSection(title="T", content_uri="s3://t", source_type="t")],
                narrative_type=ntype,
                tenant_id="001",
            )
            assert req.narrative_type == ntype


class TestNarrativeResponse:
    """Tests for NarrativeResponse model."""

    def test_response_creation(self) -> None:
        """Test response creation."""
        resp = NarrativeResponse(
            narrative="# Test\n\nContent here.",
            key_takeaways=["Takeaway 1", "Takeaway 2"],
            confidence=0.85,
            model="test-model",
        )
        assert len(resp.key_takeaways) == 2
        assert resp.confidence == 0.85

    def test_confidence_bounds(self) -> None:
        """Test confidence bounds."""
        with pytest.raises(ValueError):
            NarrativeResponse(
                narrative="Test",
                confidence=1.5,  # > 1.0
                model="test",
            )
        with pytest.raises(ValueError):
            NarrativeResponse(
                narrative="Test",
                confidence=-0.1,  # < 0.0
                model="test",
            )


class TestNarrativePrompts:
    """Tests for narrative prompt templates."""

    def test_all_types_defined(self) -> None:
        """All required narrative types have prompts."""
        required = {"executive_brief", "competitive_update", "threat_assessment"}
        assert set(NARRATIVE_PROMPTS.keys()) == required

    def test_prompts_contain_sections_placeholder(self) -> None:
        """All prompts contain {sections} placeholder."""
        for prompt in NARRATIVE_PROMPTS.values():
            assert "{sections}" in prompt

    def test_prompt_structure(self) -> None:
        """Prompts have proper structure."""
        for ntype, prompt in NARRATIVE_PROMPTS.items():
            assert "Sections:" in prompt
            assert "{sections}" in prompt
            assert "Assessment:" in prompt or "Briefing:" in prompt or "Update:" in prompt


class TestNarrativeService:
    """Tests for NarrativeService class."""

    @pytest.fixture
    def service(self) -> NarrativeService:
        """Create service instance."""
        return NarrativeService()

    @pytest.mark.asyncio
    async def test_load_model_called_once(self, service) -> None:
        """Model loading is idempotent."""
        with mock.patch("bentoml.services.narrative.LLM") as mock_llm:
            mock_instance = mock.MagicMock()
            mock_llm.return_value = mock_instance

            await service._load_model()
            await service._load_model()  # Second call

            assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_narrative(self, service) -> None:
        """Test narrative generation."""
        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(text="# Executive Brief\n\n- Key point 1\n- Key point 2")])
            ]

            request = NarrativeRequest(
                sections=[
                    NarrativeSection(
                        title="Pricing Trend",
                        content_uri="s3://trends/pricing.json",
                        source_type="trend",
                    ),
                    NarrativeSection(
                        title="Correlation",
                        content_uri="s3://correlations/corr.json",
                        source_type="correlation",
                    ),
                ],
                narrative_type="executive_brief",
                tenant_id="001",
            )

            response = await service.generate(request)

            assert isinstance(response, NarrativeResponse)
            assert response.narrative
            assert response.model == service.model_id
            assert 0.0 <= response.confidence <= 1.0
            assert isinstance(response.key_takeaways, list)

    def test_build_prompt_includes_sections(self, service) -> None:
        """Test prompt building includes all sections."""
        request = NarrativeRequest(
            sections=[
                NarrativeSection(title="Trend 1", content_uri="s3://t1", source_type="trend"),
                NarrativeSection(title="Trend 2", content_uri="s3://t2", source_type="trend"),
            ],
            narrative_type="executive_brief",
            tenant_id="001",
        )

        prompt = service._build_prompt(request)

        assert "Trend 1" in prompt
        assert "Trend 2" in prompt
        assert "s3://t1" in prompt
        assert "s3://t2" in prompt
        assert "Executive Briefing:" in prompt

    def test_extract_key_takeaways_bullet_points(self, service) -> None:
        """Test extracting key takeaways from bullet points."""
        narrative = """# Executive Brief

Key developments:
- First key takeaway
- Second key takeaway
- Third key takeaway

Conclusion here."""

        takeaways = service._extract_key_takeaways(narrative)

        assert len(takeaways) == 3
        assert takeaways[0] == "First key takeaway"
        assert takeaways[1] == "Second key takeaway"
        assert takeaways[2] == "Third key takeaway"

    def test_extract_key_takeaways_numbered_list(self, service) -> None:
        """Test extraction from numbered lists."""
        narrative = """1. First takeaway
2. Second takeaway
3. Third takeaway"""

        takeaways = service._extract_key_takeaways(narrative)

        assert len(takeaways) == 3
        assert "First takeaway" in takeaways[0]

    def test_extract_key_takeaways_fallback(self, service) -> None:
        """Test fallback to first sentences when no bullets."""
        narrative = "First sentence. Second sentence. Third sentence. Fourth sentence."

        takeaways = service._extract_key_takeaways(narrative)

        assert len(takeaways) == 3
        assert all(t.endswith(".") for t in takeaways)

    def test_compute_confidence(self, service) -> None:
        """Test confidence computation."""
        # Empty narrative
        assert service._compute_confidence("", []) == 0.0

        # Short narrative
        conf = service._compute_confidence("Short.", [{}])
        assert 0.0 <= conf <= 1.0

        # Well-structured narrative
        narrative = """# Brief\n\n## Section 1\n- Point 1\n- Point 2\n\n## Section 2\n- Point 3"""
        conf = service._compute_confidence(narrative, [{}] * 3)
        assert 0.5 <= conf <= 1.0


class TestNarrativeEndToEnd:
    """End-to-end tests for narrative generation."""

    @pytest.mark.asyncio
    async def test_executive_brief_narrative(self) -> None:
        """Test executive brief generation."""
        service = NarrativeService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(
                    text="# Executive Brief\n\nSituation overview...\n\nKey developments:\n- Point 1\n- Point 2\n\nRecommended actions:\n- Action 1"
                )])
            ]

            request = NarrativeRequest(
                sections=[
                    NarrativeSection(
                        title="Pricing Trend",
                        content_uri="s3://trends/pricing.json",
                        source_type="trend",
                    ),
                ],
                narrative_type="executive_brief",
                tenant_id="001",
            )

            response = await service.generate(request)

            assert response.narrative
            assert response.confidence > 0.0
            assert "Executive Brief" in response.narrative

    @pytest.mark.asyncio
    async def test_competitive_update_narrative(self) -> None:
        """Test competitive update narrative type."""
        service = NarrativeService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(
                    text="# Competitive Update\n\nMarket position changes...\n\nPricing moves..."
                )])
            ]

            request = NarrativeRequest(
                sections=[],
                narrative_type="competitive_update",
                tenant_id="001",
            )

            response = await service.generate(request)

            assert response.narrative
            assert "Competitive Update" in response.narrative

    @pytest.mark.asyncio
    async def test_threat_assessment_narrative(self) -> None:
        """Test threat assessment narrative type."""
        service = NarrativeService()

        with mock.patch.object(service, "_load_model"):
            mock_llm = mock.MagicMock()
            service.llm = mock_llm
            service.sampling_params = mock.MagicMock()
            service._initialized = True

            mock_llm.generate.return_value = [
                mock.MagicMock(outputs=[mock.MagicMock(
                    text="# Threat Assessment\n\nNew entrants...\n\nPricing pressure..."
                )])
            ]

            request = NarrativeRequest(
                sections=[],
                narrative_type="threat_assessment",
                tenant_id="001",
            )

            response = await service.generate(request)

            assert response.narrative
            assert "Threat Assessment" in response.narrative


class TestHealthEndpoint:
    """Tests for health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_status(self) -> None:
        """Health endpoint returns status info."""
        service = NarrativeService()
        service._initialized = True
        service.model_id = "test-model"

        health = await service.health()

        assert health["status"] == "healthy"
        assert health["model"] == "test-model"
        assert health["backend"] == "vllm"
        assert health["max_model_len"] == 16384

    @pytest.mark.asyncio
    async def test_health_not_initialized(self) -> None:
        """Health returns loading when not initialized."""
        service = NarrativeService()
        service._initialized = False

        health = await service.health()

        assert health["status"] == "loading"
        assert health["initialized"] is False
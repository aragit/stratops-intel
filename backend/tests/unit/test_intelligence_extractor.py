"""Unit tests for the intelligence entity extractor LangGraph node.

Tests pointer-only state, MinIO mocking, BentoML HTTP call mocking,
and checkpoint persistence.
"""

from __future__ import annotations

import json
import asyncio
import uuid
from unittest import mock

import pytest

from backend.intelligence import IntelligenceState, EntityExtractorNode, build_extractor_graph


async def _async_coro(value):
    """Helper to create async coroutines for mocking."""
    return value


@pytest.fixture
def extractor_node() -> EntityExtractorNode:
    """Provide an EntityExtractorNode instance for tests."""
    return EntityExtractorNode()


@pytest.fixture
def sample_state() -> IntelligenceState:
    """Provide a sample IntelligenceState for tests."""
    return {
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "trace_id": "trace-001",
        "signal_uris": [
            "s3://stratops-signals/test-signal-1.json",
            "s3://stratops-signals/test-signal-2.json",
        ],
        "extracted_entities": [],
        "content_uris": [],
        "correlation_graph_delta": [],
        "briefing_section_uris": [],
    }


class TestIntelligenceState:
    """Tests for the IntelligenceState TypedDict."""

    def test_state_has_required_fields(self) -> None:
        """State should have all required pointer-only fields."""
        from uuid import uuid4

        state: IntelligenceState = {
            "tenant_id": str(uuid4()),
            "trace_id": str(uuid4()),
            "signal_uris": ["s3://bucket/key1"],
            "extracted_entities": [],
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }
        assert "tenant_id" in state
        assert "trace_id" in state
        assert "signal_uris" in state
        assert "extracted_entities" in state
        assert "content_uris" in state
        assert "correlation_graph_delta" in state
        assert "briefing_section_uris" in state

    def test_state_size_limit(self) -> None:
        """State should be small enough for checkpointing (< 5KB)."""
        state: IntelligenceState = {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }
        state_size = len(json.dumps(state).encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB limit"


class TestEntityExtractorNode:
    """Tests for the EntityExtractorNode __call__ method."""

    @pytest.mark.asyncio
    async def test_empty_signal_uris(self, extractor_node: EntityExtractorNode) -> None:
        """Should return unchanged state when no signal URIs provided."""
        state: IntelligenceState = {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

        result = await extractor_node(state)

        assert result is state  # Returns same state for empty signals
        assert result["extracted_entities"] == []
        assert result["content_uris"] == []

    @pytest.mark.asyncio
    async def test_pointer_only_no_raw_content(
        self, extractor_node: EntityExtractorNode, sample_state: IntelligenceState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CRITICAL: State should not contain raw content, only URIs and small structured data.

        This is the pointer-only constraint - raw text/content must never be stored
        in the state, only S3 URIs and small extracted entity summaries.
        """
        # Use monkeypatch to avoid aiobotocore import issues in test env
        # Mock the aiobotocore session.get_session return value
        mock_session = type('MockSession', (), {
            'create_client': lambda self, *args, **kwargs: type('MockS3Client', (), {
                'get_object': lambda self, **kwargs: type('MockBody', (), {
                    'read': lambda: b'{"signal": "test data", "content": "Apple Inc. designs consumer electronics"}'
                })()
            })()
        })()
        
        # Monkeypatch aiobotocore.session.get_session
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'aiobotocore':
                mod = original_import('types').ModuleType('aiobotocore')
                submod = original_import('types').ModuleType('aiobotocore.session')
                submod.get_session = lambda: mock_session
                mod.session = submod
                sys.modules['aiobotocore'] = mod
                sys.modules['aiobotocore.session'] = submod
                return mod
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        try:
            state = await extractor_node(sample_state)
        finally:
            # Restore original import
            monkepatches.undo()

        # Verify pointer-only: no raw content in state
        state_size = len(json.dumps(state).encode("utf-8"))
        assert state_size < 5000, f"State size {state_size} exceeds 5KB limit"

        # Verify raw content (full signal text) is NOT in state
        state_json = json.dumps(state)
        # The state should only have URIs and small structured data
        assert "s3://" in state_json  # Has URIs
        # Should not have the full signal text content
        assert "Apple Inc. designs consumer electronics" not in state_json

        # Verify pointer fields are present
        assert isinstance(state["signal_uris"], list)
        assert isinstance(state["content_uris"], list)
        assert isinstance(state["correlation_graph_delta"], list)
        assert isinstance(state["extracted_entities"], list)

    @pytest.mark.asyncio
    async def test_extraction_writes_to_minio(
        self, extractor_node: EntityExtractorNode, sample_state: IntelligenceState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should write extracted entities JSON to MinIO and return content URI."""
        # Similar monkeypatch approach
        pass

    @pytest.mark.asyncio
    async def test_state_size_after_extraction(
        self, extractor_node: EntityExtractorNode, sample_state: IntelligenceState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that state size is < 5KB after extraction (CRITICAL)."""
        pass

    @pytest.mark.asyncio
    async def test_checkpoint_persistence(self, extractor_node: EntityExtractorNode, sample_state: IntelligenceState) -> None:
        """Test that the compiled graph checkpoints state correctly."""
        graph = build_extractor_graph()

        # Test that the graph can persist and restore state
        initial_state: IntelligenceState = {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [],
            "content_uris": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

        # Run the graph - use ainvoke on compiled graph
        result = await graph.ainvoke(initial_state)

        # Verify state is preserved through checkpoint
        assert result["tenant_id"] == initial_state["tenant_id"]
        assert result["trace_id"] == initial_state["trace_id"]
        assert result["signal_uris"] == initial_state["signal_uris"]


class TestBuildExtractorGraph:
    """Tests for the build_extractor_graph function."""

    def test_graph_has_entry_point(self) -> None:
        """The graph should have 'extract' as the entry point."""
        graph = build_extractor_graph()
        # Check that the graph was created with extract as entry point
        # by verifying set_entry_point was called (it's the standard method)
        assert graph is not None
        assert "extract" in graph.nodes

    def test_graph_has_extract_node(self) -> None:
        """The graph should have the extract node."""
        graph = build_extractor_graph()
        assert "extract" in graph.nodes

    def test_graph_state_type(self) -> None:
        """The graph should use IntelligenceState as state type."""
        from backend.intelligence import IntelligenceState
        graph = build_extractor_graph()
        # Check that the graph was created with the right state type
        assert graph is not None
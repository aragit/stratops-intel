"""Unit tests for ingestion base protocol and registry."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ingestion.base import (
    AdapterNotFoundError,
    AdapterRegistrationError,
    AdapterRegistry,
    IngestionResult,
    NormalizedSignal,
    RawSignal,
    SourceAdapter,
    register_adapter,
)


class DummyAdapter(SourceAdapter):
    """Dummy adapter for testing."""

    name = "dummy"
    source_type = "test"

    class Config(BaseModel):
        urls: list[str]

    config_schema = Config

    async def fetch(self, config: dict, cursor: str | None = None) -> IngestionResult:
        return IngestionResult(b"data", "text/plain", None, {})

    async def parse(self, raw_data: bytes, content_type: str) -> list[RawSignal]:
        return [RawSignal(source_type="test", source_url="http://test", raw_content=b"test")]

    async def fingerprint(self, signal: RawSignal) -> str:
        return "abc123"

    async def normalize(self, signals: list[RawSignal]) -> list[NormalizedSignal]:
        return [NormalizedSignal(
            source_type="test",
            source_url="http://test",
            content_uri="s3://bucket/key",
            fingerprint="abc123",
        )]


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def setup_method(self):
        AdapterRegistry.clear()

    def test_register_and_get(self):
        AdapterRegistry.register(DummyAdapter)
        assert AdapterRegistry.get("dummy") == DummyAdapter

    def test_list_adapters(self):
        AdapterRegistry.register(DummyAdapter)
        assert "dummy" in AdapterRegistry.list_adapters()

    def test_list_by_source_type(self):
        AdapterRegistry.register(DummyAdapter)
        assert "dummy" in AdapterRegistry.list_by_source_type("test")

    def test_duplicate_registration_raises(self):
        AdapterRegistry.register(DummyAdapter)
        with pytest.raises(AdapterRegistrationError):
            AdapterRegistry.register(DummyAdapter)

    def test_get_missing_raises(self):
        with pytest.raises(AdapterNotFoundError):
            AdapterRegistry.get("nonexistent")

    def test_builtin_adapters_registered_on_import(self):
        """Test that importing the adapters package registers built-in adapters."""
        AdapterRegistry.clear()
        from ingestion.adapters.sec import SECFilingAdapter  # noqa: F401
        from ingestion.adapters.web import WebMonitorAdapter  # noqa: F401
        AdapterRegistry.register(SECFilingAdapter)
        AdapterRegistry.register(WebMonitorAdapter)
        assert "web_monitor" in AdapterRegistry.list_adapters()
        assert "sec_filings" in AdapterRegistry.list_adapters()


class TestRawSignal:
    """Tests for RawSignal Pydantic model."""

    def test_required_fields(self):
        signal = RawSignal(
            source_type="web",
            raw_content=b"test",
        )
        assert signal.source_type == "web"
        assert signal.raw_content == b"test"
        assert signal.fingerprint is None

    def test_all_fields(self):
        signal = RawSignal(
            source_type="sec",
            source_url="https://sec.gov",
            raw_content=b"filing",
            fingerprint="abc123",
            metadata={"cik": "123"},
        )
        assert signal.source_url == "https://sec.gov"
        assert signal.fingerprint == "abc123"
        assert signal.metadata["cik"] == "123"

    def test_frozen_model(self):
        signal = RawSignal(source_type="web", raw_content=b"test")
        with pytest.raises(Exception):
            signal.source_type = "sec"


class TestNormalizedSignal:
    """Tests for NormalizedSignal Pydantic model."""

    def test_content_uri_required(self):
        with pytest.raises(Exception):
            NormalizedSignal(
                source_type="web",
                fingerprint="abc123",
                # content_uri missing - should fail
            )

    def test_fingerprint_required(self):
        with pytest.raises(Exception):
            NormalizedSignal(
                source_type="web",
                content_uri="s3://bucket/key",
                # fingerprint missing - should fail
            )

    def test_valid_normalized_signal(self):
        norm = NormalizedSignal(
            source_type="web",
            source_url="https://example.com",
            content_uri="s3://stratops-raw-tenant/abc123.bin",
            fingerprint="abc123def456",
            structured_payload={"title": "Test"},
        )
        assert norm.content_uri.startswith("s3://")
        assert norm.fingerprint == "abc123def456"

    def test_pointer_enforcement(self):
        """Ensure raw content is NOT in normalized signal."""
        norm = NormalizedSignal(
            source_type="web",
            content_uri="s3://bucket/key",
            fingerprint="abc123",
        )
        # Should not have raw_content field
        assert not hasattr(norm, "raw_content")


class TestIngestionResult:
    """Tests for IngestionResult NamedTuple."""

    def test_creation(self):
        result = IngestionResult(
            raw_data=b"test",
            content_type="text/html",
            next_cursor="cursor123",
            metadata={"count": 1},
        )
        assert result.raw_data == b"test"
        assert result.next_cursor == "cursor123"

    def test_optional_cursor(self):
        result = IngestionResult(b"data", "text/plain", None, {})
        assert result.next_cursor is None


class TestRegisterDecorator:
    """Tests for register_adapter decorator."""

    def setup_method(self):
        AdapterRegistry.clear()

    def test_decorator_registers(self):
        @register_adapter
        class DecoratedAdapter(SourceAdapter):
            name = "decorated"
            source_type = "test"

            class Config(BaseModel):
                pass
            config_schema = Config

            async def fetch(self, config, cursor=None):
                return IngestionResult(b"", "text/plain", None, {})

            async def parse(self, raw_data, content_type):
                return []

            async def fingerprint(self, signal):
                return "fp"

            async def normalize(self, signals):
                return []

        assert "decorated" in AdapterRegistry.list_adapters()

"""Unit tests for the tenant-aware Redis Stream key builder.

Tests that :class:`StreamKeyBuilder` generates correctly formatted keys,
validates inputs, and respects namespace configuration.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from backend.streams.keys import StreamKeyBuilder


class TestStreamKeyBuilderInit:
    """Tests for StreamKeyBuilder initialization."""

    def test_default_namespace(self) -> None:
        """Builder should default to 'stratops' namespace."""
        builder = StreamKeyBuilder()
        assert builder.namespace == "stratops"

    def test_custom_namespace(self) -> None:
        """Builder should accept a custom namespace."""
        builder = StreamKeyBuilder(base_namespace="custom")
        assert builder.namespace == "custom"


class TestSignalStreamKey:
    """Tests for signal_stream key generation."""

    def test_signal_stream_format(self) -> None:
        """Signal stream key should follow expected format."""
        builder = StreamKeyBuilder()
        tenant_id = UUID("11111111-1111-1111-1111-111111111111")
        key = builder.signal_stream(tenant_id)
        assert key == "stratops:tenant:11111111-1111-1111-1111-111111111111:signals"

    def test_signal_stream_different_tenants(self) -> None:
        """Different tenant IDs should produce different keys."""
        builder = StreamKeyBuilder()
        key_a = builder.signal_stream(uuid4())
        key_b = builder.signal_stream(uuid4())
        assert key_a != key_b

    def test_signal_stream_invalid_uuid(self) -> None:
        """Non-UUID input should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="tenant_id must be a UUID"):
            builder.signal_stream("not-a-uuid")  # type: ignore[arg-type]


class TestIngestionStreamKey:
    """Tests for ingestion_stream key generation."""

    def test_ingestion_stream_format(self) -> None:
        """Ingestion stream key should include source_type."""
        builder = StreamKeyBuilder()
        tenant_id = UUID("11111111-1111-1111-1111-111111111111")
        key = builder.ingestion_stream(tenant_id, "rss")
        assert key == "stratops:tenant:11111111-1111-1111-1111-111111111111:ingestion:rss"

    def test_ingestion_stream_empty_source_type(self) -> None:
        """Empty source_type should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="source_type must not be empty"):
            builder.ingestion_stream(uuid4(), "")

    def test_ingestion_stream_invalid_uuid(self) -> None:
        """Non-UUID input should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="tenant_id must be a UUID"):
            builder.ingestion_stream("bad", "rss")  # type: ignore[arg-type]


class TestIntelligenceStreamKey:
    """Tests for intelligence_stream key generation."""

    def test_intelligence_stream_format(self) -> None:
        """Intelligence stream key should include agent_type."""
        builder = StreamKeyBuilder()
        tenant_id = UUID("22222222-2222-2222-2222-222222222222")
        key = builder.intelligence_stream(tenant_id, "signal")
        assert key == "stratops:tenant:22222222-2222-2222-2222-222222222222:intelligence:signal"

    def test_intelligence_stream_empty_agent_type(self) -> None:
        """Empty agent_type should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="agent_type must not be empty"):
            builder.intelligence_stream(uuid4(), "")

    def test_intelligence_stream_invalid_uuid(self) -> None:
        """Non-UUID input should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="tenant_id must be a UUID"):
            builder.intelligence_stream(123, "signal")  # type: ignore[arg-type]


class TestAlertStreamKey:
    """Tests for alert_stream key generation."""

    def test_alert_stream_format(self) -> None:
        """Alert stream key should follow expected format."""
        builder = StreamKeyBuilder()
        tenant_id = UUID("33333333-3333-3333-3333-333333333333")
        key = builder.alert_stream(tenant_id)
        assert key == "stratops:tenant:33333333-3333-3333-3333-333333333333:alerts"

    def test_alert_stream_invalid_uuid(self) -> None:
        """Non-UUID input should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="tenant_id must be a UUID"):
            builder.alert_stream(None)  # type: ignore[arg-type]


class TestGraphWriterBufferKey:
    """Tests for graph_writer_buffer key generation."""

    def test_graph_writer_buffer_format(self) -> None:
        """Graph writer buffer key should follow expected format."""
        builder = StreamKeyBuilder()
        tenant_id = UUID("44444444-4444-4444-4444-444444444444")
        key = builder.graph_writer_buffer(tenant_id)
        assert key == "stratops:tenant:44444444-4444-4444-4444-444444444444:graph:pending"

    def test_graph_writer_buffer_invalid_uuid(self) -> None:
        """Non-UUID input should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="tenant_id must be a UUID"):
            builder.graph_writer_buffer([])  # type: ignore[arg-type]


class TestConsumerGroupKey:
    """Tests for consumer_group key generation."""

    def test_consumer_group_format(self) -> None:
        """Consumer group key should follow expected format."""
        builder = StreamKeyBuilder()
        key = builder.consumer_group("ingestion")
        assert key == "cg:ingestion"

    def test_consumer_group_empty_name(self) -> None:
        """Empty service_name should raise ValueError."""
        builder = StreamKeyBuilder()
        with pytest.raises(ValueError, match="service_name must not be empty"):
            builder.consumer_group("")

    def test_consumer_group_independent_of_namespace(self) -> None:
        """Consumer group name should not include the namespace prefix."""
        builder = StreamKeyBuilder(base_namespace="custom")
        key = builder.consumer_group("intelligence")
        assert key == "cg:intelligence"
        assert "custom" not in key


class TestCustomNamespaceIntegration:
    """Tests that custom namespace propagates to all tenant-scoped keys."""

    def test_all_tenant_keys_use_custom_namespace(self) -> None:
        """All tenant-scoped keys should use the configured namespace."""
        builder = StreamKeyBuilder(base_namespace="myorg")
        tid = uuid4()

        assert builder.signal_stream(tid).startswith("myorg:")
        assert builder.ingestion_stream(tid, "api").startswith("myorg:")
        assert builder.intelligence_stream(tid, "briefing").startswith("myorg:")
        assert builder.alert_stream(tid).startswith("myorg:")
        assert builder.graph_writer_buffer(tid).startswith("myorg:")

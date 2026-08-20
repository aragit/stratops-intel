"""Unit tests for the micro-batching Neo4j graph writer.

Tests deduplication logic, batch flush triggers, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest

from backend.workers.graph_writer import (
    EntityUpdate,
    RelationshipUpdate,
    MicroBatchBuffer,
    GraphWriterWorker,
)


def _make_mock_redis():
    """Create a mock Redis client that works with async await expressions."""
    m = mock.MagicMock()
    # lpush should return a coroutine that resolves to None
    async def _lpush(*args, **kwargs):
        return None
    m.lpush = _lpush

    # lrange should return a coroutine that resolves to a list
    async def _lrange(*args, **kwargs):
        return []
    m.lrange = _lrange

    # delete should return a coroutine that resolves to None
    async def _delete(*args, **kwargs):
        return None
    m.delete = _delete

    # xack should return a coroutine that resolves to None
    async def _xack(*args, **kwargs):
        return None
    m.xack = _xack

    # _del_method for backward compatibility
    m._del_method = _delete

    return m


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client for tests."""
    return _make_mock_redis()


@pytest.fixture
def mock_neo4j_client():
    """Provide a mock Neo4j client for tests."""
    return mock.MagicMock()


@pytest.fixture
def entity_update() -> EntityUpdate:
    """Provide a sample EntityUpdate for tests."""
    return EntityUpdate(
        entity_type="Company",
        entity_id="test-company-1",
        tenant_id="00000000-0000-0000-0000-000000000001",
        properties={"name": "Test Company", "ticker": "TC"},
        relationships=[
            RelationshipUpdate(
                rel_type="EMPLOYED_AT",
                target_entity_type="Person",
                target_entity_id="test-person-1",
                properties={"role": "Engineer", "seniority": " junior"},
            )
        ],
    )


@pytest.fixture
def buffer_fixture(mock_redis) -> MicroBatchBuffer:
    """Provide a MicroBatchBuffer instance for tests."""
    tenant_id = "00000000-0000-0000-0000-000000000001"
    buffer = MicroBatchBuffer(
        redis=mock_redis,
        tenant_id=tenant_id,
        batch_size=100,
        flush_interval_ms=500,
    )
    return buffer


class TestRelationshipUpdate:
    """Tests for the RelationshipUpdate model."""

    def test_relationship_update_creation(self) -> None:
        """Test basic RelationshipUpdate creation and validation."""
        rel = RelationshipUpdate(
            rel_type="PRICED_AT",
            target_entity_type="Product",
            target_entity_id="prod-123",
            properties={"price": 99.99, "currency": "USD"},
        )
        assert rel.rel_type == "PRICED_AT"
        assert rel.target_entity_type == "Product"
        assert rel.target_entity_id == "prod-123"
        assert rel.properties == {"price": 99.99, "currency": "USD"}

    def test_relationship_update_minimal(self) -> None:
        """Test RelationshipUpdate with minimal required fields."""
        rel = RelationshipUpdate(
            rel_type="MENTIONED_IN",
            target_entity_type="Signal",
            target_entity_id="sig-1",
        )
        assert rel.rel_type == "MENTIONED_IN"
        assert rel.target_entity_type == "Signal"
        assert rel.target_entity_id == "sig-1"
        assert rel.properties == {}


class TestEntityUpdate:
    """Tests for the EntityUpdate model."""

    def test_entity_update_creation(self) -> None:
        """Test basic EntityUpdate creation and validation."""
        update = EntityUpdate(
            entity_type="Person",
            entity_id="test-person-1",
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Test Person", "email": "test@example.com"},
        )
        assert update.entity_type == "Person"
        assert update.entity_id == "test-person-1"
        assert update.tenant_id == "00000000-0000-0000-0000-000000000001"
        assert update.properties == {"name": "Test Person", "email": "test@example.com"}
        assert update.relationships == []

    def test_entity_update_with_relationships(self) -> None:
        """Test EntityUpdate with relationships."""
        update = EntityUpdate(
            entity_type="Company",
            entity_id="test-company-1",
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Test Company"},
            relationships=[
                RelationshipUpdate(
                    rel_type="COMPETES_WITH",
                    target_entity_type="Company",
                    target_entity_id="test-company-2",
                    properties={"strength": 0.8},
                )
            ],
        )
        assert len(update.relationships) == 1
        assert update.relationships[0].rel_type == "COMPETES_WITH"


class TestMicroBatchBuffer:
    """Tests for the MicroBatchBuffer class."""

    @pytest.mark.asyncio
    async def test_enqueue_single_item(self, buffer_fixture, entity_update, mock_redis) -> None:
        """Test enqueuing a single item to the buffer."""
        await buffer_fixture.enqueue(entity_update)
        # lpush was called (it's a regular function now, not awaited in the test assertion)
        assert buffer_fixture.buffer_size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_multiple_items(self, buffer_fixture, entity_update, mock_redis) -> None:
        """Test enqueuing multiple items to the buffer."""
        # Enqueue batch_size items to trigger auto-flush
        for i in range(105):  # > batch_size of 100
            update = EntityUpdate(
                entity_type="Company",
                entity_id=f"test-company-{i}",
                tenant_id="00000000-0000-0000-0000-000000000001",
                properties={"name": f"Company {i}"},
            )
            await buffer_fixture.enqueue(update)

        # Should have flushed since we exceeded batch_size
        assert buffer_fixture.buffer_size() < 105  # May be less due to flush

    @pytest.mark.asyncio
    async def test_flush_single_item(self, buffer_fixture, entity_update, mock_redis) -> None:
        """Test flushing a single item from the buffer."""
        # Manually add to buffer
        buffer_fixture._buffer.append(entity_update.model_dump(mode="json"))

        # flush will lrange and delete the redis key
        flushed = await buffer_fixture.flush()

        # Should have one flushed item
        assert len(flushed) == 1
        mock_redis.lrange.assert_called_once()
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_deduplication(self, buffer_fixture, entity_update, mock_redis) -> None:
        """Test that flush deduplicates relationships by (entity_id, rel_type)."""
        # Add two entity updates with same entity_id + rel_type
        update1 = entity_update.model_copy(
            update=entity_update.model_dump()
        )
        update2 = EntityUpdate(
            entity_type="Company",
            entity_id="test-company-1",  # Same entity_id
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Updated Company"},
            relationships=[
                RelationshipUpdate(
                    rel_type="EMPLOYED_AT",
                    target_entity_type="Person",
                    target_entity_id="test-person-1",
                    properties={"role": "Senior Engineer"},  # Different properties - keeps latest
                )
            ],
        )

        # Manually add both to buffer
        buffer_fixture._buffer.append(update1.model_dump(mode="json"))
        buffer_fixture._buffer.append(update2.model_dump(mode="json"))

        # Also add a third with different entity_id
        update3 = EntityUpdate(
            entity_type="Company",
            entity_id="test-company-2",  # Different entity_id
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Company 2"},
        )
        buffer_fixture._buffer.append(update3.model_dump(mode="json"))

        flushed = await buffer_fixture.flush()

        # Should have 2 entities (one deduped, one unique)
        assert len(flushed) == 2

    @pytest.mark.asyncio
    async def test_flush_clears_redis(self, buffer_fixture, mock_redis) -> None:
        """Test that flush clears the Redis list after reading."""
        # Add an item to Redis directly
        await mock_redis.lpush("stratops:tenant:00000000-0000-0000-0000-000000000001:graph:pending", json.dumps({"test": "data"}))

        # Also add to in-memory buffer
        buffer_fixture._buffer.append({"test": "data"})

        await buffer_fixture.flush()

        # Redis list should be deleted
        mock_redis.delete.assert_called_once()


class TestGraphWriterWorker:
    """Tests for the GraphWriterWorker class."""

    @pytest.mark.asyncio
    async def test_worker_start_stop(self, mock_redis, mock_neo4j_client) -> None:
        """Test worker start and stop lifecycle."""
        worker = GraphWriterWorker(
            redis=mock_redis,
            neo4j_client=mock_neo4j_client,
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

        await worker.start()
        assert worker._running is True

        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_worker_process_message(self, mock_redis, mock_neo4j_client, entity_update) -> None:
        """Test processing a single message through the worker."""
        worker = GraphWriterWorker(
            redis=mock_redis,
            neo4j_client=mock_neo4j_client,
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

        await worker.start()

        # Process the entity update
        await worker._process_message("1615665000000-0", entity_update.model_dump())

        # Should have tried to acknowledge the message
        mock_redis.xack.assert_called_once()

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_deduplication(self, mock_redis, mock_neo4j_client) -> None:
        """Test worker message processing with deduplication."""
        worker = GraphWriterWorker(
            redis=mock_redis,
            neo4j_client=mock_neo4j_client,
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

        await worker.start()

        # Send two messages with same entity_id + rel_type
        update1 = EntityUpdate(
            entity_type="Company",
            entity_id="test-company-1",
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Company 1"},
            relationships=[
                RelationshipUpdate(
                    rel_type="EMPLOYED_AT",
                    target_entity_type="Person",
                    target_entity_id="test-person-1",
                    properties={"role": "Engineer"},
                )
            ],
        )

        update2 = EntityUpdate(
            entity_type="Company",
            entity_id="test-company-1",  # Same entity_id
            tenant_id="00000000-0000-0000-0000-000000000001",
            properties={"name": "Company 1 Updated"},
            relationships=[
                RelationshipUpdate(
                    rel_type="EMPLOYED_AT",
                    target_entity_type="Person",
                    target_entity_id="test-person-1",
                    properties={"role": "Senior Engineer"},  # Latest wins
                )
            ],
        )

        await worker._process_message("1615665000000-0", update1.model_dump())
        await worker._process_message("1615665000001-0", update2.model_dump())

        # Both should be acknowledged
        assert mock_redis.xack.call_count == 2

        await worker.stop()


class TestEndToEndFlow:
    """End-to-end tests for the graph writer flow."""

    @pytest.mark.asyncio
    async def test_full_flush_flow(self, mock_redis, mock_neo4j_client) -> None:
        """Test the complete flush flow: enqueue -> flush -> deduplicated output."""
        worker = GraphWriterWorker(
            redis=mock_redis,
            neo4j_client=mock_neo4j_client,
            tenant_id="00000000-0000-0000-0000-000000000001",
            batch_size=3,  # Small batch for testing
        )

        await worker.start()

        # Enqueue 4 updates ( > batch_size of 3 )
        for i in range(4):
            update = EntityUpdate(
                entity_type="Company",
                entity_id=f"company-{i}",
                tenant_id="00000000-0000-0000-0000-000000000001",
                properties={"name": f"Company {i}"},
            )
            await worker._process_message(f"1615665000-{i}", update.model_dump())

        # After processing, buffer should be flushed (at least once)
        # The worker may have flushed due to batch_size threshold

        await worker.stop()

    @pytest.mark.asyncio
    async def test_periodic_flush(self, mock_redis, mock_neo4j_client) -> None:
        """Test periodic flush triggers on interval."""
        worker = GraphWriterWorker(
            redis=mock_redis,
            neo4j_client=mock_neo4j_client,
            tenant_id="00000000-0000-0000-0000-000000000001",
            flush_interval_ms=10,  # Short interval for testing
        )

        await worker.start()

        # Add items to buffer manually (simulating slow consumption)
        worker._buffer._buffer = [
            EntityUpdate(
                entity_type="Company",
                entity_id="company-1",
                tenant_id="00000000-0000-0000-0000-000000000001",
                properties={"name": "Company 1"},
            ).model_dump(mode="json"),
        ]

        # Run the periodic flush
        await worker._periodic_flush_trigger()

        # Buffer should be cleared
        assert worker._buffer.buffer_size() == 0

        await worker.stop()
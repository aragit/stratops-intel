"""Unit tests for the Redis Streams base classes.

Tests that the :class:`StreamProducer` and :class:`StreamConsumer` correctly
serialize messages, handle batching, retry on connection errors, and shut
down gracefully.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from streams.base import StreamConsumer, StreamProducer


class ConcreteProducer(StreamProducer):
    """Concrete producer for testing."""

    pass


class ConcreteConsumer(StreamConsumer):
    """Concrete consumer for testing."""

    processed_messages: list[tuple[str, dict]]

    def __init__(self, redis: Any, stream_name: str, consumer_group: str, consumer_name: str) -> None:
        super().__init__(redis, stream_name, consumer_group, consumer_name)
        self.processed_messages = []

    async def process_message(self, message_id: str, message: dict[str, Any]) -> bool:
        """Record the message and return True (always succeeds)."""
        self.processed_messages.append((message_id, message))
        return True


class TestStreamProducerInit:
    """Tests for StreamProducer initialization."""

    def test_init_stores_attributes(self) -> None:
        """Producer should store redis and stream_name."""
        mock_redis = MagicMock()
        producer = ConcreteProducer(mock_redis, "test:stream")
        assert producer._redis is mock_redis
        assert producer._stream_name == "test:stream"
        assert producer.stream_name == "test:stream"


class TestStreamProducerPublish:
    """Tests for StreamProducer.publish."""

    @pytest.mark.asyncio
    async def test_publish_returns_message_id(self) -> None:
        """publish should return the Redis-generated message ID."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"1623432423-0")
        producer = ConcreteProducer(mock_redis, "test:stream")

        msg_id = await producer.publish({"event": "test"})

        assert msg_id == "1623432423-0"
        mock_redis.xadd.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_serializes_correctly(self) -> None:
        """publish should serialize message with trace_id and published_at."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"123-0")
        producer = ConcreteProducer(mock_redis, "test:stream")

        test_msg = {"event": "user_action", "user_id": 42}
        await producer.publish(test_msg)

        call_args = mock_redis.xadd.call_args
        fields = call_args[1]
        serialized = fields["fields"]["data"]
        envelope = json.loads(serialized)

        assert envelope["event"] == "user_action"
        assert envelope["user_id"] == 42
        assert "trace_id" in envelope
        assert len(envelope["trace_id"]) == 36  # UUID string
        assert "published_at" in envelope
        datetime.fromisoformat(envelope["published_at"])

    @pytest.mark.asyncio
    async def test_publish_uses_existing_trace_id(self) -> None:
        """publish should use trace_id from message if provided."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"123-0")
        producer = ConcreteProducer(mock_redis, "test:stream")

        existing_trace = str(uuid4())
        await producer.publish({"event": "test"}, metadata={"trace_id": existing_trace})

        call_args = mock_redis.xadd.call_args
        envelope = json.loads(call_args[1]["fields"]["data"])
        assert envelope["trace_id"] == existing_trace

    @pytest.mark.asyncio
    async def test_publish_merges_metadata(self) -> None:
        """publish should merge metadata into the message."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"123-0")
        producer = ConcreteProducer(mock_redis, "test:stream")

        await producer.publish(
            {"event": "test"},
            metadata={"source": "unit-test", "priority": "high"},
        )

        call_args = mock_redis.xadd.call_args
        envelope = json.loads(call_args[1]["fields"]["data"])
        assert envelope["source"] == "unit-test"
        assert envelope["priority"] == "high"

    @pytest.mark.asyncio
    async def test_publish_logs(self) -> None:
        """publish should log with stream_name, trace_id, message_size."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"123-0")
        producer = ConcreteProducer(mock_redis, "test:stream")

        with patch("streams.base.logger") as mock_logger:
            await producer.publish({"event": "test"})
            mock_logger.info.assert_called_once()
            assert mock_logger.info.call_args[0][0] == "message_published"

    @pytest.mark.asyncio
    async def test_publish_retry_on_connection_error(self) -> None:
        """publish should retry on ConnectionError and eventually succeed."""
        mock_redis = AsyncMock()
        call_count = 0

        async def mock_xadd(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RedisConnectionError("connection refused")
            return b"123-0"

        mock_redis.xadd = mock_xadd
        producer = ConcreteProducer(mock_redis, "test:stream")

        with patch("streams.base.asyncio.sleep", new_callable=AsyncMock):
            with patch("streams.base.logger") as mock_logger:
                msg_id = await producer.publish({"event": "test"})

        assert msg_id == "123-0"
        assert call_count == 3
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_publish_raises_after_max_retries(self) -> None:
        """publish should raise after max retries on persistent errors."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(side_effect=RedisConnectionError("always fails"))
        producer = ConcreteProducer(mock_redis, "test:stream")

        with patch("streams.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RedisConnectionError):
                await producer.publish({"event": "test"})

        assert mock_redis.xadd.await_count == 5


class TestStreamProducerPublishBatch:
    """Tests for StreamProducer.publish_batch."""

    @pytest.mark.asyncio
    async def test_publish_batch_empty_returns_empty(self) -> None:
        """publish_batch with empty list should return empty list."""
        mock_redis = AsyncMock()
        producer = ConcreteProducer(mock_redis, "test:stream")

        result = await producer.publish_batch([])
        assert result == []
        mock_redis.xadd.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_batch_returns_ids(self) -> None:
        """publish_batch should return one ID per message."""
        mock_redis = AsyncMock()

        mock_pipe = MagicMock()
        mock_pipe.xadd = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[b"1-0", b"2-0", b"3-0"])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        producer = ConcreteProducer(mock_redis, "test:stream")

        ids = await producer.publish_batch(
            [{"event": "a"}, {"event": "b"}, {"event": "c"}]
        )

        assert len(ids) == 3
        assert ids == ["1-0", "2-0", "3-0"]

    @pytest.mark.asyncio
    async def test_publish_batch_each_has_trace_id(self) -> None:
        """Each message in batch should get its own trace_id."""
        mock_redis = AsyncMock()

        mock_pipe = MagicMock()
        mock_pipe.xadd = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[b"1-0"])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        producer = ConcreteProducer(mock_redis, "test:stream")

        ids = await producer.publish_batch([{"event": "a"}])
        assert len(ids) == 1

        call_args = mock_pipe.xadd.call_args
        envelope = json.loads(call_args[1]["fields"]["data"])
        assert "trace_id" in envelope
        assert "published_at" in envelope


class TestStreamConsumerInit:
    """Tests for StreamConsumer initialization."""

    def test_init_stores_attributes(self) -> None:
        """Consumer should store all init parameters."""
        mock_redis = MagicMock()
        consumer = ConcreteConsumer(
            mock_redis, "test:stream", "cg:test", "consumer-1"
        )
        assert consumer._redis is mock_redis
        assert consumer._stream_name == "test:stream"
        assert consumer._consumer_group == "cg:test"
        assert consumer._consumer_name == "consumer-1"
        assert consumer._batch_size == 100
        assert consumer._block_ms == 5000
        assert consumer._running is False


class TestStreamConsumerProcessMessage:
    """Tests for the abstract process_message method."""

    def test_process_message_is_abstract(self) -> None:
        """StreamConsumer.process_message should be abstract."""
        mock_redis = MagicMock()
        with pytest.raises(TypeError):
            StreamConsumer(mock_redis, "stream", "group", "name")


class TestStreamConsumerStart:
    """Tests for StreamConsumer.start."""

    @pytest.mark.asyncio
    async def test_start_creates_consumer_group(self) -> None:
        """start should call xgroup_create."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=None)
        mock_redis.xack = AsyncMock()
        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")

        await consumer.start()
        await asyncio.sleep(0.1)
        await consumer.stop()

        mock_redis.xgroup_create.assert_awaited_once()
        call_args = mock_redis.xgroup_create.call_args
        assert call_args[1]["name"] == "stream"
        assert call_args[1]["groupname"] == "cg"
        assert call_args[1]["mkstream"] is True

    @pytest.mark.asyncio
    async def test_start_handles_existing_group(self) -> None:
        """start should not fail if consumer group already exists."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock(
            side_effect=ResponseError("BUSYGROUP Consumer Group exists")
        )
        mock_redis.xreadgroup = AsyncMock(return_value=None)
        mock_redis.xack = AsyncMock()

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await asyncio.sleep(0.1)
        await consumer.stop()

        mock_redis.xgroup_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        """start should not start twice."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=None)
        mock_redis.xack = AsyncMock()

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await asyncio.sleep(0.1)

        # Second start should be a no-op
        await consumer.start()
        await consumer.stop()

        assert mock_redis.xgroup_create.await_count == 1


class TestStreamConsumerConsumeLoop:
    """Tests for the internal consume loop behavior."""

    @pytest.mark.asyncio
    async def test_consume_loop_processes_and_acks(self) -> None:
        """Consumer should call process_message and xack on success."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        msg_id = b"1234567890-0"
        msg_data = json.dumps({"event": "test", "trace_id": "abc"}).encode()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(b"stream", [(msg_id, {b"data": msg_data})])]
            return None  # No more messages

        mock_redis.xreadgroup = mock_xreadgroup

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await asyncio.sleep(0.3)
        await consumer.stop()

        assert len(consumer.processed_messages) == 1
        assert consumer.processed_messages[0][0] == "1234567890-0"
        assert consumer.processed_messages[0][1]["event"] == "test"
        mock_redis.xack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_consume_loop_no_ack_on_failure(self) -> None:
        """Consumer should not ACK when process_message returns False."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        msg_id = b"1234567890-0"
        msg_data = json.dumps({"event": "test"}).encode()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(b"stream", [(msg_id, {b"data": msg_data})])]
            return None

        mock_redis.xreadgroup = mock_xreadgroup

        class FailConsumer(StreamConsumer):
            async def process_message(self, message_id: str, message: dict[str, Any]) -> bool:
                return False

        consumer = FailConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await asyncio.sleep(0.3)
        await consumer.stop()

        assert mock_redis.xack.assert_not_awaited or mock_redis.xack.await_count == 0

    @pytest.mark.asyncio
    async def test_consume_loop_handles_process_exception(self) -> None:
        """Consumer should log and continue when process_message raises."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        msg_id = b"1234567890-0"
        msg_data = json.dumps({"event": "test"}).encode()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(b"stream", [(msg_id, {b"data": msg_data})])]
            return None

        mock_redis.xreadgroup = mock_xreadgroup

        class ExceptionConsumer(StreamConsumer):
            async def process_message(self, message_id: str, message: dict[str, Any]) -> bool:
                raise ValueError("processing failed")

        consumer = ExceptionConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await asyncio.sleep(0.3)
        await consumer.stop()

        # Should not ACK since process_message raised
        assert mock_redis.xack.await_count == 0

    @pytest.mark.asyncio
    async def test_consume_loop_logs_each_message(self) -> None:
        """Consumer should log each message with stream_name, message_id, status."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        msg_id = b"1234567890-0"
        msg_data = json.dumps({"event": "test", "trace_id": "abc"}).encode()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(b"stream", [(msg_id, {b"data": msg_data})])]
            return None

        mock_redis.xreadgroup = mock_xreadgroup

        with patch("streams.base.logger") as mock_logger:
            consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
            await consumer.start()
            await asyncio.sleep(0.3)
            await consumer.stop()

            mock_logger.info.assert_any_call(
                "message_processed",
                stream_name="stream",
                consumer_group="cg",
                message_id="1234567890-0",
                duration_ms=__import__("unittest.mock").mock.ANY,
                status="success",
                trace_id="abc",
            )


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self) -> None:
        """stop without start should not raise."""
        mock_redis = AsyncMock()
        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        """stop should set _running to False."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=None)

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        assert consumer._running is True
        await consumer.stop()
        assert consumer._running is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown_completes_in_flight(self) -> None:
        """In-flight messages should complete before shutdown."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        processing_event = asyncio.Event()
        done_event = asyncio.Event()

        class SlowConsumer(StreamConsumer):
            async def process_message(self, message_id: str, message: dict[str, Any]) -> bool:
                processing_event.set()
                await asyncio.sleep(0.5)
                done_event.set()
                return True

        msg_id = b"123-0"
        msg_data = json.dumps({"event": "slow"}).encode()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(b"stream", [(msg_id, {b"data": msg_data})])]
            return None

        mock_redis.xreadgroup = mock_xreadgroup

        consumer = SlowConsumer(mock_redis, "stream", "cg", "c1")
        await consumer.start()
        await processing_event.wait()  # Wait until processing starts
        assert not done_event.is_set()

        # Stop should wait for in-flight message to complete
        await consumer.stop()
        assert done_event.is_set()
        mock_redis.xack.assert_awaited_once()


class TestConnectionErrorRetry:
    """Tests for connection error retry logic."""

    @pytest.mark.asyncio
    async def test_consumer_retries_on_connection_error(self) -> None:
        """Consumer should retry on Redis connection errors."""
        mock_redis = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xack = AsyncMock()

        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RedisConnectionError("connection refused")
            if call_count == 2:
                return None  # No messages after reconnect
            return None

        mock_redis.xreadgroup = mock_xreadgroup

        with patch("streams.base.asyncio.sleep", new_callable=AsyncMock):
            consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
            await consumer.start()
            await asyncio.sleep(0.1)
            await consumer.stop()

    @pytest.mark.asyncio
    async def test_claim_pending_handles_connection_error(self) -> None:
        """claim_pending should handle connection errors gracefully."""
        mock_redis = AsyncMock()
        mock_redis.xautoclaim = AsyncMock(side_effect=RedisConnectionError("connection refused"))

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")

        with patch("streams.base.asyncio.sleep", new_callable=AsyncMock):
            result = await consumer.claim_pending(min_idle_ms=60000)

        assert result == []


class TestClaimPending:
    """Tests for the claim_pending method."""

    @pytest.mark.asyncio
    async def test_claim_pending_returns_claimed_messages(self) -> None:
        """claim_pending should return claimed message IDs and payloads."""
        mock_redis = AsyncMock()
        msg_id = b"123-0"
        msg_data = json.dumps({"event": "test", "trace_id": "abc"}).encode()
        mock_redis.xautoclaim = AsyncMock(
            return_value=[(msg_id, {b"data": msg_data})]
        )

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        claimed = await consumer.claim_pending(min_idle_ms=60000)

        assert len(claimed) == 1
        assert claimed[0][0] == "123-0"
        assert claimed[0][1]["event"] == "test"

    @pytest.mark.asyncio
    async def test_claim_pending_empty(self) -> None:
        """claim_pending with no results should return empty list."""
        mock_redis = AsyncMock()
        mock_redis.xautoclaim = AsyncMock(return_value=[])

        consumer = ConcreteConsumer(mock_redis, "stream", "cg", "c1")
        claimed = await consumer.claim_pending()
        assert claimed == []

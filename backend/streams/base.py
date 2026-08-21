"""Abstract base classes for Redis Streams producers and consumers.

Provides :class:`StreamProducer` for publishing messages to Redis Streams
and :class:`StreamConsumer` for consuming messages from consumer groups
with automatic retry, graceful shutdown, and pending-message claiming.

All Redis operations are wrapped with exponential-backoff retry logic to
handle transient ``ConnectionError`` failures.
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import redis.asyncio as redis_asyncio
import structlog
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 1.0


async def _with_redis_retry(
    operation: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an async Redis operation with exponential-backoff retry.

    Retries up to :data:`_MAX_RETRIES` times with base delay
    :data:`_BASE_DELAY_SECONDS` and exponential backoff (1s, 2s, 4s, 8s,
    16s).

    Args:
        operation: An async callable that interacts with Redis.
        *args: Positional arguments forwarded to *operation*.
        **kwargs: Keyword arguments forwarded to *operation*.

    Returns:
        The return value of *operation*.

    Raises:
        redis.exceptions.ConnectionError: If all retries are exhausted.
    """
    last_exc: RedisConnectionError | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await operation(*args, **kwargs)
        except RedisConnectionError as exc:
            last_exc = exc
            delay = _BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "redis_retry",
                attempt=attempt + 1,
                max_retries=_MAX_RETRIES,
                delay=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected retry exhaustion without exception")


class StreamProducer(ABC):  # noqa: B024
    """Abstract base class for Redis Streams producers.

    Subclasses should set a meaningful ``stream_name`` and may override
    :meth:`_build_envelope` for custom message formatting.
    """

    def __init__(self, redis: redis_asyncio.Redis, stream_name: str) -> None:
        """Initialise the producer.

        Args:
            redis: A connected :class:`redis.asyncio.Redis` client.
            stream_name: The Redis Streams key to publish to.
        """
        self._redis = redis
        self._stream_name = stream_name
        logger.debug(
            "producer_initialized",
            stream_name=stream_name,
            redis_host=getattr(redis, "connection_pool", None),
        )

    @property
    def stream_name(self) -> str:
        """Return the stream name this producer publishes to."""
        return self._stream_name

    def _build_envelope(
        self, message: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build the message envelope with trace_id and published_at.

        Args:
            message: The original message payload.
            metadata: Optional metadata to merge into the envelope.

        Returns:
            A new dict with the merged message, metadata, trace_id, and
            published_at fields.
        """
        envelope: dict[str, Any] = dict(message)

        if metadata:
            envelope.update(metadata)

        if "trace_id" not in envelope:
            envelope["trace_id"] = str(uuid4())

        envelope["published_at"] = datetime.now(UTC).isoformat()

        return envelope

    async def publish(self, message: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
        """Publish a single message to the Redis stream.

        The message is serialized to JSON and stored with ``trace_id`` and
        ``published_at`` fields added automatically.

        Args:
            message: The message payload to publish.
            metadata: Optional metadata to merge into the message.

        Returns:
            The Redis-generated message ID as a string.

        Raises:
            redis.exceptions.ConnectionError: If Redis is unreachable after
                all retries.
        """
        envelope = self._build_envelope(message, metadata)
        serialized = json.dumps(envelope, default=str)

        start = time.monotonic()

        async def _do_publish() -> bytes:
            message_id: bytes = await self._redis.xadd(
                name=self._stream_name,
                fields={"data": serialized},
                id="*",
                approximate=False,
            )
            return message_id

        message_id_bytes = await _with_redis_retry(_do_publish)
        message_id = (
            message_id_bytes.decode()
            if isinstance(message_id_bytes, bytes)
            else str(message_id_bytes)
        )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        message_size = len(serialized)

        logger.info(
            "message_published",
            stream_name=self._stream_name,
            message_id=message_id,
            trace_id=envelope.get("trace_id"),
            message_size_bytes=message_size,
            duration_ms=elapsed_ms,
        )

        return message_id

    async def publish_batch(self, messages: list[dict[str, Any]]) -> list[str]:
        """Publish a batch of messages to the Redis stream.

        Uses a Redis pipeline for efficiency.  Each message is independently
        serialized with ``trace_id`` and ``published_at``.

        Args:
            messages: List of message payloads to publish.

        Returns:
            A list of Redis message IDs, one per input message.

        Raises:
            redis.exceptions.ConnectionError: If Redis is unreachable after
                all retries.
        """
        if not messages:
            return []

        start = time.monotonic()
        envelopes = [self._build_envelope(msg) for msg in messages]
        serialized_list = [json.dumps(env, default=str) for env in envelopes]
        total_size = sum(len(s) for s in serialized_list)

        async def _do_publish_batch() -> list[bytes]:
            pipe = self._redis.pipeline()
            for serialized in serialized_list:
                pipe.xadd(
                    name=self._stream_name,
                    fields={"data": serialized},
                    id="*",
                    approximate=False,
                )
            return await pipe.execute()

        results = await _with_redis_retry(_do_publish_batch)

        message_ids: list[str] = []
        for res in results:
            if isinstance(res, bytes):
                message_ids.append(res.decode())
            else:
                message_ids.append(str(res))

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        logger.info(
            "batch_published",
            stream_name=self._stream_name,
            message_count=len(messages),
            message_ids=message_ids,
            total_size_bytes=total_size,
            duration_ms=elapsed_ms,
        )

        return message_ids


class StreamConsumer(ABC):
    """Abstract base class for Redis Streams consumers.

    Subclasses must implement :meth:`process_message`.  The consumer
    automatically creates a consumer group, claims stale pending messages,
    and handles graceful shutdown on SIGTERM/SIGINT.
    """

    def __init__(
        self,
        redis: redis_asyncio.Redis,
        stream_name: str,
        consumer_group: str,
        consumer_name: str,
        batch_size: int = 100,
        block_ms: int = 5000,
    ) -> None:
        """Initialise the consumer.

        Args:
            redis: A connected :class:`redis.asyncio.Redis` client.
            stream_name: The Redis Streams key to consume from.
            consumer_group: The consumer group name.
            consumer_name: This consumer's name within the group.
            batch_size: Maximum messages to process per iteration.
            block_ms: How long to block on ``xreadgroup`` when idle (ms).
        """
        self._redis = redis
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._signal_handlers_installed: bool = False

        logger.debug(
            "consumer_initialized",
            stream_name=stream_name,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
            batch_size=batch_size,
            block_ms=block_ms,
        )

    @abstractmethod
    async def process_message(self, message_id: str, message: dict[str, Any]) -> bool:
        """Process a single message from the stream.

        Subclasses must implement this.  Return ``True`` to acknowledge
        the message, ``False`` to leave it pending for redelivery.

        Args:
            message_id: The Redis message ID.
            message: The deserialized message payload.

        Returns:
            ``True`` if the message was processed successfully (will be
            ACKed), ``False`` to skip the ACK.
        """
        ...

    async def _ensure_consumer_group(self) -> None:
        """Create the consumer group if it does not already exist."""

        async def _create() -> None:
            try:
                await self._redis.xgroup_create(
                    name=self._stream_name,
                    groupname=self._consumer_group,
                    id="0",
                    mkstream=True,
                )
                logger.info(
                    "consumer_group_created",
                    stream_name=self._stream_name,
                    consumer_group=self._consumer_group,
                )
            except ResponseError as exc:
                if "BUSYGROUP" in str(exc):
                    logger.debug(
                        "consumer_group_exists",
                        stream_name=self._stream_name,
                        consumer_group=self._consumer_group,
                    )
                else:
                    raise

        await _with_redis_retry(_create)

    async def _consume_loop(self) -> None:
        """Internal consume loop — reads, processes, and ACKs messages."""
        logger.info(
            "consume_loop_started",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
        )

        while self._running:
            try:
                messages = await _with_redis_retry(
                    self._redis.xreadgroup,
                    self._consumer_group,
                    self._consumer_name,
                    {self._stream_name: ">"},
                    self._batch_size,
                    self._block_ms,
                )
            except RedisConnectionError as exc:
                logger.error(
                    "consume_loop_redis_error",
                    stream_name=self._stream_name,
                    error=str(exc),
                )
                if not self._running:
                    break
                continue
            except Exception as exc:
                logger.error(
                    "consume_loop_error",
                    stream_name=self._stream_name,
                    error=str(exc),
                )
                if not self._running:
                    break
                continue

            if not messages:
                if not self._running:
                    break
                await asyncio.sleep(0.05)
                continue

            for stream_result in messages:
                _stream_name_bytes = stream_result[0]
                msg_list = stream_result[1]

                for raw_msg_id, raw_fields in msg_list:
                    msg_id = (
                        raw_msg_id.decode() if isinstance(raw_msg_id, bytes) else str(raw_msg_id)
                    )
                    data_field = raw_fields.get(b"data", raw_fields.get("data", b""))

                    if isinstance(data_field, bytes):
                        data_field = data_field.decode()

                    try:
                        message = json.loads(data_field)
                    except (json.JSONDecodeError, TypeError):
                        message = {"raw": data_field}

                    start = time.monotonic()

                    try:
                        success = await self.process_message(msg_id, message)
                    except Exception as exc:
                        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
                        logger.error(
                            "message_processing_failed",
                            stream_name=self._stream_name,
                            consumer_group=self._consumer_group,
                            message_id=msg_id,
                            duration_ms=elapsed_ms,
                            error=str(exc),
                            trace_id=message.get("trace_id"),
                        )
                        continue

                    elapsed_ms = round((time.monotonic() - start) * 1000, 2)

                    if success:
                        await _with_redis_retry(
                            self._redis.xack, self._stream_name, self._consumer_group, msg_id
                        )
                        logger.info(
                            "message_processed",
                            stream_name=self._stream_name,
                            consumer_group=self._consumer_group,
                            message_id=msg_id,
                            duration_ms=elapsed_ms,
                            status="success",
                            trace_id=message.get("trace_id"),
                        )
                    else:
                        logger.warning(
                            "message_processing_failed",
                            stream_name=self._stream_name,
                            consumer_group=self._consumer_group,
                            message_id=msg_id,
                            duration_ms=elapsed_ms,
                            status="skipped",
                            trace_id=message.get("trace_id"),
                        )

        logger.info(
            "consume_loop_stopped",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
        )

    async def start(self) -> None:
        """Create the consumer group and start the background consume loop.

        Raises:
            RuntimeError: If ``start`` has already been called.
        """
        if self._running:
            logger.warning("consumer_already_running", stream_name=self._stream_name)
            return

        await self._ensure_consumer_group()
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

        self._install_signal_handlers()

        logger.info(
            "consumer_started",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
        )

    def _install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown.

        Only registers on Unix platforms where
        :func:`asyncio.AbstractEventLoop.add_signal_handler` is available.
        """
        if self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except (NotImplementedError, RuntimeError):
                pass

        self._signal_handlers_installed = True

    async def stop(self) -> None:
        """Stop the consumer gracefully.

        Sets the ``_running`` flag to ``False`` and waits for the consume
        loop task to complete.  Any messages being processed will finish
        before the loop exits.
        """
        if not self._running:
            return

        self._running = False
        logger.info(
            "stopping_consumer",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
        )

        if self._task is not None:
            await self._task
            self._task = None

        # Claim any remaining pending messages for this consumer
        await self.claim_pending(min_idle_ms=60000)

        logger.info(
            "consumer_stopped",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
        )

    async def claim_pending(self, min_idle_ms: int = 60000) -> list[tuple[str, dict[str, Any]]]:
        """Claim pending messages that have been idle longer than *min_idle_ms*.

        Uses ``XAUTOCLAIM`` to transfer ownership of stale pending messages
        to this consumer.

        Args:
            min_idle_ms: Minimum idle time (ms) before a message is eligible
                for claiming.

        Returns:
            A list of ``(message_id, message)`` tuples that were claimed.
        """

        async def _do_claim() -> Any:
            return await self._redis.xautoclaim(
                name=self._stream_name,
                groupname=self._consumer_group,
                consumername=self._consumer_name,
                min_idle_time=min_idle_ms,
                start_id="0-0",
            )

        try:
            result = await _with_redis_retry(_do_claim)
        except RedisConnectionError as exc:
            logger.warning(
                "claim_pending_redis_error",
                stream_name=self._stream_name,
                error=str(exc),
            )
            return []

        claimed: list[tuple[str, dict[str, Any]]] = []

        if result:
            for item in result:
                msg_id: str
                fields: dict[str, Any]

                if isinstance(item, tuple) and len(item) >= 2:
                    msg_id_raw, msg_fields = item[0], item[1]
                    if isinstance(msg_id_raw, bytes):
                        msg_id = msg_id_raw.decode()
                    else:
                        msg_id = str(msg_id_raw)

                    data_val = msg_fields.get(b"data", msg_fields.get("data", b""))
                    if isinstance(data_val, bytes):
                        data_val = data_val.decode()

                    try:
                        fields = json.loads(data_val)
                    except (json.JSONDecodeError, TypeError):
                        fields = {"raw": data_val}

                    claimed.append((msg_id, fields))

        if claimed:
            logger.info(
                "pending_messages_claimed",
                stream_name=self._stream_name,
                consumer_group=self._consumer_group,
                consumer_name=self._consumer_name,
                count=len(claimed),
            )

        return claimed

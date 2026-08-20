"""Ingestion worker that processes raw data through adapters and publishes structured signals.

Consumes from Redis Streams, runs the full adapter pipeline (fetch → parse → fingerprint →
normalize → dedup → insert → publish), and handles graceful shutdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Import from ingestion base
from ingestion.base import (
    AdapterNotFoundError,
    AdapterRegistry,
    IngestionResult,
    NormalizedSignal,
    RawSignal,
    SourceAdapter,
)

# Import DB dependencies
try:
    from db.dependencies import get_session_manager
    from db.models import Signal, Tenant
    from db.tenant_session import TenantSessionManager

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    get_session_manager = None
    TenantSessionManager = None
    Signal = None
    Tenant = None

# Redis
try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# MinIO/S3 - lazy import
def _get_aiobotocore():
    try:
        import aiobotocore.session
        from aiobotocore.client import AioBaseClient
        return aiobotocore.session, AioBaseClient, True
    except ImportError:
        return None, None, False


AIOBOTOCORE_AVAILABLE = False
AioBaseClient = None

logger = structlog.get_logger(__name__)


class IngestionWorker:
    """Worker that processes ingestion jobs from Redis Streams.

    Pipeline:
    1. Consume message from tenant ingestion stream
    2. Get adapter from registry
    3. Fetch raw data
    4. Parse to RawSignal
    5. Fingerprint for dedup
    6. Check database for existing fingerprint
    7. Normalize (upload to MinIO, create pointers)
    8. Insert into signals table (RLS-aware)
    9. Publish to signals stream for downstream
    10. ACK message
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        consumer_group: str = "cg:ingestion_worker",
        consumer_name: str = "ingestion-worker-1",
        batch_size: int = 10,
        block_ms: int = 5000,
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
        minio_region: str = "us-east-1",
        minio_secure: bool = False,
    ):
        self.redis_url = redis_url
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.batch_size = batch_size
        self.block_ms = block_ms

        # MinIO config
        self.minio_endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.minio_region = minio_region
        self.minio_secure = minio_secure

        # Runtime
        self._redis: Optional[aioredis.Redis] = None
        self._s3_client: Optional[AioBaseClient] = None
        self._session_manager: Optional[TenantSessionManager] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _ensure_redis(self) -> aioredis.Redis:
        if self._redis is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError("redis not installed")
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            # Ensure consumer group exists
            await self._create_consumer_groups()
        return self._redis

    async def _create_consumer_groups(self) -> None:
        """Create consumer groups for all ingestion streams."""
        # We'll create groups dynamically when processing each stream
        pass

    async def _ensure_s3(self) -> AioBaseClient:
        if self._s3_client is None:
            if not AIOBOTOCORE_AVAILABLE:
                raise RuntimeError("aiobotocore not installed")
            session = aiobotocore.session.get_session()
            self._s3_client = await session.create_client(
                "s3",
                endpoint_url=f"http{'s' if self.minio_secure else ''}://{self.minio_endpoint}",
                aws_access_key_id=self.minio_access_key,
                aws_secret_access_key=self.minio_secret_key,
                region_name=self.minio_region,
            ).__aenter__()
        return self._s3_client

    async def _ensure_db(self) -> TenantSessionManager:
        if self._session_manager is None:
            if not DB_AVAILABLE:
                raise RuntimeError("db dependencies not available")
            self._session_manager = get_session_manager()
            if not self._session_manager._initialized:
                await self._session_manager.connect()
        return self._session_manager

    async def _get_ingestion_streams(self, tenant_id: str) -> list[str]:
        """Get list of ingestion streams for a tenant."""
        from streams.keys import StreamKeyBuilder
        key_builder = StreamKeyBuilder()
        # In production, we'd get this from config
        source_types = ["web", "sec", "jobs", "patents", "earnings", "github"]
        return [key_builder.ingestion_stream(tenant_id, st) for st in source_types]

    async def start(self) -> None:
        """Start the ingestion worker."""
        if self._running:
            return

        self._running = True
        redis = await self._ensure_redis()

        # Start consumption loop for all tenant streams
        # For simplicity, we use a single consumer group and read from all streams
        # In production, you might have one worker per tenant or per source type
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("ingestion_worker_started", consumer_group=self.consumer_group)

    async def stop(self) -> None:
        """Stop the ingestion worker gracefully."""
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("worker_shutdown_timeout", consumer_name=self.consumer_name)
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        if self._redis:
            await self._redis.aclose()
            self._redis = None
        if self._s3_client:
            await self._s3_client.__aexit__(None, None, None)
            self._s3_client = None

        logger.info("ingestion_worker_stopped", consumer_name=self.consumer_name)

    async def _consume_loop(self) -> None:
        """Main consumption loop - reads from all tenant ingestion streams."""
        redis = await self._ensure_redis()

        while self._running:
            try:
                # Use XREAD with multiple streams
                streams = await self._get_all_tenant_streams()
                if not streams:
                    await asyncio.sleep(5)
                    continue

                # Build stream map for XREAD
                stream_map = {stream: ">" for stream in streams}

                results = await redis.xread(
                    streams=stream_map,
                    count=self.batch_size,
                    block=self.block_ms,
                )

                if not results:
                    continue

                for stream_name, messages in results:
                    for msg_id, msg_data in messages:
                        try:
                            await self._process_message(stream_name, msg_id, msg_data)
                        except Exception as e:
                            logger.error(
                                "message_processing_failed",
                                stream=stream_name,
                                message_id=msg_id,
                                error=str(e),
                            )
                            # Don't ACK failed messages - they'll be retried

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consume_loop_error", error=str(e))
                await asyncio.sleep(5)

    async def _get_all_tenant_streams(self) -> list[str]:
        """Get all ingestion streams across all tenants."""
        # In production, this would query tenant_configs or use a pattern
        # For now, we use a pattern-based approach
        redis = await self._ensure_redis()
        # Scan for ingestion streams
        streams = []
        async for key in redis.scan_iter(match="stratops:tenant:*:ingestion:*"):
            streams.append(key)
        return streams

    async def _process_message(
        self, stream_name: str, message_id: str, message_data: dict[str, str]
    ) -> None:
        """Process a single ingestion message.

        Message format:
        {
            "tenant_id": "uuid",
            "source_type": "web",
            "adapter_name": "web_monitor",
            "config": {...},
            "trace_id": "uuid"
        }
        """
        start_time = time.monotonic()
        trace_id = message_data.get("trace_id", hashlib.md5(message_id.encode()).hexdigest()[:8])

        logger.info(
            "processing_ingestion_message",
            stream=stream_name,
            message_id=message_id,
            trace_id=trace_id,
        )

        # Parse message
        try:
            tenant_id = message_data["tenant_id"]
            source_type = message_data["source_type"]
            adapter_name = message_data.get("adapter_name", source_type)
            config = json.loads(message_data.get("config", "{}"))
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("invalid_message_format", message_id=message_id, error=str(e))
            await self._ack_message(stream_name, message_id)
            return

        # Get adapter
        try:
            adapter_class = AdapterRegistry.get(adapter_name)
            adapter = adapter_class()
        except AdapterNotFoundError as e:
            logger.error("adapter_not_found", adapter=adapter_name, error=str(e))
            await self._ack_message(stream_name, message_id)
            return

        # Get DB session
        session_manager = await self._ensure_db()

        try:
            async with session_manager.get_session(tenant_id) as session:
                # Step 1: Fetch
                logger.debug("fetching", trace_id=trace_id, adapter=adapter_name)
                result: IngestionResult = await adapter.fetch(config)

                # Step 2: Parse
                logger.debug("parsing", trace_id=trace_id)
                raw_signals: list[RawSignal] = await adapter.parse(result.raw_data, result.content_type)

                # Add tenant_id to metadata for all signals
                for sig in raw_signals:
                    sig.metadata["tenant_id"] = tenant_id

                # Step 3: Fingerprint + Dedup
                logger.debug("fingerprinting", trace_id=trace_id, count=len(raw_signals))
                fingerprinted: list[RawSignal] = []
                for sig in raw_signals:
                    fp = sig.fingerprint or await adapter.fingerprint(sig)
                    sig.fingerprint = fp

                    # Check if fingerprint exists in DB
                    existing = await session.execute(
                        select(Signal.id).where(
                            Signal.tenant_id == tenant_id,
                            Signal.fingerprint == fp,
                        )
                    )
                    if existing.scalar_one_or_none() is None:
                        fingerprinted.append(sig)
                    else:
                        logger.debug("duplicate_fingerprint_skipped", fingerprint=fp[:16])

                if not fingerprinted:
                    logger.info("all_signals_duplicate", trace_id=trace_id)
                    await self._ack_message(stream_name, message_id)
                    return

                # Step 4: Normalize (uploads to MinIO)
                logger.debug("normalizing", trace_id=trace_id, count=len(fingerprinted))
                normalized: list[NormalizedSignal] = await adapter.normalize(fingerprinted)

                # Step 5: Insert into signals table
                logger.debug("inserting_signals", trace_id=trace_id, count=len(normalized))
                for norm in normalized:
                    signal = Signal(
                        tenant_id=tenant_id,
                        source_type=norm.source_type,
                        source_url=norm.source_url,
                        content_uri=norm.content_uri,
                        fingerprint=norm.fingerprint,
                        structured_payload=norm.structured_payload,
                        collected_at=norm.collected_at,
                        meta=norm.metadata,
                    )
                    session.add(signal)

                await session.commit()

                # Step 6: Publish to signals stream
                await self._publish_to_signals_stream(tenant_id, normalized, trace_id)

                duration_ms = (time.monotonic() - start_time) * 1000
                logger.info(
                    "ingestion_complete",
                    trace_id=trace_id,
                    tenant_id=tenant_id,
                    source_type=source_type,
                    signals_processed=len(normalized),
                    duration_ms=round(duration_ms, 2),
                )

        except Exception as e:
            logger.error("ingestion_pipeline_failed", trace_id=trace_id, error=str(e))
            raise

        # ACK message on success
        await self._ack_message(stream_name, message_id)

    async def _ack_message(self, stream_name: str, message_id: str) -> None:
        """Acknowledge message in Redis Stream."""
        redis = await self._ensure_redis()
        try:
            await redis.xack(stream_name, self.consumer_group, message_id)
        except Exception as e:
            logger.warning("ack_failed", stream=stream_name, message_id=message_id, error=str(e))

    async def _publish_to_signals_stream(
        self, tenant_id: str, signals: list[NormalizedSignal], trace_id: str
    ) -> None:
        """Publish normalized signals to the signals stream for downstream processing."""
        from streams.keys import StreamKeyBuilder
        from streams.base import StreamProducer

        redis = await self._ensure_redis()
        key_builder = StreamKeyBuilder()
        stream_name = key_builder.signal_stream(tenant_id)

        producer = StreamProducer(redis, stream_name)

        for signal in signals:
            await producer.publish(
                {
                    "signal_id": signal.fingerprint[:16],
                    "tenant_id": tenant_id,
                    "source_type": signal.source_type,
                    "source_url": signal.source_url,
                    "content_uri": signal.content_uri,
                    "fingerprint": signal.fingerprint,
                    "structured_payload": signal.structured_payload,
                    "collected_at": signal.collected_at.isoformat(),
                    "metadata": signal.metadata,
                },
                trace_id=trace_id,
            )


@asynccontextmanager
async def create_ingestion_worker(**kwargs: Any) -> IngestionWorker:
    """Context manager for creating and running an ingestion worker."""
    worker = IngestionWorker(**kwargs)
    try:
        await worker.start()
        yield worker
    finally:
        await worker.stop()


async def run_ingestion_worker(**kwargs: Any) -> None:
    """Run ingestion worker until cancelled."""
    async with create_ingestion_worker(**kwargs) as worker:
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
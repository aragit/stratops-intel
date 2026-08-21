"""Entity extraction LangGraph node - pointer-only state.

This is the FIRST LangGraph node in the intelligence pipeline.
CRITICAL: pointer-only state. S3/MinIO URIs and small structured data only.
Raw payloads NEVER in state. Checkpoint target < 5KB.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

# aiobotocore imported lazily inside methods to avoid OpenSSL compatibility issues
# in the test environment (botocore.translate -> botocore.utils -> urllib3.contrib.pyopenssl -> OpenSSL)
# where _lib.GEN_EMAIL was removed from newer Python versions.
import structlog
from langgraph.graph import StateGraph

logger = structlog.get_logger(__name__)


class IntelligenceState(TypedDict):
    """Pointer-only state for the intelligence extraction pipeline.

    CRITICAL: No raw content in state. Only S3 URIs and small structured data.
    Checkpoint target < 5KB total.
    """

    tenant_id: str
    trace_id: str
    signal_uris: list[str]  # S3 URIs of signals to process
    extracted_entities: list[dict]  # Small structured data only
    content_uris: list[str]  # New URIs for extracted content
    correlation_graph_delta: list[str]  # Neo4j relationship strings
    briefing_section_uris: list[str]  # S3 URIs for briefing sections


class EntityExtractorNode:
    """LangGraph node that extracts entities from signals using vLLM."""

    def __init__(self) -> None:
        self.bentoml_base = os.getenv("BENTOML_EXTRACTION_URL", "http://bentoml-extraction:3000")

    async def _download_from_minio(self, s3_uri: str) -> str:
        """Download content from MinIO via aiobotocore.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/key)

        Returns:
            Raw text content downloaded from MinIO
        """
        import aiobotocore.session

        session = aiobotocore.session.get_session()

        # Parse s3://bucket/key format
        uri_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = uri_parts[0]
        key = uri_parts[1] if len(uri_parts) > 1 else ""

        async with session.create_client("s3", region_name="us-east-1") as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            content = await response["Body"].read()
            return str(content.decode("utf-8"))

    async def _call_bentoml_extraction(
        self, texts: list[str], schema_name: str, tenant_id: str
    ) -> list[dict]:
        """Call BentoML extraction service via HTTP (LiteLLM proxy or direct).

        Args:
            texts: List of text chunks to extract from
            schema_name: Extraction schema name
            tenant_id: Tenant identifier

        Returns:
            List of extracted results
        """
        import httpx

        payload = {
            "texts": texts,
            "schema_name": schema_name,
            "tenant_id": tenant_id,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.bentoml_base}/v1/extract",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                # The service returns list of dicts with metadata
                return data if isinstance(data, list) else [data]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logger.warning(
                    "bentoml_oom_error",
                    tenant_id=tenant_id,
                )
            raise
        except Exception as e:
            logger.error(
                "bentoml_call_failed",
                error=str(e),
                tenant_id=tenant_id,
            )
            raise

    async def _upload_to_minio(self, bucket: str, key: str, content: str) -> str:
        """Upload extracted entities JSON to MinIO.

        Args:
            bucket: MinIO bucket name
            key: S3 key (e.g., entities.json)
            content: JSON string to upload

        Returns:
            S3 URI of the uploaded content
        """
        import aiobotocore.session

        session = aiobotocore.session.get_session()

        uri_parts = bucket.replace("s3://", "").split("/", 1)
        bucket_name = uri_parts[0]

        async with session.create_client("s3", region_name="us-east-1") as client:
            await client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType="application/json",
            )

        return f"s3://{bucket}/{key}"

    async def __call__(self, state: IntelligenceState) -> IntelligenceState:
        """Extract entities from all signals in the state.

        CRITICAL: pointer-only - reads URIs from state, downloads content,
        calls extraction service, writes results to MinIO, returns new URIs.

        Args:
            state: IntelligenceState with signal_uris, tenant_id, trace_id

        Returns:
            Updated IntelligenceState with extracted entities and new URIs
        """
        start_time = time.time()
        tenant_id = state["tenant_id"]
        trace_id = state["trace_id"]
        signal_uris = state.get("signal_uris", [])

        logger = structlog.get_logger()
        logger = logger.bind(trace_id=trace_id, tenant_id=tenant_id, signal_count=len(signal_uris))

        if not signal_uris:
            logger.warning("no_signals_to_process")
            return state

        # Initialize results
        all_extracted_entities: list[dict] = []
        new_content_uris: list[str] = []
        correlation_deltas: list[str] = []
        entity_count = 0  # Initialize before loop

        # Process each signal URI
        for signal_idx, signal_uri in enumerate(signal_uris):
            signal_start = time.time()

            try:
                # Download content from MinIO (each method manages its own session)
                content = await self._download_from_minio(signal_uri)

                # Call BentoML extraction service
                # Use "auto" schema - service will determine best schema
                extraction_results = await self._call_bentoml_extraction(
                    texts=[content],
                    schema_name="auto",
                    tenant_id=tenant_id,
                )

                # Parse extracted entities from results
                for result in extraction_results:
                    result_data = result.get("result", result)
                    if isinstance(result_data, dict):
                        # Look for entities key, otherwise use the dict itself
                        entities = result_data.get("entities", result_data)
                        if isinstance(entities, list):
                            all_extracted_entities.extend(entities)
                        elif isinstance(entities, dict):
                            all_extracted_entities.append(entities)
                    elif isinstance(result_data, list):
                        all_extracted_entities.extend(result_data)

                entity_count = len(all_extracted_entities)

                # Build content URI for extracted entities
                entity_key = f"{trace_id}/signal_{signal_idx}_entities.json"
                _entity_uri = f"s3://stratops-extracted-{tenant_id}/{entity_key}"

                # Serialize entities to JSON and upload to MinIO
                # Cap at ~10KB to keep state small
                recent_entities = all_extracted_entities[-entity_count:] if entity_count else []
                entities_json = json.dumps({"entities": recent_entities}, default=str)[:10000]

                uploaded_uri = await self._upload_to_minio(
                    bucket=f"stratops-extracted-{tenant_id}",
                    key=entity_key,
                    content=entities_json,
                )

                new_content_uris.append(uploaded_uri)

                # Build correlation graph delta (Neo4j relationship strings)
                for entity in recent_entities:
                    entity_type = entity.get("company_name", entity.get("name", "unknown"))
                    delta = f"extracted:{entity_type}:{trace_id}:{signal_idx}"
                    correlation_deltas.append(delta)

                signal_duration = (time.time() - signal_start) * 1000
                logger.info(
                    "signal_processed",
                    signal_idx=signal_idx,
                    entity_count=entity_count,
                    duration_ms=round(signal_duration, 2),
                )

            except Exception as e:
                logger.error(
                    "signal_processing_failed",
                    signal_uri=signal_uri,
                    error=str(e),
                    trace_id=trace_id,
                    signal_idx=signal_idx,
                )
                # Continue processing other signals even if one fails
                continue

        # Build new state - pointer-only, no raw content
        new_state: IntelligenceState = {
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "signal_uris": state.get("signal_uris", []),
            "extracted_entities": all_extracted_entities,
            "content_uris": state.get("content_uris", []) + new_content_uris,
            "correlation_graph_delta": state.get("correlation_graph_delta", [])
            + correlation_deltas,
            "briefing_section_uris": state.get("briefing_section_uris", []),
        }

        # CRITICAL: Verify state size < 5KB
        state_size_bytes = len(json.dumps(new_state).encode("utf-8"))
        if state_size_bytes > 5000:
            logger.warning(
                "state_exceeds_checkpoint_limit",
                state_size_bytes=state_size_bytes,
                limit=5000,
                tenant_id=tenant_id,
            )

        total_duration = (time.time() - start_time) * 1000
        logger.info(
            "extraction_complete",
            tenant_id=tenant_id,
            trace_id=trace_id,
            total_duration_ms=round(total_duration, 2),
            entity_count=entity_count,
            signal_count=len(signal_uris),
        )

        return new_state


def build_extractor_graph() -> Any:
    """Build and return the LangGraph extraction pipeline.

    Creates a StateGraph with IntelligenceState and the EntityExtractorNode.
    Compiles with MemorySaver checkpoint (SQLite for now, Postgres later).

    Returns:
        Compiled StateGraph ready for use
    """
    workflow = StateGraph(IntelligenceState)
    workflow.add_node("extract", EntityExtractorNode().__call__)
    workflow.set_entry_point("extract")
    compiled = workflow.compile()  # MemorySaver used by default
    return compiled  # type: ignore[return-value]

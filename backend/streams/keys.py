"""Tenant-aware Redis Stream key generator.

Provides :class:`StreamKeyBuilder` for constructing deterministic,
namespaced Redis Stream keys scoped to individual tenants.

All keys follow the pattern ``{namespace}:tenant:{tenant_id}:{type}[:{subtype}]``
to ensure logical isolation at the Redis key level in addition to
consumer-group-level isolation.
"""

from __future__ import annotations

from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class StreamKeyBuilder:
    """Builds tenant-scoped Redis Stream key names.

    Every key is prefixed with a configurable namespace (default
    ``stratops``) and includes the tenant UUID to enforce logical
    key-space separation.

    Example::

        builder = StreamKeyBuilder()
        key = builder.signal_stream(UUID("11111111-..."))
        # -> "stratops:tenant:11111111-...:signals"
    """

    def __init__(self, base_namespace: str = "stratops") -> None:
        """Initialise the key builder.

        Args:
            base_namespace: The top-level namespace prefix for all keys.
                Defaults to ``"stratops"``.
        """
        self._namespace = base_namespace
        logger.debug("stream_key_builder_initialized", namespace=base_namespace)

    @property
    def namespace(self) -> str:
        """Return the base namespace."""
        return self._namespace

    @staticmethod
    def _validate_tenant_id(tenant_id: UUID) -> None:
        """Validate that tenant_id is a UUID instance.

        Args:
            tenant_id: The value to validate.

        Raises:
            ValueError: If *tenant_id* is not a :class:`UUID`.
        """
        if not isinstance(tenant_id, UUID):
            raise ValueError(
                f"tenant_id must be a UUID, got {type(tenant_id).__name__}"
            )

    def signal_stream(self, tenant_id: UUID) -> str:
        """Return the signal stream key for a tenant.

        Args:
            tenant_id: The tenant's UUID.

        Returns:
            A Redis Stream key string.
        """
        self._validate_tenant_id(tenant_id)
        return f"{self._namespace}:tenant:{tenant_id}:signals"

    def ingestion_stream(self, tenant_id: UUID, source_type: str) -> str:
        """Return the ingestion stream key for a tenant and source type.

        Args:
            tenant_id: The tenant's UUID.
            source_type: The ingestion source identifier (e.g. ``"rss"``,
                ``"api"``, ``"webhook"``).

        Returns:
            A Redis Stream key string.
        """
        self._validate_tenant_id(tenant_id)
        if not source_type:
            raise ValueError("source_type must not be empty")
        return f"{self._namespace}:tenant:{tenant_id}:ingestion:{source_type}"

    def intelligence_stream(self, tenant_id: UUID, agent_type: str) -> str:
        """Return the intelligence stream key for a tenant and agent type.

        Args:
            tenant_id: The tenant's UUID.
            agent_type: The intelligence agent identifier (e.g.
                ``"signal"``, ``"briefing"``).

        Returns:
            A Redis Stream key string.
        """
        self._validate_tenant_id(tenant_id)
        if not agent_type:
            raise ValueError("agent_type must not be empty")
        return f"{self._namespace}:tenant:{tenant_id}:intelligence:{agent_type}"

    def alert_stream(self, tenant_id: UUID) -> str:
        """Return the alert stream key for a tenant.

        Args:
            tenant_id: The tenant's UUID.

        Returns:
            A Redis Stream key string.
        """
        self._validate_tenant_id(tenant_id)
        return f"{self._namespace}:tenant:{tenant_id}:alerts"

    def graph_writer_buffer(self, tenant_id: UUID) -> str:
        """Return the Neo4j graph writer buffer key for a tenant.

        This stream is consumed by the Redis-buffered micro-batcher
        that writes to Neo4j via ``UNWIND ... MERGE``.

        Args:
            tenant_id: The tenant's UUID.

        Returns:
            A Redis Stream key string.
        """
        self._validate_tenant_id(tenant_id)
        return f"{self._namespace}:tenant:{tenant_id}:graph:pending"

    def consumer_group(self, service_name: str) -> str:
        """Return a consumer group name for a given service.

        Args:
            service_name: The name of the consuming service (e.g.
                ``"ingestion"``, ``"intelligence"``).

        Returns:
            A consumer group name string.
        """
        if not service_name:
            raise ValueError("service_name must not be empty")
        return f"cg:{service_name}"

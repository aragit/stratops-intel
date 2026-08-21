"""Source adapter protocol, plugin registry, and core ingestion types.

This module defines the abstract interface that all data source adapters must
implement, plus the registry for managing adapter plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, NamedTuple

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class SourceConfig(BaseModel):
    """Base configuration class for source adapters.

    Each adapter subclass should define its own config schema
    inheriting from this base class.
    """

    model_config = ConfigDict(extra="forbid")


class AdapterNotFoundError(KeyError):
    """Raised when an adapter is not found in the registry."""

    pass


class AdapterRegistrationError(ValueError):
    """Raised when adapter registration fails (e.g., duplicate name)."""

    pass


class IngestionResult(NamedTuple):
    """Result of a fetch operation from a source adapter.

    Attributes:
        raw_data: The raw bytes returned by the source.
        content_type: MIME type of the raw data (e.g., "text/html", "application/xml").
        next_cursor: Opaque cursor for pagination, or None if no more data.
        metadata: Additional metadata about the fetch operation.
    """

    raw_data: bytes
    content_type: str
    next_cursor: str | None
    metadata: dict[str, Any]


class RawSignal(BaseModel):
    """Raw signal extracted from a source, before normalization and dedup.

    This contains the raw payload which will be stored in object storage (MinIO),
    not in PostgreSQL. Only the fingerprint and metadata go to the database.

    Attributes:
        source_type: The adapter's source_type (e.g., "web", "sec").
        source_url: Original URL or identifier for the source.
        raw_content: Raw payload bytes (stored in MinIO, NOT in DB).
        fingerprint: Content hash for deduplication (SimHash, SHA-256, etc.).
        collected_at: When the signal was collected.
        metadata: Additional adapter-specific metadata.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    source_type: str = Field(..., description="Adapter source type identifier")
    source_url: str | None = Field(default=None, description="Original source URL")
    raw_content: bytes = Field(..., description="Raw payload (goes to MinIO)")
    fingerprint: str | None = Field(default=None, description="Dedup fingerprint")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Adapter-specific metadata")


class NormalizedSignal(BaseModel):
    """Normalized signal ready for database storage and downstream processing.

    CRITICAL: This uses POINTER-ONLY architecture. The raw content is stored in
    MinIO/S3 and only the URI is stored in the database. Raw payloads NEVER go
    into PostgreSQL.

    Attributes:
        source_type: The adapter's source type (e.g., "web", "sec").
        source_url: Original URL or identifier for the source.
        content_uri: S3/MinIO URI pointing to raw content (REQUIRED).
        fingerprint: Content hash for deduplication (REQUIRED).
        structured_payload: Extracted structured metadata (title, author, date, etc.).
        collected_at: When the signal was collected.
        metadata: Additional adapter-specific metadata.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    source_type: str = Field(..., description="Adapter source type identifier")
    source_url: str | None = Field(default=None, description="Original source URL")
    content_uri: str = Field(..., description="S3/MinIO URI (pointer-only, REQUIRED)")
    fingerprint: str = Field(..., description="Dedup fingerprint (REQUIRED)")
    structured_payload: dict[str, Any] = Field(
        default_factory=dict, description="Structured extracted metadata"
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="Collection timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Adapter-specific metadata")


class SourceAdapter(ABC):
    """Abstract base class for all source adapters.

    Each adapter implements the full ingestion pipeline for a specific
    data source: fetch → parse → fingerprint → normalize.

    Class attributes:
        name: Unique identifier for this adapter (e.g., "web_monitor").
        source_type: Category of source (e.g., "web", "sec", "jobs").
        config_schema: Pydantic model defining the adapter's configuration.
    """

    name: ClassVar[str]
    source_type: ClassVar[str]
    config_schema: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name") or not cls.name:
            raise TypeError(f"{cls.__name__} must define a non-empty 'name' class attribute")
        if not hasattr(cls, "source_type") or not cls.source_type:
            raise TypeError(f"{cls.__name__} must define a non-empty 'source_type' class attribute")
        if not hasattr(cls, "config_schema") or not cls.config_schema:
            raise TypeError(f"{cls.__name__} must define a 'config_schema' class attribute")

    @abstractmethod
    async def fetch(self, config: dict[str, Any], cursor: str | None = None) -> IngestionResult:
        """Fetch raw data from the source.

        Args:
            config: Adapter configuration validated against config_schema.
            cursor: Opaque pagination cursor from previous fetch, or None for first page.

        Returns:
            IngestionResult with raw data, content type, next cursor, and metadata.

        Raises:
            Exception: Any fetch error (network, auth, rate limit, etc.).
        """
        ...

    @abstractmethod
    async def parse(self, raw_data: bytes, content_type: str) -> list[RawSignal]:
        """Parse raw data into a list of RawSignal objects.

        Args:
            raw_data: Raw bytes from fetch().
            content_type: MIME type from fetch() result.

        Returns:
            List of RawSignal objects (one per logical item in the raw data).
        """
        ...

    @abstractmethod
    async def fingerprint(self, signal: RawSignal) -> str:
        """Compute a deduplication fingerprint for a signal.

        Args:
            signal: RawSignal to fingerprint.

        Returns:
            Fingerprint string (SimHash hex, SHA-256 hex, etc.).
        """
        ...

    @abstractmethod
    async def normalize(self, signals: list[RawSignal]) -> list[NormalizedSignal]:
        """Normalize raw signals to NormalizedSignal with object storage pointers.

        This method MUST:
        1. Upload each signal's raw_content to MinIO/S3
        2. Set content_uri to the S3 URI
        3. Extract structured metadata into structured_payload
        4. Ensure fingerprint is populated

        Args:
            signals: List of RawSignal objects from parse().

        Returns:
            List of NormalizedSignal with content_uri pointers.
        """
        ...


class AdapterRegistry:
    """Registry for managing source adapter plugins.

    Adapters register themselves via @AdapterRegistry.register or by calling
    register() explicitly. The registry maintains a mapping from adapter name
    to adapter class, and provides lookup by name or source_type.
    """

    _adapters: dict[str, type[SourceAdapter]] = {}
    _by_source_type: dict[str, list[str]] = {}

    @classmethod
    def register(cls, adapter_class: type[SourceAdapter]) -> None:
        """Register an adapter class.

        Args:
            adapter_class: Subclass of SourceAdapter to register.

        Raises:
            AdapterRegistrationError: If an adapter with the same name is already registered.
        """
        name = adapter_class.name
        if name in cls._adapters:
            raise AdapterRegistrationError(f"Adapter '{name}' already registered")

        cls._adapters[name] = adapter_class
        cls._by_source_type.setdefault(adapter_class.source_type, []).append(name)
        logger.info("adapter_registered", name=name, source_type=adapter_class.source_type)

    @classmethod
    def get(cls, name: str) -> type[SourceAdapter]:
        """Get an adapter class by name.

        Args:
            name: Adapter name (e.g., "web_monitor").

        Returns:
            The adapter class.

        Raises:
            AdapterNotFoundError: If no adapter with that name is registered.
        """
        if name not in cls._adapters:
            raise AdapterNotFoundError(f"Adapter '{name}' not found. Available: {list(cls._adapters.keys())}")
        return cls._adapters[name]

    @classmethod
    def list_adapters(cls) -> list[str]:
        """Return list of all registered adapter names."""
        return list(cls._adapters.keys())

    @classmethod
    def list_by_source_type(cls, source_type: str) -> list[str]:
        """Return adapter names for a given source type."""
        return cls._by_source_type.get(source_type, [])

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (for testing)."""
        cls._adapters.clear()
        cls._by_source_type.clear()
        logger.debug("adapter_registry_cleared")


def register_adapter(adapter_class: type[SourceAdapter]) -> type[SourceAdapter]:
    """Decorator to register an adapter class.

    Usage:
        @register_adapter
        class MyAdapter(SourceAdapter):
            ...
    """
    AdapterRegistry.register(adapter_class)
    return adapter_class

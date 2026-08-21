"""Ingestion engine package.

Exports the SourceAdapter protocol, AdapterRegistry, and core ingestion types.
"""

from __future__ import annotations

from .base import (
    AdapterNotFoundError,
    AdapterRegistrationError,
    AdapterRegistry,
    IngestionResult,
    NormalizedSignal,
    RawSignal,
    SourceAdapter,
    register_adapter,
)

__all__ = [
    "SourceAdapter",
    "AdapterRegistry",
    "RawSignal",
    "NormalizedSignal",
    "IngestionResult",
    "AdapterNotFoundError",
    "AdapterRegistrationError",
    "register_adapter",
]

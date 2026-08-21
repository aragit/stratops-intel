"""Adapter package - auto-registers built-in adapters on import."""

from __future__ import annotations


# Lazy import to avoid OpenSSL conflicts
def _register_adapters():
    try:
        from . import (
            sec,  # noqa: F401
            web,  # noqa: F401
        )
    except Exception:
        pass

_register_adapters()

from ..base import (
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

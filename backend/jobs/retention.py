"""Automated data retention engine for tenant-tier data lifecycle management.

Enforces retention policies on raw signals and old intelligence chunks,
deleting records older than the tier-defined threshold (Free: 90 days,
Pro: 365 days). Returns counts of purged records per table for audit
visibility.

All retention operations are tenant-scoped and async-aware.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Retention limits by tier (Free / Pro / Enterprise)
#: Free tier: 90 days, Pro tier: 365 days, Enterprise: 2555 days (7 years)
TIER_RETENTION_DAYS: dict[str, int] = {
    "free": 90,
    "pro": 365,
    "enterprise": 2555,
}


class RetentionEngine:
    """Enforce tenant-tier data retention policies on raw signals and
    intelligence chunks.

    The engine queries PostgreSQL for records older than the retention
    threshold and deletes them, returning counts of purged records per
    table for audit logging.

    Attributes:
        postgres: Async SQLAlchemy engine (or compatible) connected to the
            StratOps-Intel database.
        tenant_id: Tenant identifier for multi-tenant partitioning.
    """

    def __init__(self, postgres: Any, tenant_id: str) -> None:
        """Initialize the retention engine.

        Args:
            postgres: Async SQLAlchemy engine (or compatible) connected to
                the StratOps-Intel database.
            tenant_id: Tenant identifier for data partitioning.
        """
        self._postgres = postgres
        self._tenant_id = tenant_id

    async def _signal_table(self) -> str:
        """Return the signals table name for the current tenant.

        Tenant-partitioned tables are named ``signals_{tenant_id}``.
        """
        return f"signals_{self._tenant_id}"

    async def _intelligence_chunk_table(self) -> str:
        """Return the intelligence_chunks table name for the current tenant."""
        return f"intelligence_chunks_{self._tenant_id}"

    async def _cutoff_datetime(self, retention_days: int) -> datetime:
        """Compute the cutoff datetime (records older than this will be purged)."""
        now = datetime.now(UTC)
        return now - timedelta(days=retention_days)

    async def purge_expired_data(self, retention_days: int | None = None) -> dict[str, int]:
        """Purge signals and intelligence chunks older than the retention threshold.

        The default retention period is determined by the tenant's tier:
        ``free`` → 90 days, ``pro`` → 365 days, ``enterprise`` → 2555 days.

        Args:
            retention_days: Override the default retention window in days.
                If ``None``, the tier-appropriate limit is used.

        Returns:
            Dict with counts of purged records per table:
            ``{"signals": int, "intelligence_chunks": int}``.
        """
        # Determine retention period
        if retention_days is None:
            tier = await self._get_tier()
            retention_days = TIER_RETENTION_DAYS.get(tier, 90)

        cutoff = await self._cutoff_datetime(retention_days)

        signal_table = await self._signal_table()
        chunk_table = await self._intelligence_chunk_table()

        # Purge expired signals
        purge_signal_sql = (
            f"DELETE FROM {signal_table} WHERE created_at < :cutoff AND tenant_id = :tenant_id"
        )
        signal_result = await self._postgres.execute(
            purge_signal_sql, {"cutoff": cutoff, "tenant_id": self._tenant_id}
        )
        # PostgreSQL DELETE RETURNING returns the count of deleted rows
        # depending on the driver; we capture the rowcount.
        purged_signals = signal_result.rowcount if hasattr(signal_result, "rowcount") else 0

        # Purge expired intelligence chunks
        purge_chunk_sql = (
            f"DELETE FROM {chunk_table} WHERE created_at < :cutoff AND tenant_id = :tenant_id"
        )
        chunk_result = await self._postgres.execute(
            purge_chunk_sql, {"cutoff": cutoff, "tenant_id": self._tenant_id}
        )
        purged_chunks = chunk_result.rowcount if hasattr(chunk_result, "rowcount") else 0

        logger.info(
            "retention_purge_complete",
            tenant_id=self._tenant_id,
            retention_days=retention_days,
            cutoff=cutoff.isoformat(),
            purged_signals=purged_signals,
            purged_chunks=purged_chunks,
        )

        return {"signals": purged_signals, "intelligence_chunks": purged_chunks}

    async def _get_tier(self) -> str:
        """Determine the tenant's tier from the billing module.

        Falls back to ``free`` if the tier cannot be determined.
        """
        # Import here to avoid circular dependencies at module load time

        # We need a Redis connection to query the tier; for the retention
        # engine we simply read from the Redis key that CostTracker sets.
        # The retention job will typically have Redis available via the
        # application context; if not, we default to "free".
        try:
            import aioredis

            redis = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
            tier = await redis.get(f"billing:tenant:{self._tenant_id}:tier")
            if tier is None:
                return "free"
            return tier.decode("utf-8") if isinstance(tier, bytes) else tier
        except Exception:
            return "free"

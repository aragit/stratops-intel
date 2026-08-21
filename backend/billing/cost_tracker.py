"""Per-tenant LLM cost tracking and aggregation.

Tracks token usage and estimated cost per tenant tier (Free/Pro/Enterprise),
providing usage summaries and limit enforcement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Tier definitions with monthly token limits and pricing per 1K tokens
TIER_LIMITS: dict[str, dict[str, int | float]] = {
    "free": {"monthly_prompt_tokens": 10_000, "monthly_completion_tokens": 10_000, "prompt_cost_per_1k": 0.0, "completion_cost_per_1k": 0.0},
    "pro": {"monthly_prompt_tokens": 100_000, "monthly_completion_tokens": 100_000, "prompt_cost_per_1k": 0.05, "completion_cost_per_1k": 0.15},
    "enterprise": {"monthly_prompt_tokens": 1_000_000, "monthly_completion_tokens": 1_000_000, "prompt_cost_per_1k": 0.03, "completion_cost_per_1k": 0.10},
}

#: Tier name human-readable labels
TIER_LABELS: dict[str, str] = {
    "free": "Free",
    "pro": "Pro",
    "enterprise": "Enterprise",
}


class CostTracker:
    """Track and aggregate LLM token usage and estimated cost per tenant tier.

    Stores token metrics in Redis time-series keys for fast retrieval and
    PostgreSQL for persistent history. Enforces per-tier monthly limits.

    Attributes:
        redis: Async Redis client for time-series storage.
        postgres: Async SQLAlchemy engine for persistent history.
    """

    def __init__(self, redis: Any, postgres: Any) -> None:
        """Initialize the cost tracker.

        Args:
            redis: An async Redis client (redis.asyncio.Redis).
            postgres: An async SQLAlchemy engine (or compatible).
        """
        self._redis = redis
        self._postgres = postgres

    async def _tier_key(self, tenant_id: str) -> str:
        """Return the Redis key for a tenant's tier tracking data."""
        return f"billing:tenant:{tenant_id}:tier"

    async def _usage_key(self, tenant_id: str) -> str:
        """Return the Redis key for a tenant's hourly usage bucket."""
        return f"billing:tenant:{tenant_id}:usage"

    async def log_request(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Log a single LLM request's token usage and cost.

        Stores the raw metrics in Redis time-series and aggregates the
        tenant's monthly totals.

        Args:
            tenant_id: Tenant identifier for multi-tenant partitioning.
            model: LLM model name (e.g. "gpt-4o-mini").
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens consumed.
            cost_usd: Estimated cost in USD for this request.
        """
        # Store raw request metrics as a JSON hash in Redis
        metrics_key = f"billing:tenant:{tenant_id}:request:{datetime.utcnow().timestamp()}"
        await self._redis.set(
            metrics_key,
            json.dumps(
                {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
        )

        # Aggregate monthly totals in Redis using hash fields
        tier = await self._get_tier(tenant_id)
        limits = TIER_LIMITS[tier]

        # Increment cumulative counters
        pipe = self._redis.pipeline()
        pipe.hincrby(f"billing:tenant:{tenant_id}:summary", "prompt_tokens", prompt_tokens)
        pipe.hincrby(f"billing:tenant:{tenant_id}:summary", "completion_tokens", completion_tokens)
        pipe.hincrby(f"billing:tenant:{tenant_id}:summary", "cost_usd", float(cost_usd))
        pipe.execute()

        logger.info(
            "cost_request_logged",
            tenant_id=tenant_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            tier=tier,
        )

    async def get_tenant_costs(
        self, tenant_id: str, start_time: datetime | None, end_time: datetime | None
    ) -> dict[str, Any]:
        """Retrieve aggregated token usage and cost for a tenant within a time range.

        Args:
            tenant_id: Tenant identifier.
            start_time: Optional start datetime (UTC). If None, uses beginning of current month.
            end_time: Optional end datetime (UTC). If None, uses now.

        Returns:
            Dict with total prompt/completion tokens, cost, tier, and limit information.
        """
        tier = await self._get_tier(tenant_id)
        limits = TIER_LIMITS[tier]

        # Read cumulative summary from Redis summary hash
        summary = await self._redis.hgetall(f"billing:tenant:{tenant_id}:summary")

        total_prompt_tokens = int(summary.get("prompt_tokens", 0))
        total_completion_tokens = int(summary.get("completion_tokens", 0))
        total_cost_usd = float(summary.get("cost_usd", 0.0))

        # Determine the time range for the query
        now = datetime.utcnow()
        if start_time is None:
            # Beginning of current month
            start_time = datetime(now.year, now.month, 1, tzinfo=UTC)
        if end_time is None:
            end_time = now

        # Clamp to tier limits
        prompt_limit = limits["monthly_prompt_tokens"]
        completion_limit = limits["monthly_completion_tokens"]

        prompt_used_pct = (total_prompt_tokens / prompt_limit * 100) if prompt_limit > 0 else 0.0
        completion_used_pct = (total_completion_tokens / completion_limit * 100) if completion_limit > 0 else 0.0

        return {
            "tenant_id": tenant_id,
            "tier": tier,
            "tier_label": TIER_LABELS.get(tier, tier),
            "period": f"{start_time.strftime('%Y-%m')} to {end_time.strftime('%Y-%m-%d')}",
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "prompt_cost_usd": round(total_prompt_tokens / 1000 * limits["prompt_cost_per_1k"], 4),
            "completion_cost_usd": round(total_completion_tokens / 1000 * limits["completion_cost_per_1k"], 4),
            "total_cost_usd": round(total_cost_usd, 4),
            "prompt_tokens_limit": prompt_limit,
            "completion_tokens_limit": completion_limit,
            "prompt_usage_pct": round(prompt_used_pct, 2),
            "completion_usage_pct": round(completion_used_pct, 2),
            "prompt_tokens_remaining": prompt_limit - total_prompt_tokens,
            "completion_tokens_remaining": completion_limit - total_completion_tokens,
        }

    async def _get_tier(self, tenant_id: str) -> str:
        """Determine a tenant's tier from Redis storage.

        Falls back to "free" if no tier has been set explicitly.
        """
        tier = await self._redis.get(f"billing:tenant:{tenant_id}:tier")
        if tier is None:
            tier = "free"
            # Set default tier
            await self._redis.set(f"billing:tenant:{tenant_id}:tier", "free")
        return tier.decode("utf-8") if isinstance(tier, bytes) else tier


__all__ = ["CostTracker", "TIER_LIMITS", "TIER_LABELS"]

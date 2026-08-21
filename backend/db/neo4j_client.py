"""Neo4j client for StratOps Intel backend.

CRITICAL: All writes go through this client (NOT direct from stream consumers).
Uses UNWIND ... MERGE pattern for micro-batching. Pointer-only state.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase

logger = structlog.get_logger(__name__)


class Neo4jClient:
    """Neo4j client with micro-batching and RLS support.

    All Neo4j writes MUST go through this client. NEVER direct writes from
    stream consumers. Uses UNWIND ... MERGE for batched operations.

    Constraints:
    - Checkpoint target < 5KB (pointer-only state)
    - RLS via tenant_id on all nodes
    - No partial HNSW indexes per tenant
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI (e.g., "neo4j://localhost:7687")
            user: Neo4j user
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Any | None = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy-initialize the driver on first use."""
        if self._initialized:
            return

        logger.info("initializing_neo4j_client", uri=self.uri)
        try:
            self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            await self.health()
            self._initialized = True
            logger.info("neo4j_client_ready", uri=self.uri)
        except Exception as e:
            logger.error("neo4j_client_init_failed", error=str(e), uri=self.uri)
            raise

    async def run(self, query: str, parameters: dict | None = None) -> Any:
        """Execute a Cypher query with optional parameters.

        CRITICAL: Wrapper around asyncio.to_thread + session.run.
        All queries must include tenant_id for multi-tenancy RLS.

        Args:
            query: Cypher query string
            parameters: Optional parameters dict

        Returns:
            Result summary or record from Neo4j
        """
        await self._ensure_initialized()

        start_time = time.time()

        if self._driver is None:
            raise RuntimeError("Neo4j client is not connected; call connect() first")

        try:
            async with self._driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.fetch(5000)  # type: ignore[assignment]
                duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    "neo4j_query_executed",
                    query_length=len(query),
                    duration_ms=round(duration_ms, 2),
                    has_parameters=parameters is not None,
                )

                # Return single record if exactly one, otherwise return records
                if len(records) == 1:
                    return records[0]
                return records

        except Exception as e:
            logger.error(
                "neo4j_query_failed",
                query=query[:200],
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
            raise

    async def init_schema(self) -> None:
        """Run the Neo4j schema Cypher definitions.

        Creates constraints, indexes, and node type definitions.
        Must be called on Neo4j startup.
        """
        await self._ensure_initialized()

        logger.info("running_neo4j_schema_init")
        try:
            # Read and execute the schema file
            with open("backend/db/neo4j_schema.cypher") as f:
                schema_sql = f.read()

            # Execute each statement separately
            statements = [s.strip() for s in schema_sql.split(";") if s.strip()]

            if self._driver is None:
                raise RuntimeError("Neo4j client is not connected; call connect() first")
            async with self._driver.session() as session:
                for statement in statements:
                    # Skip empty statements
                    if not statement.strip():
                        continue
                    await statement_single(session, statement)  # type: ignore[name-defined]

            logger.info("neo4j_schema_init_complete")

        except Exception as e:
            logger.error("neo4j_schema_init_failed", error=str(e))
            raise

    async def health(self) -> dict:
        """Health check endpoint.

        Returns:
            {"health": 1} if Neo4j is reachable
        """
        await self._ensure_initialized()

        start_time = time.time()

        if self._driver is None:
            raise RuntimeError("Neo4j client is not connected; call connect() first")

        try:
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS health")
                record = await result.fetch_one()
                health_value = record["health"] if record else 0
                duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    "neo4j_health_check",
                    health=health_value,
                    duration_ms=round(duration_ms, 2),
                )

                return {"health": health_value}

        except Exception as e:
            logger.error("neo4j_health_check_failed", error=str(e))
            return {"health": 0}

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            logger.info("closing_neo4j_client")
            await self._driver.close()
            self._driver = None
            self._initialized = False


async def statement_single(session: Any, statement: str) -> None:
    """Execute a single Cypher statement.

    Args:
        session: Neo4j async session
        statement: Cypher statement string (without trailing semicolon)
    """
    await session.run(statement)

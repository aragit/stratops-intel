"""Unit tests for the Neo4j client.

Mocks the Neo4j async driver and tests schema initialization,
health checks, and query execution.
"""

from __future__ import annotations

from unittest import mock

import pytest

from backend.db.neo4j_client import Neo4jClient


class TestNeo4jClient:
    """Tests for the Neo4jClient class."""

    @pytest.fixture
    def neo4j_client(self) -> Neo4jClient:
        """Provide a Neo4jClient instance for tests."""
        return Neo4jClient(uri="neo4j://localhost:7687", user="neo4j", password="test")

    @pytest.mark.asyncio
    async def test_health_check(self, neo4j_client: Neo4jClient) -> None:
        """Test Neo4j health check returns {"health": 1} when connected."""
        # Mock the driver and async session context manager
        mock_driver = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.run = mock.AsyncMock()
        mock_session.run.return_value.fetch_one = mock.AsyncMock(
            return_value={"health": 1}
        )
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        result = await neo4j_client.health()

        assert result == {"health": 1}
        mock_driver.session.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failed(self, neo4j_client: Neo4jClient) -> None:
        """Test Neo4j health check returns {"health": 0} on failure."""
        mock_driver = mock.MagicMock()
        mock_driver.session.side_effect = Exception("Connection failed")
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        result = await neo4j_client.health()

        assert result == {"health": 0}

    @pytest.mark.asyncio
    async def test_run_query(self, neo4j_client: Neo4jClient) -> None:
        """Test executing a Cypher query with parameters."""
        mock_driver = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.run = mock.AsyncMock()
        # fetch(5000) returns multiple records -> client returns the list
        mock_session.run.return_value.fetch = mock.AsyncMock(
            return_value=[{"name": "A"}, {"name": "B"}]
        )
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        query = "CREATE (n:Person {name: $name, tenant_id: $tenant_id})"
        parameters = {"name": "Test Person", "tenant_id": "00000000-0000-0000-0000-000000000001"}

        result = await neo4j_client.run(query, parameters)

        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once_with(query, parameters)
        assert result == [{"name": "A"}, {"name": "B"}]

    @pytest.mark.asyncio
    async def test_run_query_no_parameters(self, neo4j_client: Neo4jClient) -> None:
        """Query without parameters is executed with an empty dict."""
        mock_driver = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.run = mock.AsyncMock()
        mock_session.run.return_value.fetch = mock.AsyncMock(
            return_value=[{"ok": True}]
        )
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        query = "CREATE (n:Person {name: 'Test', tenant_id: $tid})"

        result = await neo4j_client.run(query)

        # Production coerces missing parameters to {}
        mock_session.run.assert_called_once_with(query, {})
        # Single record is returned unwrapped
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_init_schema(self, neo4j_client: Neo4jClient) -> None:
        """Schema init splits the cypher file and executes each statement."""
        mock_driver = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        schema_content = "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE; CREATE INDEX i1 IF NOT EXISTS FOR (n:Company) ON (n.tenant_id);"

        with mock.patch(
            "builtins.open", mock.mock_open(read_data=schema_content)
        ):
            with mock.patch(
                "backend.db.neo4j_client.statement_single"
            ) as mock_exec:
                mock_exec.side_effect = mock.AsyncMock(return_value=None)
                await neo4j_client.init_schema()

        assert mock_exec.await_count == 2

    @pytest.mark.asyncio
    async def test_close(self, neo4j_client: Neo4jClient) -> None:
        """Test closing the Neo4j driver."""
        mock_driver = mock.AsyncMock()
        neo4j_client._driver = mock_driver
        await neo4j_client.close()
        mock_driver.close.assert_awaited_once()
        assert neo4j_client._driver is None

    @pytest.mark.asyncio
    async def test_run_without_init_fails(self, neo4j_client: Neo4jClient) -> None:
        """Test that running a query without initialization raises an error."""
        # neo4j_client is not initialized
        with pytest.raises(Exception):
            await neo4j_client.run("RETURN 1")

    @pytest.mark.asyncio
    async def test_health_without_init_fails(self, neo4j_client: Neo4jClient) -> None:
        """Test that health check without initialization raises an error."""
        with pytest.raises(Exception):
            await neo4j_client.health()

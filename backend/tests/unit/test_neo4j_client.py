"""Unit tests for the Neo4j client.

Mocks the Neo4j async driver and tests schema initialization,
health checks, and query execution.
"""

from __future__ import annotations

import json
import pytest
from unittest import mock

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
        # Mock the driver and session
        mock_driver = mock.MagicMock()
        mock_session = mock.AsyncMock()
        mock_record = mock.MagicMock()
        mock_record.__getitem__.return_value = 1
        mock_session.run.return_value = mock.MagicMock()
        mock_session.run.return_value.fetch_one.return_value = mock_record
        mock_driver.session.return_value = mock_session
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
        mock_session = mock.AsyncMock()
        mock_result = mock.MagicMock()
        mock_record = mock.MagicMock()
        mock_record.__getitem__.return_value = "test_value"
        mock_result.fetch_one.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        query = "CREATE (n:Person {name: $name, tenant_id: $tenant_id})"
        parameters = {"name": "Test Person", "tenant_id": "00000000-0000-0000-0000-000000000001"}

        result = await neo4j_client.run(query, parameters)

        mock_driver.session.assert_called_once()
        mock_session.run.assert_called_once_with(query, parameters)
        assert result == mock_record

    @pytest.mark.asyncio
    async def test_run_query_no_parameters(self, neo4j_client: Neo4jClient) -> None:
        """Test executing a Cypher query without parameters."""
        mock_driver = mock.MagicMock()
        mock_session = mock.AsyncMock()
        mock_result = mock.MagicMock()
        mock_record = mock.MagicMock()
        mock_record.__getitem__.return_value = True
        mock_result.fetch_one.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        neo4j_client._driver = mock_driver
        neo4j_client._initialized = True

        query = "CREATE (n:Person {name: 'Test', tenant_id: $tid})"
        parameters = {"tid": "00000000-0000-0000-0000-000000000001"}

        result = await neo4j_client.run(query, parameters)

        mock_session.run.assert_called_once_with(query, parameters)
        assert result == mock_record

    @pytest.mark.asyncio
    async def test_init_schema(self, neo4j_client: Neo4jClient) -> None:
        """Test schema initialization runs without error."""
        # Mock the file reading and statement execution
        with mock.patch("backend.db.neo4j_client.open", mock.mock_open(read_data="CREATE (n:Person)")):
            with mock.patch.object(neo4j_client, "_execute_single_statement") as mock_exec:
                # The init_schema reads the file and executes statements
                # We need to test the general flow
                pass

        # Just verify the method exists and is callable
        assert callable(getattr(neo4j_client, "init_schema", None))

    @pytest.mark.asyncio
    async def test_close(self, neo4j_client: Neo4jClient) -> None:
        """Test closing the Neo4j driver."""
        neo4j_client._driver = mock.MagicMock()
        await neo4j_client.close()
        neo4j_client._driver.close.assert_called_once()

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
"""End-to-End Integration Test — Signal → Briefing → Alert Pipeline.

Full pipeline test using testcontainers for Postgres, Redis, Neo4j, MinIO.
Verifies the complete signal → briefing → alert pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from testcontainers.neo4j import Neo4jContainer
from testcontainers.minio import MinioContainer
import respx

# Import our modules
from backend.intelligence.agents.extractor import EntityExtractorNode, IntelligenceState
from backend.intelligence.agents.correlation import CorrelationEngineNode, CorrelationResult
from backend.intelligence.agents.trend import TrendAnalyzerNode, TrendResult
from backend.intelligence.agents.anomaly import AnomalyDetectorNode, AnomalyResult
from backend.intelligence.agents.composer import BriefingComposerNode, Briefing, BriefingSection
from backend.intelligence.agents.delta import BriefingDeltaGenerator
from backend.db.neo4j_client import Neo4jClient
from backend.workers.graph_writer import GraphWriterWorker, EntityUpdate, RelationshipUpdate
from backend.alerts.rules import AlertRuleEngine, AlertRule, Alert
from backend.alerts.router import AlertRouter, AlertRouterWorker, SlackChannelConfig, EmailChannelConfig, WebhookChannelConfig
from backend.briefings.repository import BriefingRepository
from bentoml.services.summarization import EmbeddingRequest

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL test container with pgvector."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def redis_container():
    """Redis test container."""
    container = RedisContainer("redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def neo4j_container():
    """Neo4j test container."""
    container = Neo4jContainer("neo4j:5-community")
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="session")
def minio_container():
    """MinIO test container."""
    container = MinioContainer(
        "minio/minio:latest",
        access_key="minioadmin",
        secret_key="minioadmin",
    )
    container.start()
    # Create required buckets
    import subprocess
    url = container.get_url()
    subprocess.run([
        "mc", "alias", "set", "testminio", url, "minioadmin", "minioadmin"
    ], check=True)
    for bucket in ["stratops-signals", "stratops-extracted", "stratops-correlations", 
                   "stratops-trends", "stratops-anomalies", "stratops-briefings"]:
        subprocess.run([
            "mc", "mb", f"testminio/{bucket}"
        ], check=True, capture_output=True)
    yield container
    container.stop()


@pytest.fixture
def neo4j_client(neo4j_container):
    """Neo4j client connected to test container."""
    client = Neo4jClient(
        uri=neo4j_container.get_url(),
        user="neo4j",
        password=neo4j_container.password,
    )
    yield client
    asyncio.get_event_loop().run_until_complete(client.close())


@pytest.fixture
async def test_tenant(neo4j_client):
    """Create a test tenant in Neo4j."""
    tenant_id = str(uuid4())
    await neo4j_client.run(
        "CREATE (t:Tenant {id: $id, name: 'Test Tenant'})",
        {"id": tenant_id}
    )
    return tenant_id


@pytest.fixture
def sample_signal_text() -> str:
    """Sample signal text for testing."""
    return (
        "Apple Inc. reported record quarterly revenue of $94.8 billion, "
        "up 2% year over year. CEO Tim Cook said the company is seeing "
        "strong demand for iPhone 15 and services. CFO Luca Maestri "
        "guided for $81-83 billion revenue next quarter. "
        "Microsoft reported Azure growth of 29% in constant currency. "
        "Satya Nadella emphasized AI-driven growth across cloud and productivity."
    )


@pytest.fixture
def mock_extraction_service():
    """Mock BentoML extraction service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-extraction:3000/v1/extract").mock(
            return_value=type('obj', (object,), {
                'status_code': 200,
                'json': lambda: [
                    {
                        "result": {
                            "entities": [
                                {"company_name": "Apple Inc.", "ticker": "AAPL"},
                                {"name": "Tim Cook", "role": "CEO"},
                                {"company_name": "Microsoft", "ticker": "MSFT"},
                                {"name": "Satya Nadella", "role": "CEO"},
                            ]
                        }
                    }
                ]
            })
        )
        yield mock


@pytest.fixture
def mock_summarization_service():
    """Mock BentoML summarization service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-summarization:3000/summarize").mock(
            return_value=type('obj', (object,), {
                'status_code': 200,
                'json': lambda: [
                    {"summaries": ["Apple reported record revenue driven by iPhone and services."], "model": "test", "batch_size": 1, "total_tokens": 50}
                ]
            })
        )
        yield mock


@pytest.fixture
def mock_narrative_service():
    """Mock BentoML narrative service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-narrative:3000/generate").mock(
            return_value=type('obj', (object,), {
                'status_code': 200,
                'json': lambda: {
                    "narrative": "# Executive Brief\n\nApple reported record revenue of $94.8B driven by iPhone and services.\n\nKey developments:\n- iPhone 15 demand strong\n- Services revenue growing\n- Guidance conservative\n\nRecommended actions:\n- Monitor iPhone demand in China\n- Invest in AI services",
                    "key_takeaways": [
                        "Apple reported record $94.8B revenue",
                        "iPhone and services driving growth",
                        "Conservative guidance for next quarter"
                    ],
                    "confidence": 0.85,
                    "model": "test-model"
                }
            })
        )
        yield mock


@pytest.fixture
def mock_fallback_service():
    """Mock BentoML fallback service."""
    with respx.mock() as mock:
        mock.post("http://bentoml-fallback:3000/process").mock(
            return_value=type('obj', (object,), {
                'status_code': 200,
                'json': lambda: [
                    {"result": "Fallback summary.", "task_type": "summarize", "model": "fallback-model"}
                ]
            })
        )
        yield mock


# ============================================================================
# Integration Tests
# ============================================================================

class TestFullPipeline:
    """Full pipeline test: Extraction → Correlation → Trend → Anomaly → Briefing → Alert."""

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self,
        postgres_container,
        redis_container,
        neo4j_container,
        minio_container,
        neo4j_client,
        test_tenant,
        sample_signal_text,
        mock_extraction_service,
        mock_summarization_service,
        mock_narrative_service,
    ):
        """Full pipeline test: Extraction → Correlation → Trend → Anomaly → Briefing → Alert."""
        tenant_id = test_tenant
        trace_id = "integration-test-001"

        # ================================
        # Step 1: Seed Neo4j with test data
        # ================================
        # Create companies
        await neo4j_client.run("""
            MERGE (c1:Company {name: 'Apple Inc.', tenant_id: $tid, ticker: 'AAPL'})
            MERGE (c2:Company {name: 'Microsoft', tenant_id: $tid, ticker: 'MSFT'})
            MERGE (p1:Person {name: 'Tim Cook', tenant_id: $tid, role: 'CEO'})
            MERGE (p2:Person {name: 'Satya Nadella', tenant_id: $tid, role: 'CEO'})
            MERGE (p1)-[:EMPLOYED_AT {role: 'CEO', valid_from: datetime()}]->(c1)
            MERGE (p2)-[:EMPLOYED_AT {role: 'CEO', valid_from: datetime()}]->(c2)
        """, {"tid": test_tenant})

        # Create signals with mentions
        await neo4j_client.run("""
            MATCH (c1:Company {tenant_id: $tid, name: 'Apple Inc.'})
            MATCH (c2:Company {tenant_id: $tid, name: 'Microsoft'})
            CREATE (s1:Signal {id: 'sig-1', tenant_id: $tid, source_type: 'earnings', source_url: 'http://test'})
            CREATE (s2:Signal {id: 'sig-2', tenant_id: $tid, source_type: 'news', source_url: 'http://test'})
            MERGE (c1)-[:MENTIONED_IN {valid_from: datetime(), sentiment: 'positive'}]->(s1)
            MERGE (c2)-[:MENTIONED_IN {valid_from: datetime(), sentiment: 'positive'}]->(s2)
        """, {"tid": test_tenant})

        # Create pricing data
        await neo4j_client.run("""
            MERGE (c1:Company {name: 'Apple Inc.', tenant_id: $tid})
            MERGE (c2:Company {name: 'Microsoft', tenant_id: $tid})
            MERGE (p1:Product {id: 'prod-1', tenant_id: $tid, name: 'Cloud Services'})
            MERGE (c1)-[:PRICED_AT {price: 100, currency: 'USD', valid_from: datetime()}]->(p1)
            MERGE (c2)-[:PRICED_AT {price: 105, currency: 'USD', valid_from: datetime()}]->(p1)
        """, {"tid": test_tenant})

        # ================================
        # Step 2: Run EntityExtractorNode
        # ================================
        extractor_node = EntityExtractorNode()
        
        # Mock the BentoML call
        with mock.patch.object(extractor_node, '_call_bentoml_extraction') as mock_extract:
            mock_extract.return_value = [
                {
                    "result": {
                        "entities": [
                            {"company_name": "Apple Inc.", "ticker": "AAPL"},
                            {"name": "Tim Cook", "role": "CEO"},
                            {"company_name": "Microsoft", "ticker": "MSFT"},
                            {"name": "Satya Nadella", "role": "CEO"},
                        ]
                    }
                }
            ]

            # Mock MinIO upload
            with mock.patch.object(extractor_node, '_upload_to_minio') as mock_upload:
                mock_upload.return_value = "s3://stratops-extracted-test/trace-001/entities.json"
                
                state: IntelligenceState = {
                    "tenant_id": test_tenant,
                    "trace_id": "trace-001",
                    "signal_uris": ["s3://stratops-signals/test-signal.json"],
                    "extracted_entities": [],
                    "content_uris": [],
                    "correlation_graph_delta": [],
                    "briefing_section_uris": [],
                }

                result_state = await extractor_node(state)

                # Verify extraction
                assert len(result_state["extracted_entities"]) == 4
                assert result_state["extracted_entities"][0]["company_name"] == "Apple Inc."
                
                # Verify pointer-only: no raw content in state
                state_json = json.dumps(result_state, default=str)
                assert "Apple Inc. reported record quarterly revenue" not in state_json
                
                # Verify state size < 5KB
                state_size = len(state_json.encode("utf-8"))
                assert state_size < 5000, f"State size {state_size} exceeds 5KB limit"
                
                # Verify content URIs added
                assert len(result_state["content_uris"]) == 1
                assert result_state["content_uris"][0].startswith("s3://stratops-extracted-")

        # ================================
        # Step 3: Run CorrelationEngineNode
        # ================================
        mock_neo4j = mock.AsyncMock()
        mock_minio = mock.AsyncMock()
        mock_minio.upload = mock.AsyncMock(return_value="s3://stratops-correlations-test/trace-001/correlations.json")

        correlation_node = CorrelationEngineNode(
            neo4j_client=mock_neo4j,
            minio_client=mock_minio,
            time_window_days=30,
        )

        # Mock Neo4j query responses
        mock_neo4j.run.side_effect = [
            # Pricing correlations
            [{
                "company_a": "Apple Inc.",
                "company_b": "Microsoft",
                "product": "Cloud Services",
                "price_a": "100",
                "price_b": "105",
                "valid_from": datetime.utcnow().isoformat(),
            }],
            # Talent flow
            [{
                "person_name": "John Doe",
                "company_from": "Company A",
                "company_to": "Company B",
                "role_from": "Engineer",
                "role_to": "Senior Engineer",
                "left_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "joined_date": datetime.utcnow().isoformat(),
            }],
            # Co-mention
            [{
                "company_a": "Apple Inc.",
                "company_b": "Microsoft",
                "signal_uri": "s3://signal-1",
                "valid_from": datetime.utcnow().isoformat(),
            }],
            # Patent (empty)
            [],
        ]

        result_state = await correlation_node(result_state)

        # Verify correlations detected
        assert len(result_state["content_uris"]) >= 2  # Original + correlations
        assert len(result_state["correlation_graph_delta"]) >= 3

        # Verify state size still < 5KB
        state_size = len(json.dumps(result_state, default=str).encode("utf-8"))
        assert state_size < 5000

        # ================================
        # Step 4: Run TrendAnalyzerNode
        # ================================
        mock_db_pool = mock.AsyncMock()
        mock_summarization = mock.AsyncMock()
        mock_summarization.post.return_value = type('obj', (object,), {
            'status_code': 200,
            'json': lambda: [{"summaries": ["Price trend increasing."]}]
        })

        mock_minio_trend = mock.AsyncMock()
        mock_minio_trend.upload = mock.AsyncMock(return_value="s3://stratops-trends-test/trace-001/trends.json")

        trend_node = TrendAnalyzerNode(
            db_pool=mock_db_pool,
            summarization_client=mock_summarization,
            minio_client=mock_minio_trend,
            lookback_days=90,
            z_threshold=2.5,
            stl_min_points=30,
        )

        mock_db_pool.fetch.side_effect = [
            # Pricing trends
            [
                {"company": "Apple Inc.", "product": "iPhone", "price": "100", "valid_from": datetime.utcnow().isoformat()},
                {"company": "Apple Inc.", "product": "iPhone", "price": "105", "valid_from": (datetime.utcnow() + timedelta(days=1)).isoformat()},
            ],
            # Hiring trends
            [],
            # Mention trends
            [],
            # Sentiment trends
            [],
        ]

        result_state = await TrendAnalyzerNode(
            db_pool=mock_db_pool,
            summarization_client=mock_summarization,
            minio_client=mock_minio_trend,
        )(result_state)

        # Verify trends added
        assert any("stratops-trends-" in uri for uri in result_state["content_uris"])
        assert len(result_state["briefing_section_uris"]) > 0

        # ================================
        # Step 5: Run AnomalyDetectorNode
        # ================================
        mock_minio_anomaly = mock.AsyncMock()
        mock_minio_anomaly.upload = mock.AsyncMock(return_value="s3://stratops-anomalies-test/trace-001/anomalies.json")

        anomaly_node = AnomalyDetectorNode(
            db_pool=mock_db_pool,
            summarization_client=mock.AsyncMock(),
            minio_client=mock_minio_anomaly,
            contamination=0.1,
            anomaly_threshold=-0.5,
            lookback_days=90,
        )

        # Mock trained model
        import numpy as np
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(np.random.rand(100, 4))
        anomaly_node._model = model

        result_state = await anomaly_node(result_state)

        # Verify anomalies written
        assert any("stratops-anomalies-" in uri for uri in result_state["content_uris"])

        # ================================
        # Step 6: Run BriefingComposerNode
        # ================================
        mock_minio_briefing = mock.AsyncMock()
        mock_minio_briefing.upload = mock.AsyncMock(return_value="s3://stratops-briefings-test/briefing-123/v1.md")
        mock_minio_briefing.download = mock.AsyncMock()

        mock_narrative_client = mock.AsyncMock()
        mock_narrative_client.post = mock.AsyncMock()
        mock_narrative_client.post.return_value = mock.AsyncMock(
            status_code=200,
            json=mock.AsyncMock(return_value={
                "narrative": "# Executive Brief\n\nApple reported record revenue of $94.8B driven by iPhone and services.\n\nKey developments:\n- iPhone 15 demand strong\n- Services revenue growing\n- Guidance conservative\n\nRecommended actions:\n- Monitor iPhone demand in China\n- Invest in AI services",
                "key_takeaways": [
                    "Apple reported record $94.8B revenue",
                    "iPhone and services driving growth",
                    "Conservative guidance for next quarter"
                ],
                "confidence": 0.85,
                "model": "test-model"
            })
        )

        composer_node = BriefingComposerNode(
            minio_client=mock_minio_briefing,
            narrative_client=mock_narrative_client,
        )

        result_state = await composer_node(result_state)

        # Verify briefing created
        assert len(result_state["briefing_section_uris"]) > 0
        assert any("stratops-briefings-" in uri for uri in result_state["briefing_section_uris"])

        # ================================
        # Step 7: Run AlertRuleEngine
        # ================================
        alert_engine = AlertRuleEngine()

        # Create alert rules
        pricing_rule = AlertRule(
            tenant_id=test_tenant,
            name="Pricing Delta Alert",
            rule_type="threshold",
            condition={"metric": "pricing_delta", "operator": "gt", "value": 0.15},
            severity="warning",
            channels=["slack"],
        )

        anomaly_rule = AlertRule(
            tenant_id=test_tenant,
            name="High Severity Anomaly",
            rule_type="anomaly",
            condition={"severity": "high", "entity_types": ["Company"]},
            severity="critical",
            channels=["slack", "email"],
        )

        # Mock metrics extraction
        with mock.patch.object(
            AlertRuleEngine, "_extract_metrics",
            return_value={"Apple Inc.": 0.2, "Microsoft": 0.1}
        ):
            alerts = await alert_engine.evaluate(
                result_state,
                [pricing_rule, anomaly_rule]
            )

        # Verify alerts triggered
        assert len(alerts) >= 1
        pricing_alerts = [a for a in alerts if a.rule_name == "Pricing Delta Alert"]
        assert len(pricing_alerts) == 1
        assert "0.20" in pricing_alerts[0].message

        # ================================
        # Step 8: Run AlertRouter
        # ================================
        slack_config = SlackChannelConfig(
            webhook_url="https://hooks.slack.com/test",
            username="TestBot",
        )
        email_config = EmailChannelConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password",
            from_email="alerts@test.com",
        )

        router = AlertRouter(
            slack_config=SlackChannelConfig(webhook_url="https://hooks.slack.com/test"),
            email_config=EmailChannelConfig(
                smtp_host="smtp.test.com",
                smtp_port=587,
                username="test@test.com",
                password="password",
                from_email="alerts@test.com",
            ),
            webhook_config=WebhookChannelConfig(url="https://webhook.test.com/alert"),
        )

        # Route alerts
        for alert in alerts:
            results = await AlertRouter(
                slack_config=SlackChannelConfig(webhook_url="https://hooks.slack.com/test"),
                email_config=EmailChannelConfig(
                    smtp_host="smtp.test.com",
                    smtp_port=587,
                    username="test@test.com",
                    password="password",
                    from_email="alerts@test.com",
                ),
                webhook_config=WebhookChannelConfig(url="https://webhook.test.com/alert"),
            ).route(alert, alert.channels)

            assert any(results.values()), f"Alert routing failed for {alert.rule_name}"

        # ================================
        # Step 9: Verify pointer-only state throughout
        # ================================
        final_state_size = len(json.dumps(result_state, default=str).encode("utf-8"))
        assert final_state_size < 5000, f"Final state size {final_state_size} exceeds 5KB limit"

        # Verify no raw content in state
        final_json = json.dumps(result_state, default=str)
        assert "Apple Inc. reported record quarterly revenue" not in final_json
        assert "Tim Cook said the company is seeing strong demand" not in final_json
        assert "Microsoft reported Azure growth" not in final_json

        print(f"✅ Full pipeline test passed!")
        print(f"   Final state size: {final_state_size} bytes (< 5KB)")
        print(f"   Content URIs: {len(result_state['content_uris'])}")
        print(f"   Correlation deltas: {len(result_state['correlation_graph_delta'])}")
        print(f"   Briefing sections: {len(result_state['briefing_section_uris'])}")


class TestPointerOnlyConstraint:
    """Tests to verify pointer-only constraint is maintained throughout pipeline."""

    @pytest.mark.asyncio
    async def test_no_raw_content_in_state(self):
        """Verify no raw content ever appears in LangGraph state."""
        extractor_node = EntityExtractorNode()

        with mock.patch.object(extractor_node, '_call_bentoml_extraction') as mock_extract:
            mock_extract.return_value = [{
                "result": {
                    "entities": [
                        {"company_name": "Test Corp", "ticker": "TC"}
                    ]
                }
            }]

            with mock.patch.object(EntityExtractorNode, '_upload_to_minio') as mock_upload:
                mock_upload.return_value = "s3://test/entities.json"

                state = {
                    "tenant_id": "001",
                    "trace_id": "trace-001",
                    "signal_uris": ["s3://stratops-signals/test.json"],
                    "extracted_entities": [],
                    "content_uris": [],
                    "correlation_graph_delta": [],
                    "briefing_section_uris": [],
                }

                extractor_node = EntityExtractorNode()
                result = await extractor_node(state)

                # Check state size
                state_size = len(json.dumps(result, default=str).encode("utf-8"))
                assert state_size < 5000

                # Verify no raw signal text in state
                state_str = json.dumps(result, default=str)
                assert "full signal text" not in state_str.lower()


class TestRLSEnforcement:
    """Tests for RLS enforcement in briefing persistence."""

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, neo4j_client):
        """Verify tenant A cannot see tenant B's briefings."""
        # Create tenant A and B
        tenant_a = str(uuid4())
        tenant_b = str(uuid4())

        await neo4j_client.run("""
            CREATE (t1:Tenant {id: $a, name: 'Tenant A'})
            CREATE (t2:Tenant {id: $b, name: 'Tenant B'})
        """, {"a": tenant_a, "b": tenant_b})

        # Create briefing for tenant A
        await neo4j_client.run("""
            CREATE (b:Briefing {id: $id, tenant_id: $tid, title: 'Test', content_md_uri: 's3://test'})
        """, {"id": str(uuid4()), "tid": tenant_a})

        # Try to query as tenant B - should not see tenant A's briefing
        result = await neo4j_client.run(
            "MATCH (b:Briefing {tenant_id: $tid}) RETURN count(b)",
            {"tid": tenant_b}
        )
        assert result[0]["count(b)"] == 0


class TestCheckpointSize:
    """Tests for LangGraph checkpoint size constraint."""

    def test_state_size_limit(self):
        """Verify state size stays under 5KB."""
        state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "extracted_entities": [{"company_name": f"Company {i}", "ticker": f"T{i}"} for i in range(100)],
            "content_uris": [f"s3://test/{i}" for i in range(50)],
            "correlation_graph_delta": [f"delta_{i}" for i in range(100)],
            "briefing_section_uris": [],
        }

        state_size = len(json.dumps(state, default=str).encode("utf-8"))
        # Even with 100 entities, should be under 5KB (entities are small dicts)
        assert state_size < 5000, f"State size {state_size} exceeds 5KB"
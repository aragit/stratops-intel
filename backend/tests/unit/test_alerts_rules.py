"""Unit tests for the Alert Rule Engine."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from backend.alerts.rules import (
    Alert,
    AlertRule,
    AlertRuleEngine,
)


class TestAlertRule:
    """Tests for AlertRule model."""

    def test_rule_creation(self) -> None:
        """Test basic rule creation."""
        rule = AlertRule(
            tenant_id="001",
            name="Test Rule",
            rule_type="threshold",
            condition={"metric": "pricing_delta", "operator": "gt", "value": 0.15},
            severity="warning",
            channels=["slack", "email"],
        )
        assert rule.tenant_id == "001"
        assert rule.name == "Test Rule"
        assert rule.rule_type == "threshold"
        assert rule.severity == "warning"
        assert rule.channels == ["slack", "email"]

    def test_rule_defaults(self) -> None:
        """Test default values."""
        rule = AlertRule(
            tenant_id="001",
            name="Test",
            rule_type="threshold",
            condition={"metric": "test", "operator": "gt", "value": 1},
        )
        assert rule.severity == "warning"
        assert rule.channels == []
        assert rule.is_active is True
        assert rule.id  # UUID generated

    def test_severity_validation(self) -> None:
        """Test severity validation."""
        for severity in ["info", "warning", "critical"]:
            rule = AlertRule(
                tenant_id="001",
                name="Test",
                rule_type="threshold",
                condition={"metric": "test", "operator": "gt", "value": 1},
                severity=severity,
            )
            assert rule.severity == severity

    def test_rule_type_validation(self) -> None:
        """Test rule_type validation."""
        for rtype in ["threshold", "anomaly", "correlation", "trend"]:
            rule = AlertRule(
                tenant_id="001",
                name="Test",
                rule_type=rtype,
                condition={"metric": "test", "operator": "gt", "value": 1},
            )
            assert rule.rule_type == rtype


class TestAlert:
    """Tests for Alert model."""

    def test_alert_creation(self) -> None:
        """Test alert creation."""
        alert = Alert(
            tenant_id="001",
            rule_id="rule-123",
            rule_name="Test Rule",
            severity="critical",
            message="Pricing delta exceeded threshold",
            evidence={"metric": "pricing_delta", "value": 0.2},
        )
        assert alert.tenant_id == "001"
        assert alert.rule_id == "rule-123"
        assert alert.severity == "critical"
        assert alert.evidence["metric"] == "pricing_delta"


class TestAlertRuleEngine:
    """Tests for AlertRuleEngine."""

    @pytest.fixture
    def engine(self):
        """Provide an AlertRuleEngine instance."""
        return AlertRuleEngine()

    @pytest.fixture
    def sample_state(self) -> dict:
        """Sample IntelligenceState for testing."""
        return {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "content_uris": [],
            "extracted_entities": [],
            "correlation_graph_delta": [],
            "briefing_section_uris": [],
        }

    def test_operator_mapping(self, engine: Any) -> None:
        """Test operator function mapping."""
        assert engine._operators["gt"](5, 3) is True
        assert engine._operators["gt"](3, 5) is False
        assert engine._operators["gte"](5, 5) is True
        assert engine._operators["lt"](3, 5) is True
        assert engine._operators["lte"](5, 5) is True
        assert engine._operators["eq"](5, 5) is True
        assert engine._operators["neq"](5, 3) is True

    def test_create_alert(self, engine):
        """Test alert creation."""
        rule = mock.MagicMock()
        rule.tenant_id = "001"
        rule.id = "rule-123"
        rule.name = "Test Rule"
        rule.severity = "warning"

        alert = engine._create_alert(
            rule=rule,
            message="Test message",
            evidence={"metric": "test", "value": 1.0},
        )

        assert alert.tenant_id == "001"
        assert alert.rule_id == "rule-123"
        assert alert.rule_name == "Test Rule"
        assert alert.severity == rule.severity
        assert alert.message == "Test message"
        assert alert.evidence["metric"] == "test"
        assert alert.evidence["value"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_inactive_rule_skipped(self, engine: AlertRuleEngine):
        """Inactive rules should be skipped."""
        rule = AlertRule(
            tenant_id="001",
            name="Inactive Rule",
            rule_type="threshold",
            condition={"metric": "test", "operator": "gt", "value": 1},
            is_active=False,
        )

        state = {"tenant_id": "001", "trace_id": "trace-001"}
        alerts = await engine.evaluate(state, [rule])

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_threshold_rule(self, engine):
        """Test threshold rule evaluation."""
        rule = AlertRule(
            tenant_id="001",
            name="Pricing Alert",
            rule_type="threshold",
            condition={
                "metric": "pricing_delta",
                "operator": "gt",
                "value": 0.15,
            },
            severity="warning",
        )

        _state = {
            "tenant_id": "001",
            "trace_id": "trace-001",
        }

        # Mock _extract_metrics to return test data
        with mock.patch.object(
            AlertRuleEngine, "_extract_metrics", return_value={"Company A": 0.2, "Company B": 0.1}
        ):
            alerts = await engine.evaluate({"tenant_id": "001", "trace_id": "trace-001"}, [rule])

        assert len(alerts) == 1
        assert alerts[0].message == "pricing_delta for Company A (0.20) gt threshold (0.15)"

    @pytest.mark.asyncio
    async def test_threshold_operator_lt(self, engine):
        """Test threshold with lt operator."""
        rule = AlertRule(
            tenant_id="001",
            name="Low Volume Alert",
            rule_type="threshold",
            condition={
                "metric": "mention_frequency",
                "operator": "lt",
                "value": 5,
            },
            severity="warning",
        )

        with mock.patch.object(
            AlertRuleEngine, "_extract_metrics", return_value={"Company A": 3, "Company B": 10}
        ):
            alerts = await engine.evaluate({"tenant_id": "001", "trace_id": "trace-001"}, [rule])

        assert len(alerts) == 1
        assert "lt" in alerts[0].message

    @pytest.mark.asyncio
    async def test_threshold_entity_filter(self, engine):
        """Test threshold with entity filter."""
        rule = AlertRule(
            tenant_id="001",
            name="Apple Pricing Alert",
            rule_type="threshold",
            condition={
                "metric": "pricing_delta",
                "operator": "gt",
                "value": 0.1,
                "entity_filter": {"type": "Company", "name": "Apple Inc."},
            },
            severity="warning",
        )

        with mock.patch.object(
            AlertRuleEngine, "_extract_metrics", return_value={"Apple Inc.": 0.2, "Microsoft": 0.15}
        ):
            alerts = await engine.evaluate({"tenant_id": "001", "trace_id": "trace-001"}, [rule])

        assert len(alerts) == 1
        assert alerts[0].evidence["entity"] == "Apple Inc."

    @pytest.mark.asyncio
    async def test_multiple_rules_some_trigger(self):
        """Multiple rules - some trigger, some don't."""
        _engine = AlertRuleEngine()

        rule1 = AlertRule(
            tenant_id="001",
            name="High Alert",
            rule_type="threshold",
            condition={"metric": "test", "operator": "gt", "value": 10},
            severity="critical",
        )
        rule2 = AlertRule(
            tenant_id="001",
            name="Low Alert",
            rule_type="threshold",
            condition={"metric": "test", "operator": "gt", "value": 100},
            severity="warning",
        )

        with mock.patch.object(
            AlertRuleEngine, "_extract_metrics", return_value={"Entity A": 50, "Entity B": 5}
        ):
            alerts = await AlertRuleEngine().evaluate(
                {"tenant_id": "001", "trace_id": "trace-001"},
                [rule1, rule2],
            )

        assert len(alerts) == 1
        assert alerts[0].rule_name == "High Alert"

    @pytest.mark.asyncio
    async def test_unknown_rule_type(self, engine: AlertRuleEngine):
        """Unknown rule type should be skipped with warning."""
        rule = AlertRule(
            tenant_id="001",
            name="Unknown Type",
            rule_type="unknown_type",
            condition={},
        )

        with mock.patch("structlog.get_logger") as mock_logger:
            alerts = await engine.evaluate({"tenant_id": "001"}, [rule])

        assert len(alerts) == 0
        mock_logger.return_value.warning.assert_called_once()

    def test_create_alert_structure(self):
        """Test alert creation with proper structure."""
        from backend.alerts.rules import Alert, AlertRuleEngine

        engine = AlertRuleEngine()
        rule = mock.MagicMock()
        rule.tenant_id = "001"
        rule.id = "rule-123"
        rule.name = "Test Rule"
        rule.severity = "warning"

        alert = engine._create_alert(rule, "Test message", {"key": "value"})

        assert isinstance(alert, Alert)
        assert alert.tenant_id == "001"
        assert alert.rule_id == "rule-123"
        assert alert.rule_name == "Test Rule"
        assert alert.severity == "warning"
        assert alert.message == "Test message"
        assert alert.evidence == {"key": "value"}
        assert alert.id
        assert alert.created_at


class TestExtractMetrics:
    """Tests for AlertRuleEngine._extract_metrics."""

    @pytest.fixture
    def engine(self) -> AlertRuleEngine:
        return AlertRuleEngine()

    @pytest.fixture
    def rich_state(self) -> dict:
        return {
            "tenant_id": "001",
            "trace_id": "trace-001",
            "signal_uris": [],
            "content_uris": [],
            "briefing_section_uris": [],
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "type": "Company",
                    "anomaly_score": 0.92,
                    "anomaly_baseline": 0.4,
                    "mention_count": 12,
                    "pricing_delta": 0.22,
                    "metrics_history": {"pricing": [10.0, 10.5, 11.0, 14.0]},
                },
                {
                    "company_name": "Google",
                    "type": "Company",
                    "anomaly_score": 0.3,
                },
            ],
            "correlation_graph_delta": [
                "MERGE (a:Entity {id: 'Apple'})-[r:CORRELATED_WITH "
                "{type: 'pricing', strength: 0.85, valid_from: '2026-08-01T00:00:00'}]"
                "->(b:Entity {id: 'Google'})",
            ],
        }

    def test_full_bundle(self, engine: AlertRuleEngine, rich_state: dict) -> None:
        metrics = engine._extract_metrics(rich_state)

        assert set(metrics.keys()) == {
            "graph_density",
            "anomaly_scores",
            "anomaly_baselines",
            "entity_trends",
            "pricing_delta",
            "mention_frequency",
            "hiring_velocity",
            "sentiment",
        }
        assert metrics["anomaly_scores"] == {"Apple": 0.92, "Google": 0.3}
        assert metrics["anomaly_baselines"] == {"Apple": 0.4}
        assert metrics["entity_trends"]["Apple"]["pricing"] == [10.0, 10.5, 11.0, 14.0]
        assert metrics["mention_frequency"] == {"Apple": 12}
        assert 0.0 <= metrics["graph_density"] <= 1.0

    def test_graph_density_computation(self, engine: AlertRuleEngine, rich_state: dict) -> None:
        # 2 nodes, 1 edge -> density = 1 / 1
        assert engine._extract_metrics(rich_state)["graph_density"] == 1.0

    def test_graph_density_no_edges(self, engine: AlertRuleEngine) -> None:
        state = {"correlation_graph_delta": []}
        assert engine._extract_metrics(state)["graph_density"] == 0.0

    def test_flat_metric_query(self, engine: AlertRuleEngine, rich_state: dict) -> None:
        flat = engine._extract_metrics(rich_state, "pricing_delta")
        assert flat == {"Apple": 0.22}

    def test_flat_metric_unknown_returns_empty(
        self, engine: AlertRuleEngine, rich_state: dict
    ) -> None:
        assert engine._extract_metrics(rich_state, "nonexistent") == {}


class TestAnomalyEvaluator:
    """Tests for the anomaly rule evaluator."""

    @pytest.fixture
    def engine(self) -> AlertRuleEngine:
        return AlertRuleEngine()

    @pytest.mark.asyncio
    async def test_triggers_above_floor_and_baseline(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Anomaly High",
            rule_type="anomaly",
            condition={"severity": "high"},
        )
        state = {
            "extracted_entities": [
                {"company_name": "Apple", "anomaly_score": 0.92, "anomaly_baseline": 0.4},
            ],
        }

        alerts = await engine._evaluate_anomaly(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["entity"] == "Apple"
        assert alerts[0].evidence["anomaly_score"] == 0.92
        assert alerts[0].evidence["baseline"] == 0.4

    @pytest.mark.asyncio
    async def test_no_trigger_below_severity_floor(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Anomaly Critical",
            rule_type="anomaly",
            condition={"severity": "critical"},
        )
        state = {
            "extracted_entities": [
                {"company_name": "Apple", "anomaly_score": 0.8},
            ],
        }

        alerts = await engine._evaluate_anomaly(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_no_trigger_at_or_below_baseline(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Anomaly High",
            rule_type="anomaly",
            condition={"severity": "high"},
        )
        state = {
            "extracted_entities": [
                {"company_name": "Apple", "anomaly_score": 0.9, "anomaly_baseline": 0.9},
            ],
        }

        alerts = await engine._evaluate_anomaly(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_entity_type_filter(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Company Anomalies Only",
            rule_type="anomaly",
            condition={"severity": "low", "entity_types": ["Company"]},
        )
        state = {
            "extracted_entities": [
                {"company_name": "Apple", "type": "Company", "anomaly_score": 0.5},
                {"company_name": "Widget", "type": "Product", "anomaly_score": 0.95},
            ],
        }

        alerts = await engine._evaluate_anomaly(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["entity"] == "Apple"


class TestCorrelationEvaluator:
    """Tests for the correlation rule evaluator."""

    @pytest.fixture
    def engine(self) -> AlertRuleEngine:
        return AlertRuleEngine()

    @pytest.fixture
    def merge_delta(self) -> str:
        return (
            "MERGE (a:Entity {id: 'Apple'})-[r:CORRELATED_WITH "
            "{type: 'pricing', strength: 0.85, valid_from: '2026-08-01T00:00:00'}]"
            "->(b:Entity {id: 'Google'})"
        )

    @pytest.mark.asyncio
    async def test_triggers_on_strong_correlation(
        self, engine: AlertRuleEngine, merge_delta: str
    ) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Pricing Correlation",
            rule_type="correlation",
            condition={"correlation_type": "pricing", "min_strength": 0.7},
        )
        state = {"correlation_graph_delta": [merge_delta]}

        alerts = await engine._evaluate_correlation(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["strength"] == 0.85
        assert alerts[0].evidence["entity_a"] == "Apple"
        assert alerts[0].evidence["entity_b"] == "Google"

    @pytest.mark.asyncio
    async def test_no_trigger_below_min_strength(
        self, engine: AlertRuleEngine, merge_delta: str
    ) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Strong Pricing Correlation",
            rule_type="correlation",
            condition={"correlation_type": "pricing", "min_strength": 0.9},
        )
        state = {"correlation_graph_delta": [merge_delta]}

        alerts = await engine._evaluate_correlation(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_no_trigger_on_type_mismatch(
        self, engine: AlertRuleEngine, merge_delta: str
    ) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Talent Correlation",
            rule_type="correlation",
            condition={"correlation_type": "talent", "min_strength": 0.5},
        )
        state = {"correlation_graph_delta": [merge_delta]}

        alerts = await engine._evaluate_correlation(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_structured_correlations_honored(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Co-mention",
            rule_type="correlation",
            condition={"correlation_type": "co_mention", "min_strength": 0.7},
        )
        state = {
            "correlations": [
                {
                    "correlation_type": "co_mention",
                    "strength": 0.8,
                    "entity_a": {"type": "Company", "name": "Apple"},
                    "entity_b": {"type": "Company", "name": "Google"},
                    "valid_from": "2026-08-01T00:00:00",
                },
            ],
        }

        alerts = await engine._evaluate_correlation(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["entity_a"] == "Apple"

    @pytest.mark.asyncio
    async def test_missing_correlation_type_returns_empty(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Bad Rule",
            rule_type="correlation",
            condition={},
        )

        alerts = await engine._evaluate_correlation({}, rule)
        assert alerts == []


class TestTrendEvaluator:
    """Tests for the trend rule evaluator."""

    @pytest.fixture
    def engine(self) -> AlertRuleEngine:
        return AlertRuleEngine()

    @pytest.mark.asyncio
    async def test_upward_trend_triggers(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Pricing Up",
            rule_type="trend",
            condition={"trend_type": "pricing", "direction": "up", "z_score_min": 2.0},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"pricing": [10.0, 10.5, 11.0, 14.0]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["direction"] == "up"
        assert alerts[0].evidence["z_score"] >= 2.0
        assert alerts[0].evidence["checkpoints"] == 4

    @pytest.mark.asyncio
    async def test_downward_trend_triggers(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Mentions Down",
            rule_type="trend",
            condition={"trend_type": "mention_frequency", "direction": "down", "z_score_min": 1.5},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"mention_frequency": [50.0, 48.0, 20.0]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["z_score"] <= -1.5

    @pytest.mark.asyncio
    async def test_direction_mismatch_does_not_trigger(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Pricing Down",
            rule_type="trend",
            condition={"trend_type": "pricing", "direction": "down", "z_score_min": 2.0},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"pricing": [10.0, 10.5, 11.0, 14.0]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_insufficient_checkpoints_skipped(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Pricing Up",
            rule_type="trend",
            condition={"trend_type": "pricing", "direction": "up", "z_score_min": 1.0},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"pricing": [10.0, 99.0]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_zero_variance_series_skipped(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Flat Pricing",
            rule_type="trend",
            condition={"trend_type": "pricing", "direction": "up", "z_score_min": 1.0},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"pricing": [10.0, 10.0, 10.0]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_structured_trends_from_state(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Hiring Up",
            rule_type="trend",
            condition={"trend_type": "hiring", "direction": "up", "z_score_min": 2.0},
        )
        state = {
            "trends": [
                {
                    "trend_type": "hiring",
                    "entity_name": "Apple",
                    "direction": "up",
                    "z_score": 2.7,
                    "confidence": 0.9,
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)

        assert len(alerts) == 1
        assert alerts[0].evidence["source"] == "trend_analyzer"

    @pytest.mark.asyncio
    async def test_anomalous_direction_matches_negative_z(self, engine: AlertRuleEngine) -> None:
        rule = AlertRule(
            tenant_id="001",
            name="Sentiment Anomalous",
            rule_type="trend",
            condition={"trend_type": "sentiment", "direction": "anomalous", "z_score_min": 2.0},
        )
        state = {
            "extracted_entities": [
                {
                    "company_name": "Apple",
                    "metrics_history": {"sentiment": [0.5, 0.52, -0.4]},
                },
            ],
        }

        alerts = await engine._evaluate_trend(state, rule)

        assert len(alerts) == 1

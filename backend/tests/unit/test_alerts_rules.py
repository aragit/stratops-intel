"""Unit tests for the Alert Rule Engine."""

from __future__ import annotations

from datetime import datetime
from unittest import mock

import pytest

from backend.alerts.rules import (
    AlertRuleEngine,
    AlertRule,
    Alert,
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
            "extracted_entities": [
                {"company_name": "Company A", "ticker": "A"},
                {"company_name": "Company B", "ticker": "B"},
            ],
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

        state = {
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
        engine = AlertRuleEngine()

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
        from backend.alerts.rules import AlertRuleEngine, Alert

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
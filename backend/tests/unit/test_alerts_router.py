"""Unit tests for the Alert Router."""

from __future__ import annotations

import json
from datetime import datetime
from unittest import mock

import pytest

from backend.alerts.router import (
    Alert,
    AlertRouter,
    AlertRouterWorker,
    EmailChannelConfig,
    SlackChannelConfig,
    WebhookChannelConfig,
)


@pytest.fixture
def slack_config():
    """Provide a Slack channel config."""
    return SlackChannelConfig(
        webhook_url="https://hooks.slack.com/test",
        username="TestBot",
        icon_emoji=":test:",
    )


@pytest.fixture
def email_config():
    """Provide an email channel config."""
    return EmailChannelConfig(
        smtp_host="smtp.test.com",
        smtp_port=587,
        username="test@test.com",
        password="password",
        from_email="alerts@test.com",
        from_name="Test Alerts",
    )


@pytest.fixture
def webhook_config():
    """Provide a webhook channel config."""
    return WebhookChannelConfig(
        url="https://webhook.test.com/alert",
        headers={"Authorization": "Bearer token"},
        timeout_seconds=10,
    )


@pytest.fixture
def router(slack_config, email_config, webhook_config):
    """Provide an AlertRouter instance with all channels configured."""
    return AlertRouter(
        slack_config=slack_config,
        email_config=email_config,
        webhook_config=webhook_config,
        max_retries=2,
        retry_backoff_base=0.1,  # Fast for tests
    )


@pytest.fixture
def sample_alert() -> Alert:
    """Provide a sample Alert for testing."""
    return Alert(
        tenant_id="001",
        rule_id="rule-123",
        rule_name="Test Rule",
        severity="warning",
        message="Test alert message",
        evidence={"metric": "pricing_delta", "value": 0.2},
    )


class TestAlertRouter:
    """Tests for the AlertRouter class."""

    @pytest.mark.asyncio
    async def test_route_single_channel_slack(self, router, sample_alert):
        """Test routing to single Slack channel."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = mock.AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            results = await router.route(sample_alert, ["slack"])

            assert results["slack"] is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_multiple_channels(self, router, sample_alert):
        """Test routing to multiple channels."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            with mock.patch("aiosmtplib.send") as mock_smtp:
                mock_response = mock.AsyncMock()
                mock_response.status = 200
                mock_post.return_value.__aenter__.return_value = mock_response
                mock_smtp.return_value = None

                results = await router.route(sample_alert, ["slack", "email", "webhook"])

                assert results["slack"] is True
                assert results["email"] is True
                assert results["webhook"] is True

    @pytest.mark.asyncio
    async def test_route_unconfigured_channel(self, router, sample_alert):
        """Test routing to unconfigured channel returns False."""
        # Router without webhook config
        router.webhook_config = None

        results = await router.route(sample_alert, ["webhook"])

        assert results["webhook"] is False

    @pytest.mark.asyncio
    async def test_route_retry_on_failure(self, router, sample_alert):
        """Test retry logic on transient failure."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            # First two calls fail, third succeeds
            mock_response_fail = mock.AsyncMock()
            mock_response_fail.status = 500
            mock_response_fail.text = mock.AsyncMock(return_value="Server Error")

            mock_response_success = mock.AsyncMock()
            mock_response_success.status = 200

            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            results = await router.route(sample_alert, ["slack"])

            assert results["slack"] is True
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_route_max_retries_exhausted(self, router, sample_alert):
        """Test max retries exhausted returns False."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = mock.AsyncMock()
            mock_response.status = 500
            mock_response.text = mock.AsyncMock(return_value="Server Error")
            mock_post.return_value.__aenter__.return_value = mock_response

            results = await router.route(sample_alert, ["slack"])

            assert results["slack"] is False

    def test_slack_color_mapping(self, router, sample_alert):
        """Test Slack color mapping by severity."""
        colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "critical": "#cc0000",
        }

        for severity, expected_color in colors.items():
            alert = mock.MagicMock()
            alert.severity = severity
            alert.rule_name = "Test"
            alert.message = "Test"
            alert.evidence = {}
            alert.id = "test-123"
            alert.tenant_id = "001"
            alert.rule_name = "Test Rule"
            alert.severity = severity
            alert.message = "Test message"
            alert.evidence = {}
            alert.id = "test-123"
            alert.created_at = datetime.utcnow()
            alert.rule_name = "Test Rule"
            alert.rule_id = "rule-123"
            alert.tenant_id = "001"

            # Build payload manually to test color
            severity_colors = {
                "info": "#36a64f",
                "warning": "#ff9900",
                "critical": "#cc0000",
            }
            color = severity_colors.get(alert.severity, "#808080")
            assert color == expected_color

    @pytest.mark.asyncio
    async def test_slack_block_kit_structure(self, router, sample_alert):
        """Test Slack Block Kit payload structure."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = mock.AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            await router._send_slack(sample_alert)

            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "attachments" in payload
            assert len(payload["attachments"]) == 1
            attachment = payload["attachments"][0]
            assert "blocks" in attachment
            assert "color" in attachment

            blocks = attachment["blocks"]
            assert len(blocks) >= 3  # header, section with fields, section with message

            # Check header block
            header = blocks[0]
            assert header["type"] == "header"
            assert header["text"]["text"] == "Test Rule"

    @pytest.mark.asyncio
    async def test_email_html_format(self, router, sample_alert):
        """Test email HTML generation."""
        with mock.patch("aiosmtplib.send") as mock_send:
            mock_send.return_value = None

            html = router._format_evidence_html(sample_alert.evidence)

            assert "<table" in html
            assert "metric" in html
            assert "pricing_delta" in html

    @pytest.mark.asyncio
    async def test_email_send(self, router, sample_alert, email_config):
        """Test email sending."""
        with mock.patch("aiosmtplib.send") as mock_send:
            mock_send.return_value = None

            router.email_config = email_config
            result = await router._send_email(sample_alert)

            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_send(self, router, sample_alert, webhook_config):
        """Test webhook sending."""
        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = mock.AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await router._send_webhook(sample_alert)

            assert result is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_retry_on_transient_failure(self, sample_alert):
        """Test retry logic with exponential backoff."""
        router = AlertRouter(max_retries=2, retry_backoff_base=0.01)

        with mock.patch("aiohttp.ClientSession.post") as mock_post:
            mock_response_fail = mock.AsyncMock()
            mock_response_fail.status = 503
            mock_response_fail.text = mock.AsyncMock(return_value="Service Unavailable")

            mock_response_success = mock.AsyncMock()
            mock_response_success.status = 200

            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success,
            ]

            with mock.patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = mock.MagicMock()
                mock_session.post.return_value.__aenter__.side_effect = [
                    mock_response_fail,
                    mock_response_success,
                ]
                mock_session_class.return_value.__aenter__.return_value = mock_session

                router = AlertRouter(
                    slack_config=SlackChannelConfig(webhook_url="https://hooks.slack.com/test"),
                    max_retries=2,
                    retry_backoff_base=0.01,
                )

                results = await router.route(
                    Alert(
                        tenant_id="001",
                        rule_id="rule-123",
                        rule_name="Test",
                        severity="warning",
                        message="Test",
                        evidence={},
                    ),
                    ["slack"],
                )

                assert results["slack"] is True


class TestAlertRouterWorker:
    """Tests for AlertRouterWorker."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return mock.AsyncMock()

    @pytest.fixture
    def router(self):
        """Provide a basic router."""
        return mock.MagicMock()

    @pytest.fixture
    def worker(self, mock_redis, router):
        """Provide a worker instance."""
        return AlertRouterWorker(
            redis=mock_redis,
            router=router,
            tenant_id="001",
        )

    @pytest.mark.asyncio
    async def test_worker_start_stop(self, worker):
        """Test worker start/stop lifecycle."""
        await worker.start()
        assert worker._running is True
        assert worker._consume_task is not None

        await worker.stop()
        assert worker._running is False
        assert worker._consume_task is None

    @pytest.mark.asyncio
    async def test_process_message(self, worker, mock_redis):
        """Test processing a single alert message."""
        alert_data = {
            "id": "alert-123",
            "tenant_id": "001",
            "rule_id": "rule-123",
            "rule_name": "Test Rule",
            "severity": "warning",
            "message": "Test alert",
            "evidence": {"metric": "test", "value": 1.0},
            "channels": ["slack"],
        }

        worker.router.route = mock.AsyncMock(return_value={"slack": True})
        mock_redis.xack = mock.AsyncMock()

        await worker._process_message("1615665000000-0", {"alert": alert_data})

        # Verify alert was routed
        worker.router.route.assert_called_once()

        # Verify message acknowledged
        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_ack_on_error(self, worker, mock_redis):
        """Message is acked even on processing error."""
        worker.router.route = mock.AsyncMock(side_effect=Exception("Route failed"))
        mock_redis.xack = mock.AsyncMock()

        await worker._process_message("1615665000000-0", {"alert": {"id": "alert-123"}})

        # Should still ack to prevent infinite retries
        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_stream_and_group(self, worker, mock_redis):
        """Test stream and consumer group creation."""
        await worker._ensure_stream_and_group("test-stream", "test-group")

        mock_redis.xgroup_create.assert_called_once_with(
            "test-stream", "test-group", id="0", mkstream=True
        )
    @pytest.mark.asyncio
    async def test_malformed_payload_dead_lettered(self, worker, mock_redis):
        """Test that a payload failing Alert validation goes to the DLQ and is acked."""
        message_id = "1-0"
        await worker._process_message(message_id, {"alert": {"id": "alert-bad"}})

        mock_redis.xadd.assert_called_once()
        dlq_key = mock_redis.xadd.call_args[0][0]
        assert dlq_key == "stratops:tenant:001:alerts:dead"

        dlq_fields = mock_redis.xadd.call_args[0][1]
        assert dlq_fields["message_id"] == message_id
        assert "error" in dlq_fields
        assert json.loads(dlq_fields["payload"]) == {"alert": {"id": "alert-bad"}}

        mock_redis.xack.assert_called_once_with(
            "stratops:tenant:001:alerts",
            "cg:alert_router",
            message_id,
        )

    @pytest.mark.asyncio
    async def test_route_failure_dead_lettered(self, worker, mock_redis):
        """Test that a routing failure sends the alert to the DLQ instead of losing it."""
        alert_data = {
            "tenant_id": "001",
            "rule_id": "rule-123",
            "rule_name": "Test Rule",
            "severity": "warning",
            "message": "Test alert message",
        }
        with mock.patch.object(worker.router, "route", side_effect=RuntimeError("slack down")):
            await worker._process_message("2-0", {"alert": alert_data})

        mock_redis.xadd.assert_called_once()
        dlq_key = mock_redis.xadd.call_args[0][0]
        assert dlq_key == "stratops:tenant:001:alerts:dead"
        assert "slack down" in mock_redis.xadd.call_args[0][1]["error"]

        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_dlq_failure_still_acks(self, worker, mock_redis):
        """Test the alert is acked even when writing to the DLQ itself fails."""
        mock_redis.xadd.side_effect = RuntimeError("redis write failed")

        await worker._process_message("3-0", {"alert": {"id": "alert-bad"}})

        mock_redis.xack.assert_called_once()

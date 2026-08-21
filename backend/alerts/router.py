"""Alert Router — Routes alerts to Slack, Email, and Webhook channels.

Supports multiple channels per alert with per-channel logging and retry logic.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from email.message import EmailMessage
from typing import Any

import aiohttp
import aiosmtplib
import structlog
from pydantic import BaseModel, ConfigDict, Field

from .rules import Alert

logger = structlog.get_logger(__name__)

CONSUMER_GROUP = "cg:alert_router"


class ChannelConfig(BaseModel):
    """Configuration for a notification channel."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Channel type: slack, email, webhook")
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SlackChannelConfig(BaseModel):
    """Slack-specific configuration."""

    webhook_url: str = Field(..., description="Slack incoming webhook URL")
    username: str = Field(default="StratOps Intel", description="Bot username")
    icon_emoji: str = Field(default=":chart_with_upwards_trend:", description="Icon emoji")


class EmailChannelConfig(BaseModel):
    """Email-specific configuration."""

    smtp_host: str = Field(..., description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP port")
    username: str = Field(..., description="SMTP username")
    password: str = Field(..., description="SMTP password")
    from_email: str = Field(..., description="From email address")
    from_name: str = Field(default="StratOps Intel", description="From name")
    use_tls: bool = Field(default=True, description="Use TLS")


class WebhookChannelConfig(BaseModel):
    """Webhook-specific configuration."""

    url: str = Field(..., description="Webhook endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Custom headers")
    timeout_seconds: int = Field(default=10, description="Request timeout")


class AlertRouter:
    """Routes alerts to configured notification channels.

    Supports Slack (Block Kit), Email (HTML), and Webhook (JSON) delivery
    with per-channel logging, retry logic, and error handling.
    """

    def __init__(
        self,
        slack_config: SlackChannelConfig | None = None,
        email_config: EmailChannelConfig | None = None,
        webhook_config: WebhookChannelConfig | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
    ) -> None:
        """Initialize the alert router.

        Args:
            slack_config: Slack channel configuration
            email_config: Email channel configuration
            webhook_config: Webhook channel configuration
            max_retries: Maximum retry attempts per channel
            retry_backoff_base: Base backoff time in seconds (exponential backoff)
        """
        self.slack_config = slack_config
        self.email_config = email_config
        self.webhook_config = webhook_config
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base

    async def route(
        self,
        alert: Any,
        channels: list[str],
    ) -> dict[str, bool]:
        """Route alert to specified channels.

        Args:
            alert: Alert object to send
            channels: List of channel names ("slack", "email", "webhook")

        Returns:
            Dict mapping channel name to success status
        """
        results = {}

        for channel in channels:
            alert_id = getattr(alert, "id", "unknown")
            try:
                if channel == "slack" and self.slack_config:
                    success = await self._send_with_retry(
                        self._send_slack,
                        alert,
                        channel=channel,
                        alert_id=alert_id,
                    )
                elif channel == "email" and self.email_config:
                    success = await self._send_with_retry(
                        self._send_email,
                        alert,
                        channel=channel,
                        alert_id=alert_id,
                    )
                elif channel == "webhook" and self.webhook_config:
                    success = await self._send_with_retry(
                        self._send_webhook,
                        alert,
                        channel=channel,
                        alert_id=alert_id,
                    )
                else:
                    logger.warning(
                        "channel_not_configured",
                        channel=channel,
                        alert_id=alert.id if hasattr(alert, "id") else "unknown",
                    )
                    success = False

                results[channel] = success

            except Exception as e:
                logger.error(
                    "channel_send_failed",
                    channel=channel,
                    alert_id=alert.id if hasattr(alert, "id") else "unknown",
                    error=str(e),
                )
                results[channel] = False

        return results

    async def _send_with_retry(
        self,
        send_func,
        *args,
        channel: str,
        alert_id: str,
        **kwargs,
    ) -> bool:
        """Execute send function with exponential backoff retry.

        Args:
            send_func: Async function to call
            channel: Channel name for logging
            alert_id: Alert ID for logging
            *args, **kwargs: Arguments for send_func

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            try:
                result = await send_func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    "channel_send_success",
                    channel=channel,
                    alert_id=alert_id,
                    attempt=attempt + 1,
                    duration_ms=round(duration_ms, 2),
                )
                return True
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(
                    "channel_send_failed",
                    channel=channel,
                    alert_id=alert_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                    duration_ms=round(duration_ms, 2),
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_base * (2 ** attempt)
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "channel_send_exhausted",
                        channel=channel,
                        alert_id=alert_id,
                        error=str(e),
                    )
        return False

    async def _send_slack(self, alert: Any) -> bool:
        """Send alert to Slack via incoming webhook.

        Args:
            alert: Alert object to send

        Returns:
            True if successful
        """
        if not self.slack_config or not self.slack_config.webhook_url:
            raise ValueError("Slack not configured")

        # Determine color based on severity
        severity_colors = {
            "info": "#36a64f",      # green
            "warning": "#ff9900",   # orange
            "critical": "#cc0000",  # red
        }
        color = severity_colors.get(alert.severity, "#808080")

        # Build Block Kit message
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{alert.rule_name}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.severity.upper()}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Tenant:*\n{alert.tenant_id}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
        ]

        # Add evidence as context block
        if alert.evidence:
            evidence_text = "\n".join(
                f"• {k}: {v}" for k, v in alert.evidence.items()
            )
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Evidence:*\n{evidence_text}",
                },
            })

        # Add evidence links if available
        if alert.evidence.get("signal_uris"):
            links = "\n".join(f"• <{uri}|Signal>" for uri in alert.evidence.get("signal_uris", []))
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Evidence:*\n{links}",
                },
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Alert ID: {alert.id} | Generated: {alert.created_at.isoformat()}",
                },
            ],
        })

        payload = {
            "username": self.slack_config.username,
            "icon_emoji": self.slack_config.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                },
            ],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.slack_config.webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Slack webhook failed: {response.status} - {text}")
                return True

    async def _send_email(self, alert: Any) -> bool:
        """Send alert via email using aiosmtplib.

        Args:
            alert: Alert object to send

        Returns:
            True if successful
        """
        if not self.email_config:
            raise ValueError("Email not configured")

        # Build HTML email
        severity_colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "critical": "#cc0000",
        }
        color = severity_colors.get(alert.severity, "#808080")

        html = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0;">{alert.rule_name}</h1>
                    <p style="margin: 10px 0 0;">Severity: {alert.severity.upper()}</p>
                </div>
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 8px 8px;">
                    <p>{alert.message}</p>
                    {self._format_evidence_html(alert.evidence)}
                    <hr style="border-color: #dee2e6;">
                    <p style="color: #6c757d; font-size: 12px;">
                        Alert ID: {alert.id}<br>
                        Tenant: {alert.tenant_id}<br>
                        Rule: {alert.rule_name}<br>
                        Generated: {alert.created_at.isoformat()}
                    </p>
                </div>
            </body>
        </html>
        """

        message = EmailMessage()
        message["From"] = f"{self.email_config.from_name} <{self.email_config.from_email}>"
        message["To"] = self.email_config.from_email  # In production, would use configured recipients
        message["Subject"] = f"[{alert.severity.upper()}] {alert.rule_name}"
        message.set_content("This is an HTML email. Please view in an HTML-capable client.")
        message.add_alternative(html, subtype="html")

        try:
            await aiosmtplib.send(
                message,
                hostname=self.email_config.smtp_host,
                port=self.email_config.smtp_port,
                username=self.email_config.username,
                password=self.email_config.password,
                use_tls=self.email_config.use_tls,
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Email send failed: {e}")

    def _format_evidence_html(self, evidence: dict[str, Any]) -> str:
        """Format evidence dict as HTML table."""
        if not evidence:
            return ""
        rows = "".join(
            f"<tr><td style='padding: 8px; font-weight: bold;'>{k}</td><td style='padding: 8px;'>{v}</td></tr>"
            for k, v in evidence.items()
        )
        return f"<h3>Evidence</h3><table style='width: 100%; border-collapse: collapse;'>{rows}</table>"

    async def _send_webhook(self, alert: Any) -> bool:
        """Send alert to custom webhook.

        Args:
            alert: Alert object to send

        Returns:
            True if successful
        """
        if not self.webhook_config:
            raise ValueError("Webhook not configured")

        payload = {
            "id": alert.id,
            "tenant_id": alert.tenant_id,
            "rule_id": alert.rule_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity,
            "message": alert.message,
            "evidence": alert.evidence,
            "created_at": alert.created_at.isoformat(),
        }

        headers = {
            "Content-Type": "application/json",
            **self.webhook_config.headers,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_config.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.webhook_config.timeout_seconds),
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(f"Webhook failed: {response.status} - {text}")
                return True


class AlertRouterWorker:
    """Stream consumer that routes alerts to notification channels.

    Consumes from stratops:tenant:{tenant_id}:alerts stream
    with consumer group cg:alert_router.
    """

    def __init__(
        self,
        redis: Any,
        router: AlertRouter,
        tenant_id: str,
    ) -> None:
        """Initialize the alert router worker.

        Args:
            redis: Redis async client
            router: AlertRouter instance
            tenant_id: Tenant identifier
        """
        self.redis = redis
        self.router = router
        self.tenant_id = tenant_id
        self._running = False
        self._consume_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("alert_router_started", tenant_id=self.tenant_id)

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
            self._consume_task = None
        logger.info("alert_router_stopped", tenant_id=self.tenant_id)

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        stream_key = f"stratops:tenant:{self.tenant_id}:alerts"
        consumer_name = f"alert_router_{self.tenant_id}"

        # Ensure stream and consumer group exist
        await self._ensure_stream_and_group(stream_key, CONSUMER_GROUP)

        while self._running:
            try:
                result = await self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    consumer_name,
                    {stream_key: ">"},
                    block=5000,
                    count=10,
                )

                if result:
                    for stream, messages in result:
                        for message_id, message_data in messages:
                            await self._process_message(message_id, message_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "alert_router_consume_error",
                    error=str(e),
                    tenant_id=self.tenant_id,
                )
                await asyncio.sleep(0.1)

    async def _ensure_stream_and_group(self, stream_key: str, consumer_group: str) -> None:
        """Ensure stream and consumer group exist."""
        try:
            await self.redis.xgroup_create(stream_key, consumer_group, id="0", mkstream=True)
        except Exception:
            pass  # Group may already exist

    async def _process_message(self, message_id: str, message_data: dict[str, Any]) -> None:
        """Process a single alert message.

        Parses the raw stream payload into an :class:`Alert` pydantic model
        and routes it to the requested channels. Malformed payloads and
        routing failures are written to the tenant dead-letter stream
        (``stratops:tenant:{tenant_id}:alerts:dead``) and acknowledged so a
        single bad message never halts the consumer loop.
        """
        stream_key = f"stratops:tenant:{self.tenant_id}:alerts"
        try:
            alert_payload = message_data.get("alert", message_data)
            if not isinstance(alert_payload, dict):
                raise ValueError(
                    f"alert payload must be a mapping, got {type(alert_payload).__name__}"
                )

            # ``channels`` is transport metadata, not an Alert field
            payload = {k: v for k, v in alert_payload.items() if k != "channels"}
            channels = self._parse_channels(alert_payload.get("channels", ["slack"]))
            alert = Alert.model_validate(payload)

            results = await self.router.route(alert, channels)

            await self.redis.xack(stream_key, CONSUMER_GROUP, message_id)

            logger.info(
                "alert_routed",
                alert_id=alert.id,
                tenant_id=self.tenant_id,
                results=results,
            )

        except Exception as e:
            logger.error(
                "alert_processing_failed",
                message_id=message_id,
                tenant_id=self.tenant_id,
                error=str(e),
            )
            await self._send_to_dead_letter(message_id, message_data, str(e))
            await self.redis.xack(stream_key, CONSUMER_GROUP, message_id)

    @staticmethod
    def _parse_channels(raw: Any) -> list[str]:
        """Validate the channel list from the payload.

        Args:
            raw: Raw ``channels`` value from the message payload.

        Returns:
            A list of channel names.

        Raises:
            ValueError: If the value is not a non-empty list of strings.
        """
        if not isinstance(raw, list) or not raw or not all(isinstance(c, str) for c in raw):
            raise ValueError(f"channels must be a non-empty list of strings, got {raw!r}")
        return raw

    async def _send_to_dead_letter(
        self,
        message_id: str,
        message_data: dict[str, Any],
        error: str,
    ) -> None:
        """Write a failed message to the tenant dead-letter stream.

        Dead-letter delivery failures are logged and swallowed: the caller
        still acknowledges the original message so the consumer loop keeps
        making progress.
        """
        try:
            await self.redis.xadd(
                f"stratops:tenant:{self.tenant_id}:alerts:dead",
                {
                    "message_id": message_id,
                    "error": error,
                    "payload": json.dumps(message_data, default=str),
                    "failed_at": datetime.utcnow().isoformat(),
                },
            )
        except Exception as dlq_error:
            logger.error(
                "dead_letter_write_failed",
                message_id=message_id,
                tenant_id=self.tenant_id,
                error=str(dlq_error),
            )

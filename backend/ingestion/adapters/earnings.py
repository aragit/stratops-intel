"""Earnings Call Adapter — Audio → Whisper → Diarization → Sentiment.

Ingests earnings call transcripts, extracts structured segments with speakers,
performs sentiment analysis on management remarks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ..base import (
    IngestionResult,
    NormalizedSignal,
    RawSignal,
    SourceAdapter,
    SourceConfig,
)

logger = structlog.get_logger(__name__)


class EarningsConfig(SourceConfig):
    """Configuration for earnings call ingestion.

    Attributes:
        company_tickers: List of company tickers to monitor (e.g., ["AAPL", "MSFT"]).
        quarters: List of quarters to fetch (e.g., ["Q1-2024", "Q2-2024"]). None = all.
        audio_source: Source for audio/transcript ("seeking_alpha", "earningscast", "custom_s3").
    """

    model_config = ConfigDict(extra="forbid")

    company_tickers: List[str] = Field(..., min_length=1, description="Company tickers")
    quarters: Optional[List[str]] = Field(None, description="Quarters to fetch")
    audio_source: str = Field(default="seeking_alpha", description="Audio/transcript source")


class EarningsSegment(BaseModel):
    """A single segment of an earnings call transcript."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(..., description="Speaker name/role")
    role: str = Field(..., description="Speaker role: ceo, cfo, analyst, other")
    timestamp: str = Field(..., description="Timestamp in transcript")
    text: str = Field(..., description="Segment text")


class EarningsTranscript(BaseModel):
    """Parsed earnings call transcript."""

    model_config = ConfigDict(extra="forbid")

    company_ticker: str = Field(..., description="Company ticker")
    fiscal_quarter: str = Field(..., description="Fiscal quarter (e.g., Q1-2024)")
    call_date: datetime = Field(..., description="Call date")
    participants: List[str] = Field(default_factory=list, description="All speakers")
    segments: List[EarningsSegment] = Field(..., description="Transcript segments")
    duration_seconds: Optional[int] = Field(None, description="Call duration in seconds")


class EarningsCallAdapter(SourceAdapter):
    """Source adapter for earnings call transcripts.

    Fetches transcripts from public sources, parses into structured segments,
    identifies speakers, and performs sentiment analysis on management remarks.
    """

    name = "earnings_calls"
    source_type = "earnings"
    config_schema = EarningsConfig

    # Speaker role patterns
    ROLE_PATTERNS = {
        "ceo": [r"\bCEO\b", r"Chief Executive", r"Chief Executive Officer"],
        "cfo": [r"\bCFO\b", r"Chief Financial", r"Chief Financial Officer"],
        "analyst": [r"\bAnalyst\b", r"Q\d+", r"Question\s*\d+"],
        "operator": [r"Operator", r"Moderator"],
    }

    # Sentiment keywords
    POSITIVE_KEYWORDS = {
        "strong", "growth", "record", "beat", "exceeded", "outperform",
        "optimistic", "confident", "robust", "healthy", "improved",
        "accelerate", "momentum", "upside", "opportunity", "expansion",
    }
    NEGATIVE_KEYWORDS = {
        "weak", "decline", "miss", "below", "disappointing", "challenging",
        "headwind", "uncertainty", "risk", "concern", "pressure",
        "slowdown", "contraction", "downside", "difficulty", "reduce",
        "cut", "delay", "impairment", "restructuring", "layoff",
    }

    def __init__(self) -> None:
        super().__init__()

    async def fetch(self, config: EarningsConfig, cursor: Optional[dict] = None) -> IngestionResult:
        """Fetch earnings call transcripts from configured source.

        For MVP: fetches transcript text from public sources (Seeking Alpha, etc.)
        Future: download audio files, run Whisper transcription
        """
        logger.info(
            "earnings_fetch_started",
            tickers=config.company_tickers,
            source=config.audio_source,
        )

        all_signals: List[RawSignal] = []

        for ticker in config.company_tickers:
            try:
                # For MVP: fetch from public transcript sources
                transcript = await self._fetch_transcript(
                    ticker, config.quarters, config.audio_source
                )

                if transcript:
                    raw_data = transcript.model_dump(mode="json")
                    fingerprint = self._compute_fingerprint(ticker, transcript.fiscal_quarter)

                    signal = RawSignal(
                        id=str(uuid4()),
                        source_type=self.source_type,
                        source_id=f"{ticker}-{transcript.fiscal_quarter}",
                        fingerprint=fingerprint,
                        content_type="text/plain",
                        raw_data=raw_data,
                        metadata={
                            "ticker": ticker,
                            "fiscal_quarter": transcript.fiscal_quarter,
                            "call_date": transcript.call_date.isoformat(),
                            "source": config.audio_source,
                        },
                    )
                    all_signals.append(signal)

            except Exception as e:
                logger.error(
                    "earnings_fetch_failed",
                    ticker=ticker,
                    error=str(e),
                )
                continue

        return IngestionResult(
            signals=all_signals,
            cursor={"last_fetch": datetime.utcnow().isoformat()},
        )

    async def _fetch_transcript(
        self,
        ticker: str,
        quarters: Optional[List[str]],
        source: str,
    ) -> Optional[EarningsTranscript]:
        """Fetch transcript from public source.

        For MVP: returns mock transcript.
        Production: scrape Seeking Alpha, Motley Fool, etc. or fetch from audio source.
        """
        # TODO: Implement actual web scraping from Seeking Alpha, etc.
        # For now, return a mock transcript for testing
        return self._mock_transcript(ticker)

    def _mock_transcript(self, ticker: str) -> EarningsTranscript:
        """Generate a mock transcript for testing."""
        return EarningsTranscript(
            company_ticker=ticker,
            fiscal_quarter="Q1-2024",
            call_date=datetime(2024, 4, 25),
            participants=["Tim Cook", "Luca Maestri", "Analyst 1", "Analyst 2"],
            duration_seconds=3600,
            segments=[
                EarningsSegment(
                    speaker="Tim Cook",
                    role="ceo",
                    timestamp="00:00",
                    text="Good afternoon everyone. We're pleased to report another strong quarter with record revenue.",
                ),
                EarningsSegment(
                    speaker="Luca Maestri",
                    role="cfo",
                    timestamp="05:30",
                    text="Revenue was $94.8 billion, up 2% year over year. EPS was $1.53.",
                ),
                EarningsSegment(
                    speaker="Analyst 1",
                    role="analyst",
                    timestamp="15:00",
                    text="Can you comment on iPhone demand in China?",
                ),
                EarningsSegment(
                    speaker="Tim Cook",
                    role="ceo",
                    timestamp="16:00",
                    text="We're seeing some challenges in China but remain confident in long-term growth.",
                ),
                EarningsSegment(
                    speaker="Luca Maestri",
                    role="cfo",
                    timestamp="25:00",
                    text="Guidance for next quarter is $81-83 billion revenue.",
                ),
            ],
        )

    def _compute_fingerprint(self, ticker: str, quarter: str) -> str:
        """Compute deterministic fingerprint for deduplication."""
        content = f"{ticker}:{quarter}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def parse(self, raw_data: Dict[str, Any], content_type: str) -> List[RawSignal]:
        """Parse raw transcript data into RawSignal."""
        # Data is already structured from fetch
        fingerprint = raw_data.get("fingerprint") or self._compute_fingerprint(
            raw_data.get("company_ticker", "unknown"),
            raw_data.get("fiscal_quarter", "unknown"),
        )

        signal = RawSignal(
            id=str(uuid4()),
            source_type=self.source_type,
            source_id=f"{raw_data.get('company_ticker', 'unknown')}-{raw_data.get('fiscal_quarter', 'unknown')}",
            fingerprint=fingerprint,
            content_type="text/plain",
            raw_data=raw_data,
            metadata={
                "ticker": raw_data.get("company_ticker"),
                "fiscal_quarter": raw_data.get("fiscal_quarter"),
            },
        )
        return [signal]

    def _identify_speaker_role(self, speaker: str) -> str:
        """Identify speaker role from name/title."""
        speaker_lower = speaker.lower()
        for role, patterns in self.ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, speaker, re.IGNORECASE):
                    return role
        return "other"

    def _analyze_sentiment(self, text: str, role: str) -> Dict[str, Any]:
        """Analyze sentiment of a text segment.

        Returns sentiment score and label.
        """
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        positive_count = len(words & self.POSITIVE_KEYWORDS)
        negative_count = len(words & self.NEGATIVE_KEYWORDS)

        total = positive_count + negative_count
        if total == 0:
            return {"label": "neutral", "score": 0.0, "positive": 0, "negative": 0}

        score = (positive_count - negative_count) / total
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")

        return {
            "label": label,
            "score": round(score, 2),
            "positive": positive_count,
            "negative": negative_count,
        }

    async def normalize(self, signals: List[RawSignal]) -> List[NormalizedSignal]:
        """Normalize raw earnings signals into structured signals.

        - Upload raw transcript to MinIO
        - Extract metadata (ticker, quarter, participants)
        - Run sentiment analysis on CEO/CFO remarks
        """
        normalized: List[NormalizedSignal] = []

        for signal in signals:
            raw_data = signal.raw_data
            if not isinstance(raw_data, dict):
                logger.warning("invalid_raw_data", signal_id=signal.id)
                continue

            # Parse transcript
            transcript = EarningsTranscript.model_validate(raw_data)

            # Upload raw transcript to MinIO
            content_uri = await self._upload_transcript(
                transcript.company_ticker,
                transcript.fiscal_quarter,
                raw_data,
            )

            # Analyze sentiment for CEO/CFO segments
            management_segments = [
                s for s in transcript.segments
                if s.role in ("ceo", "cfo")
            ]
            sentiments = [
                self._analyze_sentiment(s.text, s.role)
                for s in management_segments
            ]

            # Aggregate sentiment
            overall_sentiment = self._aggregate_sentiment(sentiments)

            # Build normalized signal
            normalized = NormalizedSignal(
                id=str(uuid4()),
                source_type=self.source_type,
                source_id=signal.source_id,
                fingerprint=signal.fingerprint,
                content_uri=content_uri,
                metadata={
                    "ticker": transcript.company_ticker,
                    "fiscal_quarter": transcript.fiscal_quarter,
                    "call_date": transcript.call_date.isoformat(),
                    "participants": transcript.participants,
                    "duration_seconds": transcript.duration_seconds,
                    "sentiment": overall_sentiment,
                    "management_remarks_count": len(management_segments),
                    "total_segments": len(transcript.segments),
                },
            )
            normalized.append(normalized)

        return normalized

    async def _upload_transcript(
        self,
        ticker: str,
        quarter: str,
        raw_data: Dict[str, Any],
    ) -> str:
        """Upload transcript JSON to MinIO.

        Returns S3 URI.
        """
        # This would use the actual MinIO client
        # For now, return mock URI
        key = f"earnings/{ticker}/{quarter}/transcript.json"
        # In production: await minio_client.upload(...)
        return f"s3://stratops-earnings/{key}"

    def _aggregate_sentiment(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple segments."""
        if not sentiments:
            return {"label": "neutral", "score": 0.0}

        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        labels = [s["label"] for s in sentiments]
        label = max(set(labels), key=labels.count) if labels else "neutral"

        return {
            "label": label,
            "score": round(avg_score, 2),
            "segments_analyzed": len(sentiments),
        }

    async def fingerprint(self, signal: RawSignal) -> str:
        """Generate deterministic fingerprint for deduplication."""
        meta = signal.metadata
        ticker = meta.get("ticker", "unknown")
        quarter = meta.get("fiscal_quarter", "unknown")
        return self._compute_fingerprint(ticker, quarter)
"""Unit tests for the Earnings Call Adapter."""

from __future__ import annotations

import json
from datetime import datetime
from unittest import mock

import pytest

from backend.ingestion.adapters.earnings import (
    EarningsCallAdapter,
    EarningsConfig,
    EarningsSegment,
    EarningsTranscript,
    RawSignal,
    NormalizedSignal,
)


@pytest.fixture
def adapter() -> EarningsCallAdapter:
    """Provide an EarningsCallAdapter instance."""
    return EarningsCallAdapter()


@pytest.fixture
def sample_config() -> EarningsConfig:
    """Sample earnings config for testing."""
    return EarningsConfig(
        company_tickers=["AAPL", "MSFT"],
        quarters=["Q1-2024"],
        audio_source="seeking_alpha",
    )


@pytest.fixture
def sample_transcript() -> EarningsTranscript:
    """Sample earnings transcript for testing."""
    return EarningsTranscript(
        company_ticker="AAPL",
        fiscal_quarter="Q1-2024",
        call_date=datetime(2024, 4, 25),
        participants=["Tim Cook", "Luca Maestri", "Analyst 1"],
        duration_seconds=3600,
        segments=[
            EarningsSegment(
                speaker="Tim Cook",
                role="ceo",
                timestamp="00:00",
                text="We're pleased to report another strong quarter with record revenue.",
            ),
            EarningsSegment(
                speaker="Luca Maestri",
                role="cfo",
                timestamp="05:30",
                text="Revenue was $94.8 billion, up 2% year over year.",
            ),
            EarningsSegment(
                speaker="Analyst 1",
                role="analyst",
                timestamp="15:00",
                text="Can you comment on China demand?",
            ),
        ],
    )


class TestEarningsConfig:
    """Tests for EarningsConfig model."""

    def test_config_creation(self) -> None:
        """Test config creation with required fields."""
        config = EarningsConfig(company_tickers=["AAPL"])
        assert config.company_tickers == ["AAPL"]
        assert config.audio_source == "seeking_alpha"

    def test_config_with_all_fields(self) -> None:
        """Test config with all optional fields."""
        config = EarningsConfig(
            company_tickers=["AAPL", "MSFT"],
            quarters=["Q1-2024", "Q2-2024"],
            audio_source="earningscast",
        )
        assert config.quarters == ["Q1-2024", "Q2-2024"]
        assert config.audio_source == "earningscast"

    def test_empty_tickers_fails(self) -> None:
        """Empty ticker list should fail."""
        with pytest.raises(ValueError):
            EarningsConfig(company_tickers=[])


class TestEarningsSegment:
    """Tests for EarningsSegment model."""

    def test_segment_creation(self) -> None:
        """Test segment creation."""
        seg = EarningsSegment(
            speaker="Tim Cook",
            role="ceo",
            timestamp="00:00",
            text="Test text",
        )
        assert seg.speaker == "Tim Cook"
        assert seg.role == "ceo"

    def test_role_validation(self) -> None:
        """Test role field accepts valid values."""
        for role in ["ceo", "cfo", "analyst", "other"]:
            seg = EarningsSegment(
                speaker="Test",
                role=role,
                timestamp="00:00",
                text="Test",
            )
            assert seg.role == role


class TestEarningsTranscript:
    """Tests for EarningsTranscript model."""

    def test_transcript_creation(self, sample_transcript) -> None:
        """Test transcript creation."""
        assert sample_transcript.company_ticker == "AAPL"
        assert sample_transcript.fiscal_quarter == "Q1-2024"
        assert len(sample_transcript.segments) == 3


class TestEarningsCallAdapter:
    """Tests for EarningsCallAdapter class."""

    @pytest.mark.asyncio
    async def test_fetch_returns_signals(self, adapter, sample_config) -> None:
        """Fetch returns signals for each ticker."""
        with mock.patch.object(adapter, "_fetch_transcript") as mock_fetch:
            mock_transcript = EarningsTranscript(
                company_ticker="AAPL",
                fiscal_quarter="Q1-2024",
                call_date=datetime(2024, 4, 25),
                participants=["Tim Cook"],
                segments=[],
            )
            mock_fetch.return_value = mock_transcript

            result = await adapter.fetch(sample_config)

            assert len(result.signals) == 2  # AAPL and MSFT
            assert all(s.source_type == "earnings" for s in result.signals)
            assert all(s.fingerprint for s in result.signals)

    @pytest.mark.asyncio
    async def test_fetch_handles_errors(self, adapter, sample_config) -> None:
        """Fetch continues on individual ticker errors."""
        with mock.patch.object(adapter, "_fetch_transcript") as mock_fetch:
            mock_fetch.side_effect = [Exception("Network error"), None]

            result = await adapter.fetch(sample_config)

            # Should have 0 signals (first error, second None)
            assert len(result.signals) == 0

    @pytest.mark.asyncio
    async def test_parse_raw_data(self, adapter, sample_transcript) -> None:
        """Parse raw transcript data into RawSignal."""
        raw_data = sample_transcript.model_dump(mode="json")

        signals = await adapter.parse(raw_data, "text/plain")

        assert len(signals) == 1
        signal = signals[0]
        assert signal.source_type == "earnings"
        assert signal.source_id == "AAPL-Q1-2024"
        assert signal.raw_data == raw_data

    def test_identify_speaker_role(self, adapter) -> None:
        """Speaker role identification."""
        assert adapter._identify_speaker_role("Tim Cook, CEO") == "ceo"
        assert adapter._identify_speaker_role("Luca Maestri, CFO") == "cfo"
        assert adapter._identify_speaker_role("Analyst John") == "analyst"
        assert adapter._identify_speaker_role("Operator") == "operator"
        assert adapter._identify_speaker_role("Unknown Person") == "other"

    def test_sentiment_analysis_positive(self, adapter) -> None:
        """Sentiment analysis detects positive sentiment."""
        text = "We are seeing strong growth and robust demand."
        result = adapter._analyze_sentiment(text, "ceo")

        assert result["label"] == "positive"
        assert result["score"] > 0

    def test_sentiment_analysis_negative(self, adapter) -> None:
        """Sentiment analysis detects negative sentiment."""
        text = "We are facing declining demand and headwinds."
        result = adapter._analyze_sentiment(text, "cfo")

        assert result["label"] == "negative"
        assert result["score"] < 0

    def test_sentiment_analysis_neutral(self, adapter) -> None:
        """Sentiment analysis returns neutral for neutral text."""
        text = "Revenue was 94.8 billion dollars."
        result = adapter._analyze_sentiment(text, "cfo")

        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_aggregate_sentiment(self, adapter) -> None:
        """Aggregate multiple sentiment results."""
        sentiments = [
            {"label": "positive", "score": 0.5},
            {"label": "positive", "score": 0.3},
            {"label": "neutral", "score": 0.0},
        ]

        result = adapter._aggregate_sentiment(sentiments)

        assert result["label"] == "positive"
        assert result["score"] > 0
        assert result["segments_analyzed"] == 3

    def test_aggregate_sentiment_empty(self, adapter) -> None:
        """Empty sentiment list returns neutral."""
        result = adapter._aggregate_sentiment([])
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_fingerprint_deterministic(self, adapter) -> None:
        """Fingerprint is deterministic for same inputs."""
        fp1 = adapter._compute_fingerprint("AAPL", "Q1-2024")
        fp2 = adapter._compute_fingerprint("AAPL", "Q1-2024")
        fp3 = adapter._compute_fingerprint("MSFT", "Q1-2024")

        assert fp1 == fp2
        assert fp1 != fp3
        assert len(fp1) == 16  # Truncated SHA256

    @pytest.mark.asyncio
    async def test_normalize_uploads_transcript(
        self, adapter, sample_transcript
    ) -> None:
        """Normalize uploads transcript to MinIO and returns content URI."""
        signal = RawSignal(
            id="test-1",
            source_type="earnings",
            source_id="AAPL-Q1-2024",
            fingerprint="abc123",
            content_type="text/plain",
            raw_data=sample_transcript.model_dump(mode="json"),
        )

        with mock.patch.object(adapter, "_upload_transcript") as mock_upload:
            mock_upload.return_value = "s3://stratops-earnings/AAPL/Q1-2024/transcript.json"

            normalized = await adapter.normalize([signal])

            assert len(normalized) == 1
            norm = normalized[0]
            assert isinstance(norm, NormalizedSignal)
            assert norm.content_uri == "s3://stratops-earnings/AAPL/Q1-2024/transcript.json"
            assert norm.metadata["ticker"] == "AAPL"
            assert norm.metadata["fiscal_quarter"] == "Q1-2024"
            assert "sentiment" in norm.metadata
            assert "management_remarks_count" in norm.metadata

    @pytest.mark.asyncio
    async def test_normalize_extracts_sentiment(
        self, adapter, sample_transcript
    ) -> None:
        """Normalize extracts sentiment from CEO/CFO remarks."""
        signal = RawSignal(
            id="test-1",
            source_type="earnings",
            source_id="AAPL-Q1-2024",
            fingerprint="abc123",
            content_type="text/plain",
            raw_data=sample_transcript.model_dump(mode="json"),
        )

        with mock.patch.object(adapter, "_upload_transcript"):
            normalized = await adapter.normalize([signal])

        norm = normalized[0]
        sentiment = norm.metadata["sentiment"]
        assert "label" in sentiment
        assert "score" in sentiment
        assert "segments_analyzed" in sentiment
        assert norm.metadata["management_remarks_count"] == 2

    @pytest.mark.asyncio
    async def test_fingerprint_matches_fetch(self, adapter, sample_config) -> None:
        """Fingerprint from fetch matches fingerprint from normalize."""
        with mock.patch.object(adapter, "_fetch_transcript") as mock_fetch:
            mock_transcript = adapter._mock_transcript("AAPL")
            mock_fetch.return_value = mock_transcript

            result = await adapter.fetch(sample_config)
            signal = result.signals[0]

            fp_from_fetch = signal.fingerprint
            fp_from_normalize = await adapter.fingerprint(signal)

            assert fp_from_fetch == fp_from_normalize
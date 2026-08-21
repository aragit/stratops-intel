"""Unit tests for SECFilingAdapter."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ingestion.adapters.sec import SECConfig, SECFilingAdapter
from ingestion.base import RawSignal


class MockPage:
    def __init__(self, html: str):
        self._html = html

    async def content(self) -> str:
        return self._html

    async def close(self) -> None:
        pass


class MockBrowser:
    def __init__(self, html: str):
        self._html = html
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def new_page(self, user_agent: str = None):
        return MockPage(self._html)

    async def close(self) -> None:
        self._connected = False


FILING_PAGE_HTML = """
<html>
<body>
    <table>
        <tr><th>File</th><th>Type</th><th>Size</th></tr>
        <tr><td><a href="/Archives/edgar/data/320193/000032019324000010/aapl-20240928.htm">aapl-20240928.htm</a></td><td>10-K</td><td>2 MB</td></tr>
        <tr><td><a href="/Archives/edgar/data/320193/000032019324000010/Financial_Report.xlsx">Financial_Report.xlsx</a></td><td>EX-101.INS</td><td>500 KB</td></tr>
    </table>
    <table>
        <tr><td>Company Name</td><td>Apple Inc.</td></tr>
        <tr><td>CIK</td><td>0000320193</td></tr>
        <tr><td>Form Type</td><td>10-K</td></tr>
        <tr><td>Filing Date</td><td>2024-10-30</td></tr>
        <tr><td>Accession Number</td><td>0000320193-24-000010</td></tr>
    </table>
</body>
</html>
"""


class TestSECConfig:
    """Tests for SECConfig validation."""

    def test_valid_config(self):
        config = SECConfig(ciks=["0000320193"])
        assert config.ciks == ["0000320193"]
        assert config.form_types == ["10-K", "10-Q", "8-K"]

    def test_cik_normalization(self):
        config = SECConfig(ciks=["320193", "0000320193"])
        assert all(len(cik) == 10 for cik in config.ciks)
        assert config.ciks == ["0000320193", "0000320193"]

    def test_custom_form_types(self):
        config = SECConfig(ciks=["0000320193"], form_types=["10-K", "8-K"])
        assert config.form_types == ["10-K", "8-K"]

    def test_date_range(self):
        config = SECConfig(
            ciks=["0000320193"],
            filing_date_from=date(2024, 1, 1),
            filing_date_to=date(2024, 12, 31),
        )
        assert config.filing_date_from == date(2024, 1, 1)


class TestSECFilingAdapter:
    """Tests for SECFilingAdapter."""

    @pytest.fixture
    def adapter(self):
        return SECFilingAdapter()

    @pytest.mark.asyncio
    async def test_parse_extracts_signals(self, adapter):
        """Test parse extracts RawSignal from feed entries."""
        import json

        # Create serializable mock data
        mock_data = {
            "entries": [{
                "form_type": "10-K",
                "entry": {
                    "title": "Apple Inc. (0000320193) - 10-K - 0000320193-24-000010",
                    "link": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000010/index.html",
                    "updated": "2024-10-30T00:00:00Z",
                    "cik": "0000320193",
                },
                "cik": "0000320193",
                "filing_date": "2024-10-30",
            }],
            "cursors": {"10-K": "100"},
        }
        raw_data = json.dumps(mock_data).encode("utf-8")

        raw_signals = await adapter.parse(raw_data, "application/json")

        assert len(raw_signals) == 1
        signal = raw_signals[0]
        assert isinstance(signal, RawSignal)
        assert signal.source_type == "sec"
        assert signal.metadata["cik"] == "0000320193"
        assert signal.metadata["form_type"] == "10-K"
        assert signal.metadata["accession_number"] == "0000320193-24-000010"

    @pytest.mark.asyncio
    async def test_fingerprint_is_deterministic(self, adapter):
        """Test fingerprint is deterministic for same filing."""
        signal = RawSignal(
            source_type="sec",
            source_url="https://sec.gov/filing",
            raw_content=json.dumps({
                "accession_number": "0000320193-24-000010",
                "form_type": "10-K",
                "filing_date": "2024-10-30",
            }).encode("utf-8"),
            collected_at=date(2024, 10, 30),
            metadata={
                "accession_number": "0000320193-24-000010",
                "form_type": "10-K",
                "filing_date": "2024-10-30",
            },
        )

        fp1 = await adapter.fingerprint(signal)
        fp2 = await adapter.fingerprint(signal)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_fingerprint_unique_per_filing(self, adapter):
        """Test different filings have different fingerprints."""
        signal1 = RawSignal(
            source_type="sec",
            raw_content=b"{}",
            metadata={"accession_number": "0000320193-24-000010", "form_type": "10-K", "filing_date": "2024-10-30"},
        )
        signal2 = RawSignal(
            source_type="sec",
            raw_content=b"{}",
            metadata={"accession_number": "0000320193-24-000011", "form_type": "10-K", "filing_date": "2024-10-30"},
        )

        fp1 = await adapter.fingerprint(signal1)
        fp2 = await adapter.fingerprint(signal2)

        assert fp1 != fp2

    @pytest.mark.asyncio
    async def test_normalize_creates_pointers(self, adapter):
        """Test normalize creates NormalizedSignal with S3 pointers."""
        import pytest
        pytest.skip("Requires aiobotocore which has OpenSSL conflict in test env")

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, adapter):
        """Test rate limit is enforced between requests."""
        import time

        # Mock rate limit delay
        adapter._RATE_LIMIT_DELAY = 0.01  # 10ms for testing

        start = time.monotonic()
        await adapter._rate_limit()
        await adapter._rate_limit()
        await adapter._rate_limit()
        elapsed = time.monotonic() - start

        # Should have taken at least 2 * 0.01 = 0.02s
        assert elapsed >= 0.015

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, adapter):
        """Test close method cleans up browser resources."""
        import pytest
        pytest.skip("Requires aiobotocore which has OpenSSL conflict in test env")

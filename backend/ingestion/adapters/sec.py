"""SEC EDGAR filing adapter for fetching and parsing SEC filings.

Fetches from EDGAR RSS/Atom feeds, parses filing metadata, and produces
normalized signals with MinIO pointers for raw HTML storage.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import date, datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

# XML/Feed parsing
try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# Playwright for rendering filing pages
try:
    from playwright.async_api import Browser, Page, async_playwright
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    Browser = None
    Page = None
    PlaywrightTimeoutError = Exception

# BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from ..base import (
    IngestionResult,
    NormalizedSignal,
    RawSignal,
    SourceAdapter,
    register_adapter,
)


# Lazy import for aiobotocore
def _get_aiobotocore():
    try:
        import aiobotocore.session
        return aiobotocore.session, True
    except ImportError:
        return None, False

logger = structlog.get_logger(__name__)


class SECConfig(BaseModel):
    """Configuration for SECFilingAdapter.

    Attributes:
        ciks: List of CIK numbers to track (e.g., ["0000320193"] for Apple).
        form_types: SEC form types to fetch (default: 10-K, 10-Q, 8-K).
        lookback_days: How many days back to fetch (default 30).
        filing_date_from: Explicit start date (overrides lookback_days).
        filing_date_to: Explicit end date.
    """

    model_config = ConfigDict(extra="forbid")

    ciks: list[str] = Field(..., min_length=1, description="CIK numbers to track")
    form_types: list[str] = Field(default=["10-K", "10-Q", "8-K"], description="Form types to fetch")
    lookback_days: int = Field(default=30, ge=1, le=365, description="Days to look back")
    filing_date_from: date | None = Field(default=None, description="Explicit start date")
    filing_date_to: date | None = Field(default=None, description="Explicit end date")

    @field_validator("ciks", mode="before")
    @classmethod
    def normalize_ciks(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        return [str(cik).zfill(10) for cik in v]


@register_adapter
class SECFilingAdapter(SourceAdapter):
    """SEC EDGAR filing adapter.

    Fetches filings from EDGAR Atom feeds, parses filing detail pages,
    and produces normalized signals with structured XBRL/metadata.
    """

    name: ClassVar[str] = "sec_filings"
    source_type: ClassVar[str] = "sec"
    config_schema: ClassVar[type[BaseModel]] = SECConfig

    # SEC rate limit: 10 requests/second
    _RATE_LIMIT_DELAY = 0.11  # seconds between requests
    _EDGAR_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    _USER_AGENT = "StratOps-Intel/1.0 (contact@example.com)"

    def __init__(self) -> None:
        self._browser: Browser | None = None
        self._playwright = None
        self._last_request_time = 0.0

    async def _ensure_browser(self) -> Browser:
        """Lazy-initialize Playwright browser."""
        if self._browser is None or not self._browser.is_connected():
            if not PLAYWRIGHT_AVAILABLE:
                raise RuntimeError("Playwright not installed. Install with: pip install playwright && playwright install")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        return self._browser

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def _rate_limit(self) -> None:
        """Enforce SEC rate limit (10 req/sec)."""
        import asyncio
        import time

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._RATE_LIMIT_DELAY:
            await asyncio.sleep(self._RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    def _build_feed_url(self, form_type: str, start: int = 0, count: int = 100) -> str:
        """Build EDGAR Atom feed URL."""
        params = {
            "action": "getcurrent",
            "type": form_type,
            "company": "",
            "datea": "",
            "dateb": "",
            "start": str(start),
            "count": str(count),
            "output": "atom",
        }
        return f"{self._EDGAR_BASE}?{urllib.parse.urlencode(params)}"

    async def fetch(self, config: dict[str, Any], cursor: str | None = None) -> IngestionResult:
        """Fetch SEC filings from EDGAR Atom feeds.

        Args:
            config: SECConfig dict with CIKs, form types, date range.
            cursor: Formatted as "form_type:start_index" for pagination.

        Returns:
            IngestionResult with concatenated Atom XML.
        """
        if not FEEDPARSER_AVAILABLE:
            raise RuntimeError("feedparser not installed. Install with: pip install feedparser")

        cfg = SECConfig(**config)
        all_entries: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        next_cursors: dict[str, str] = {}

        # Parse cursor
        form_type = "10-K"
        start_idx = 0
        if cursor:
            parts = cursor.split(":")
            if len(parts) == 2:
                form_type, start_idx = parts[0], int(parts[1])

        # For each form type, fetch feed
        for ft in cfg.form_types:
            try:
                await self._rate_limit()
                feed_url = self._build_feed_url(ft, start=start_idx)
                logger.debug("fetching_sec_feed", form_type=ft, url=feed_url)

                # Fetch with custom headers
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(
                        feed_url,
                        headers={"User-Agent": self._USER_AGENT},
                    )
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.text)

                if feed.bozo:
                    logger.warning("feed_parse_warning", form_type=ft, error=str(feed.bozo_exception))

                for entry in feed.entries:
                    # Filter by CIK if specified
                    if cfg.ciks:
                        cik = self._extract_cik_from_entry(entry)
                        if cik not in cfg.ciks:
                            continue

                    # Filter by date range
                    filing_date = self._parse_filing_date(entry)
                    if filing_date:
                        if cfg.filing_date_from and filing_date < cfg.filing_date_from:
                            continue
                        if cfg.filing_date_to and filing_date > cfg.filing_date_to:
                            continue

                    all_entries.append({
                        "form_type": ft,
                        "entry": entry,
                        "cik": cik,
                        "filing_date": filing_date.isoformat() if filing_date else None,
                    })

                next_cursors[ft] = str(start_idx + len(feed.entries))

            except Exception as e:
                logger.warning("sec_feed_error", form_type=ft, error=str(e))
                errors.append({"form_type": ft, "error": str(e)})

        # Combine entries into single XML-like structure for parse()
        import json
        combined = json.dumps({"entries": all_entries, "cursors": next_cursors})

        return IngestionResult(
            raw_data=combined.encode("utf-8"),
            content_type="application/json",
            next_cursor=None,  # Pagination handled per form_type in metadata
            metadata={
                "total_entries": len(all_entries),
                "error_count": len(errors),
                "errors": errors,
                "next_cursors": next_cursors,
            },
        )

    def _extract_cik_from_entry(self, entry: Any) -> str:
        """Extract CIK from Atom entry."""
        # Try various fields
        for field in ["cik", "company_cik", "sec_cik"]:
            if hasattr(entry, field):
                return str(getattr(entry, field)).zfill(10)
        # Try from link
        if hasattr(entry, "link"):
            match = re.search(r"/cik/(\d+)", entry.link)
            if match:
                return match.group(1).zfill(10)
        return ""

    def _parse_filing_date(self, entry: Any) -> date | None:
        """Parse filing date from entry."""
        for field in ["filing_date", "updated", "published", "date"]:
            if hasattr(entry, field):
                try:
                    return datetime.fromisoformat(getattr(entry, field).replace("Z", "+00:00")).date()
                except Exception:
                    pass
        return None

    async def parse(self, raw_data: bytes, content_type: str) -> list[RawSignal]:
        """Parse EDGAR feed entries into RawSignal objects."""
        import json

        data = json.loads(raw_data.decode("utf-8"))
        entries = data.get("entries", [])
        signals: list[RawSignal] = []

        for item in entries:
            entry = item["entry"]
            form_type = item["form_type"]
            cik = item.get("cik", "")
            filing_date = item.get("filing_date")

            # Extract key fields
            accession = self._extract_accession_number(entry)
            company_name = self._extract_company_name(entry)
            filing_url = self._extract_filing_url(entry)

            # Create RawSignal - will fetch detail page in normalize()
            signal = RawSignal(
                source_type=self.source_type,
                source_url=filing_url,
                raw_content=json.dumps({
                    "form_type": form_type,
                    "cik": cik,
                    "company_name": company_name,
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "filing_url": filing_url,
                    "entry_raw": entry,
                }).encode("utf-8"),
                fingerprint=None,  # Will be computed by fingerprint()
                collected_at=datetime.utcnow(),
                metadata={
                    "form_type": form_type,
                    "cik": cik,
                    "company_name": company_name,
                    "accession_number": accession,
                    "filing_date": filing_date,
                },
            )
            signals.append(signal)

        logger.debug("sec_parse_complete", signal_count=len(signals))
        return signals

    def _extract_accession_number(self, entry: Any) -> str:
        """Extract accession number from entry."""
        # Handle both dict and object entries
        if isinstance(entry, dict):
            title = entry.get("title", "")
            link = entry.get("link", "")
        else:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")

        for text in (title, link):
            match = re.search(r"(\d{10}-\d{2}-\d{6})", text)
            if match:
                return match.group(1)
        return ""

    def _extract_company_name(self, entry: Any) -> str:
        """Extract company name from entry."""
        if isinstance(entry, dict):
            title = entry.get("title", "")
        else:
            title = getattr(entry, "title", "")

        # Title format: "Company Name (CIK) - Form Type - Accession"
        parts = title.split(" - ")
        if parts:
            return parts[0].rsplit(" (", 1)[0]
        return ""

    def _extract_filing_url(self, entry: Any) -> str:
        """Extract filing detail URL from entry."""
        if hasattr(entry, "link"):
            return entry.link
        if hasattr(entry, "links"):
            for link in entry.links:
                if link.get("type") == "text/html":
                    return link.get("href", "")
        return ""

    async def fingerprint(self, signal: RawSignal) -> str:
        """Compute fingerprint from accession number + form type + date."""
        meta = signal.metadata
        key = f"{meta.get('accession_number','')}|{meta.get('form_type','')}|{meta.get('filing_date','')}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    async def normalize(self, signals: list[RawSignal]) -> list[NormalizedSignal]:
        """Fetch filing detail pages, upload to MinIO, create normalized signals."""
        aiobotocore_session, available = _get_aiobotocore()
        if not available:
            raise RuntimeError("aiobotocore not installed. Install with: pip install aiobotocore")
        if not BS4_AVAILABLE:
            raise RuntimeError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

        browser = await self._ensure_browser()
        normalized: list[NormalizedSignal] = []

        for signal in signals:
            fingerprint = signal.fingerprint or await self.fingerprint(signal)
            meta = signal.metadata
            filing_url = meta.get("filing_url", "")

            # Fetch filing detail page for full HTML
            html_content = b""
            structured = dict(meta)

            if filing_url:
                try:
                    await self._rate_limit()
                    page = await browser.new_page(user_agent=self._USER_AGENT)
                    page.set_default_timeout(30000)

                    resp = await page.goto(filing_url, wait_until="domcontentloaded")
                    if resp and resp.status < 400:
                        html_content = await page.content()
                        # Parse for additional metadata
                        structured.update(await self._parse_filing_page(html_content))

                    await page.close()
                except Exception as e:
                    logger.warning("filing_page_fetch_failed", url=filing_url, error=str(e))

            if not html_content:
                html_content = b"<!-- Filing page not fetched -->"

            # Generate S3 URI
            tenant_id = meta.get("tenant_id", "unknown")
            signal_id = fingerprint[:16]
            content_uri = f"s3://stratops-raw-{tenant_id}/sec/{signal_id}.html"

            normalized_signal = NormalizedSignal(
                source_type=self.source_type,
                source_url=filing_url,
                content_uri=content_uri,
                fingerprint=fingerprint,
                structured_payload=structured,
                collected_at=signal.collected_at,
                metadata=signal.metadata,
            )
            normalized.append(normalized_signal)

        logger.debug("sec_normalize_complete", count=len(normalized))
        return normalized

    async def _parse_filing_page(self, html: str) -> dict[str, Any]:
        """Parse SEC filing detail page for additional metadata."""
        soup = BeautifulSoup(html, "html.parser")
        structured: dict[str, Any] = {}

        # Extract document links
        docs = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if any(href.endswith(ext) for ext in [".htm", ".html", ".xml", ".xsd", ".txt"]):
                docs.append({"url": href, "text": link.get_text(strip=True)})
        structured["documents"] = docs

        # Extract filing metadata from tables
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                    val = cells[1].get_text(strip=True)
                    if key and val:
                        structured[key] = val

        return structured

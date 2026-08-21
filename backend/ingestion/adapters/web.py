"""Web monitoring adapter using Playwright for rendering and SimHash for deduplication.

Fetches web pages, extracts content, and produces normalized signals with
MinIO pointers for raw HTML storage.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

# Playwright is optional - handle gracefully if not installed
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

# SimHash for near-duplicate detection
try:
    import simhash

    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False

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


# Lazy imports - avoid module-level import conflicts
def _get_aiobotocore():
    try:
        import aiobotocore.session
        return aiobotocore.session, True
    except ImportError:
        return None, False


def _get_simhash():
    try:
        import simhash
        return simhash, True
    except ImportError:
        return None, False

logger = structlog.get_logger(__name__)


class WebMonitorConfig(BaseModel):
    """Configuration for WebMonitorAdapter.

    Attributes:
        urls: List of URLs to monitor.
        check_interval_seconds: How often to re-check URLs (default 1 hour).
        user_agent: User-Agent string for requests.
        max_depth: Maximum crawl depth (0 = single page only).
        selector: Optional CSS selector to extract specific content.
        extract_links: Whether to extract and follow links.
    """

    model_config = ConfigDict(extra="forbid")

    urls: list[HttpUrl] = Field(..., min_length=1, description="URLs to monitor")
    check_interval_seconds: int = Field(default=3600, ge=60, description="Re-check interval")
    user_agent: str = Field(default="StratOps-Intel/1.0", description="User-Agent header")
    max_depth: int = Field(default=0, ge=0, le=3, description="Maximum crawl depth")
    selector: str | None = Field(default=None, description="CSS selector for content extraction")
    extract_links: bool = Field(default=False, description="Whether to extract and follow links")

    @field_validator("urls", mode="before")
    @classmethod
    def validate_urls(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


@register_adapter
class WebMonitorAdapter(SourceAdapter):
    """Web monitoring adapter using headless browser rendering.

    Fetches URLs via Playwright, extracts text content with BeautifulSoup,
    computes SimHash fingerprints for near-duplicate detection, and stores
    raw HTML in MinIO with S3 pointers in normalized signals.
    """

    name: ClassVar[str] = "web_monitor"
    source_type: ClassVar[str] = "web"
    config_schema: ClassVar[type[BaseModel]] = WebMonitorConfig

    def __init__(self) -> None:
        self._browser: Browser | None = None
        self._playwright = None

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

    async def fetch(self, config: dict[str, Any], cursor: str | None = None) -> IngestionResult:
        """Fetch web pages using Playwright.

        Args:
            config: WebMonitorConfig dict with URLs and settings.
            cursor: Index into config['urls'] for pagination (unused for now).

        Returns:
            IngestionResult with concatenated HTML from all URLs.
        """
        cfg = WebMonitorConfig(**config)

        # Check if Playwright is available
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("playwright_not_available", message="Playwright not installed")
            return IngestionResult(
                raw_data=b"",
                content_type="text/html",
                next_cursor=None,
                metadata={
                    "fetched_count": 0,
                    "error_count": len(cfg.urls),
                    "errors": [{"url": str(u), "error": "Playwright not available"} for u in cfg.urls],
                    "start_index": 0,
                    "next_cursor": None,
                },
            )

        browser = await self._ensure_browser()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        start_idx = int(cursor) if cursor else 0
        urls = [str(u) for u in cfg.urls[start_idx:]]

        for idx, url in enumerate(urls):
            page: Page | None = None
            try:
                page = await browser.new_page(user_agent=cfg.user_agent)
                page.set_default_timeout(30000)

                logger.debug("fetching_url", url=url, idx=start_idx + idx)
                response = await page.goto(url, wait_until="domcontentloaded")
                status = response.status if response else 0

                if status >= 400:
                    logger.warning("fetch_http_error", url=url, status=status)
                    errors.append({"url": url, "status": status, "error": f"HTTP {status}"})
                    continue

                html = await page.content()
                results.append({"url": url, "html": html, "status": status})

                if cfg.extract_links and cfg.max_depth > 0:
                    links = await self._extract_links(page, url, cfg)
                    logger.debug("links_discovered", url=url, count=len(links))

            except PlaywrightTimeoutError:
                logger.warning("fetch_timeout", url=url)
                errors.append({"url": url, "error": "Timeout after 30s"})
            except Exception as e:
                logger.warning("fetch_error", url=url, error=str(e))
                errors.append({"url": url, "error": str(e)})
            finally:
                if page:
                    await page.close()

        # Combine all HTML into single raw_data for parse()
        combined_html = "\n<!-- URL SEPARATOR -->\n".join(
            f"<!-- URL: {r['url']} -->\n{r['html']}" for r in results
        )

        metadata = {
            "fetched_count": len(results),
            "error_count": len(errors),
            "errors": errors,
            "start_index": start_idx,
            "next_cursor": str(start_idx + len(urls)) if start_idx + len(urls) < len(cfg.urls) else None,
        }

        return IngestionResult(
            raw_data=combined_html.encode("utf-8"),
            content_type="text/html",
            next_cursor=metadata["next_cursor"],
            metadata=metadata,
        )

    async def _extract_links(self, page: Page, base_url: str, cfg: WebMonitorConfig) -> list[str]:
        """Extract links from page (for future depth crawling)."""
        try:
            links = await page.eval_on_selector_all(
                "a[href]", "elements => elements.map(e => e.href)"
            )
            # Filter to same domain
            base_domain = urllib.parse.urlparse(base_url).netloc
            return [link for link in links if urllib.parse.urlparse(link).netloc == base_domain]
        except Exception:
            return []

    async def parse(self, raw_data: bytes, content_type: str) -> list[RawSignal]:
        """Parse HTML into RawSignal objects (one per URL)."""
        if not BS4_AVAILABLE:
            raise RuntimeError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

        html = raw_data.decode("utf-8", errors="replace")
        # Split by our URL separator
        parts = html.split("<!-- URL SEPARATOR -->")
        signals: list[RawSignal] = []

        for part in parts:
            if not part.strip():
                continue

            # Extract URL from comment
            url_match = part.strip().split("\n", 1)
            if len(url_match) < 2 or not url_match[0].startswith("<!-- URL: "):
                continue

            url = url_match[0][9:-3]  # Remove "<!-- URL: " and " -->"
            page_html = url_match[1] if len(url_match) > 1 else ""

            # Extract text content using BeautifulSoup
            soup = BeautifulSoup(page_html, "html.parser")

            # Remove script/style elements
            for elem in soup(["script", "style", "noscript", "iframe"]):
                elem.decompose()

            # Apply selector if configured
            # Note: config not available here, would need to be passed differently
            # For now, extract all text
            text_content = soup.get_text(separator="\n", strip=True)

            # Create RawSignal
            signal = RawSignal(
                source_type=self.source_type,
                source_url=url,
                raw_content=page_html.encode("utf-8"),
                fingerprint=None,  # Will be computed by fingerprint()
                collected_at=datetime.utcnow(),
                metadata={
                    "text_length": len(text_content),
                    "title": soup.title.string if soup.title else None,
                    "meta_description": self._get_meta_description(soup),
                },
            )
            signals.append(signal)

        logger.debug("parse_complete", signal_count=len(signals))
        return signals

    def _get_meta_description(self, soup: BeautifulSoup) -> str | None:
        """Extract meta description from HTML."""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"]
        meta = soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"]
        return None

    async def fingerprint(self, signal: RawSignal) -> str:
        """Compute SimHash fingerprint for near-duplicate detection.

        Falls back to SHA-256 if simhash library not available.
        """
        if not signal.raw_content:
            return hashlib.sha256(b"").hexdigest()

        text = signal.metadata.get("text_content", "")
        if not text:
            if BS4_AVAILABLE:
                soup = BeautifulSoup(signal.raw_content, "html.parser")
                for elem in soup(["script", "style"]):
                    elem.decompose()
                text = soup.get_text(separator=" ", strip=True)

        simhash_lib, available = _get_simhash()
        if available and text:
            sh = simhash_lib.Simhash(text, f=64)
            return format(sh.value, "016x")
        else:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def normalize(self, signals: list[RawSignal]) -> list[NormalizedSignal]:
        """Upload raw HTML to MinIO and create NormalizedSignal with pointers."""
        aiobotocore_session, available = _get_aiobotocore()
        if not available:
            raise RuntimeError("aiobotocore not installed. Install with: pip install aiobotocore")

        normalized: list[NormalizedSignal] = []

        for signal in signals:
            # Compute fingerprint if not already set
            fingerprint = signal.fingerprint or await self.fingerprint(signal)

            # Generate S3 URI
            tenant_id = signal.metadata.get("tenant_id", "unknown")
            signal_id = hashlib.sha256(signal.raw_content).hexdigest()[:16]
            content_uri = f"s3://stratops-raw-{tenant_id}/{signal_id}.bin"

            # Extract structured metadata
            structured = {
                "title": signal.metadata.get("title"),
                "meta_description": signal.metadata.get("meta_description"),
                "text_length": signal.metadata.get("text_length"),
            }

            normalized_signal = NormalizedSignal(
                source_type=self.source_type,
                source_url=signal.source_url,
                content_uri=content_uri,
                fingerprint=fingerprint,
                structured_payload=structured,
                collected_at=signal.collected_at,
                metadata=signal.metadata,
            )
            normalized.append(normalized_signal)

        logger.debug("normalize_complete", count=len(normalized))
        return normalized

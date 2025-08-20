"""Async RSS/JSON feed ingester with sentiment + reach clustering and MongoDB upsert.
---------------------------------------------------------------------------
Key points
- Uses an hourly cache (see `CACHE_TTL_SECONDS`).
- Fetches feeds concurrently with retries/back‑off.
- Accepts **both** classic RSS/Atom XML _and_ the JSON structure you posted
  (keys like `publishedAt`, `imageUrl`).
- After deduplication, each article is enriched with:
    • `id`            – md5 hash of the URL.
    • `cron_timestamp`– time the ingest cron ran.
    • `sentiment`     – POSITIVE / NEGATIVE / NEUTRAL (Hindi + English).
    • `cluster_id`    – label from strict title‑similarity clustering.
- Results are written to MongoDB with upsert (unique on `url`).
"""
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
import asyncio
import json
import logging
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

# third‑party
import aiohttp
import feedparser
from dateutil import parser as dt  # publishedAt parsing for JSON feeds
from sklearn.cluster import AgglomerativeClustering  # noqa: F401 – imported elsewhere

# local
from .cache import default_cache
from .config import CACHE_TTL_SECONDS, RSS_FEEDS
from .db import upsert_articles
from .models import Article
from .reach import cluster_with_labels
from .sentiment import analyze_article_sentiments, get_or_load_sentiment_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# JSON helper – converts the sample structure directly to Article
# ---------------------------------------------------------------------------

def json_entry_to_article(item: dict[str, Any]) -> Article:
    """Convert a JSON item (with publishedAt/imageUrl) into an `Article`."""
    url = item["url"]
    return Article(
        id           = md5(url.encode()).hexdigest(),
        url          = url,
        title        = item["title"],
        source       = item["source"],
        published_at = dt.parse(item["publishedAt"], fuzzy=True),
        description  = item.get("description", ""),
        image_url    = item.get("imageUrl"),
        author       = item.get("author", "Unknown"),
        categories   = item.get("categories", []),
    )

# ---------------------------------------------------------------------------
# Feed fetching helpers
# ---------------------------------------------------------------------------


class FeedFetchError(Exception):
    """Raised when all attempts to fetch a feed fail."""


class FeedParser:
    """Async RSS / Atom **XML** parser with robust retries+timeouts."""

    def __init__(self, timeout: int = 30, max_retries: int = 3, retry_delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

    # ---------------------------------------------------------------------
    # Async context management
    # ---------------------------------------------------------------------
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, ttl_dns_cache=300, use_dns_cache=True)
        timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10, sock_read=20)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "NewsAnalytics/1.0 (+github.com/example)",
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    # ---------------------------------------------------------------------
    # Networking helpers
    # ---------------------------------------------------------------------
    async def fetch_feed_content(self, url: str) -> Optional[str]:
        if not self.session:
            raise RuntimeError("Session not initialised; use as async context manager.")

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logger.debug("Fetching %s (attempt %s/%s)", url, attempt + 1, self.max_retries)
                async with self.session.get(url) as resp:
                    status = resp.status
                    if status == 200:
                        text = await resp.text()
                        return text.strip() or None
                    if status in {404, 403}:
                        logger.warning("Permanent %s for %s", status, url)
                        return None
                    if status in {429, 502, 503, 504}:
                        logger.warning("Temporary %s for %s – retrying", status, url)
                        await asyncio.sleep(self.retry_delay * 2**attempt)
                        continue
                    logger.error("HTTP %s for %s", status, url)
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logger.warning("%s on %s – retrying", type(exc).__name__, url)
                await asyncio.sleep(self.retry_delay * 2**attempt)
                continue
        logger.error("Failed to fetch %s after %s attempts – %s", url, self.max_retries, last_exc)
        return None

    # ---------------------------------------------------------------------
    # Parsing helpers
    # ---------------------------------------------------------------------
    async def parse_feed_content(self, content: str, source_url: str) -> List[Article]:
        """Parse XML/Atom string into Article objects (thread‑off to avoid block)."""
        feed = await asyncio.to_thread(feedparser.parse, content)
        if feed.get("bozo"):
            logger.warning("Feed parsing warning for %s: %s", source_url, feed.get("bozo_exception"))

        source_title = feed.get("feed", {}).get("title", self._extract_domain(source_url))
        entries: List[dict[str, Any]] = feed.get("entries", [])
        articles: List[Article] = []
        for entry in entries:
            try:
                art = self._parse_entry(entry, source_title, source_url)
                if art:
                    articles.append(art)
            except Exception as exc:  # noqa: BLE001 – want log but continue
                logger.error("Error parsing entry from %s: %s", source_url, exc)
        return articles

    # ---------------------------------------------------------------------
    # Internal entry‑to‑model converter
    # ---------------------------------------------------------------------
    def _parse_entry(self, entry: Dict[str, Any], source_title: str, source_url: str) -> Optional[Article]:
        """Handle **classic** RSS/Atom entries. JSON items are intercepted earlier."""
        # If this entry already looks like the JSON structure → short‑circuit
        if "publishedAt" in entry:
            return json_entry_to_article(entry)

        title = self._clean_text(entry.get("title", "Untitled"))
        link = entry.get("link", "").strip()
        if not link:
            return None
        # Relative URL handling
        if link.startswith("/"):
            base = f"{urlparse(source_url).scheme}://{urlparse(source_url).netloc}"
            link = urljoin(base, link)
        elif not link.startswith(("http://", "https://")):
            link = f"https://{link}"

        published_at = self._parse_date(entry)
        description  = self._extract_description(entry)
        image_url    = self._extract_image_url(entry)
        author       = self._clean_text(entry.get("author", "Unknown"))
        categories   = self._extract_categories(entry)

        return Article(
            title        = title,
            url          = link,
            source       = source_title,
            published_at = published_at,
            description  = description,
            image_url    = image_url,
            author       = author,
            categories   = categories,
        )

    # ------------------------------------------------------------------
    # utility helpers (unchanged)
    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:  # noqa: D401 – simple helper
        import re
        if not text:
            return ""
        if hasattr(text, "value"):
            text = text.value
        elif isinstance(text, dict):
            text = text.get("value", str(text))
        text = re.sub(r"<[^>]+>", "", str(text)).strip()
        return re.sub(r"\s+", " ", text)

    def _parse_date(self, entry: Dict[str, Any]) -> datetime:
        for fld in ("published_parsed", "updated_parsed", "created_parsed"):
            ts = entry.get(fld)
            if ts and len(ts) >= 6:
                return datetime(*ts[:6])
        import email.utils
        for fld in ("published", "updated", "created"):
            if entry.get(fld):
                p = email.utils.parsedate_tz(entry[fld])
                if p:
                    return datetime.fromtimestamp(email.utils.mktime_tz(p))
        return datetime.utcnow()

    def _extract_description(self, entry: Dict[str, Any]) -> str:
        for fld in ("summary", "description", "content"):
            if entry.get(fld):
                content = entry[fld][0] if isinstance(entry[fld], list) else entry[fld]
                if hasattr(content, "value"):
                    content = content.value
                return self._clean_text(str(content))
        return ""

    def _extract_image_url(self, entry: Dict[str, Any]) -> Optional[str]:
        for fld in ("media_thumbnail", "media_content", "image", "enclosures"):
            media = entry.get(fld)
            if not media:
                continue
            media = media[0] if isinstance(media, list) else media
            if isinstance(media, dict):
                url = media.get("url") or media.get("href")
                if url and url.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png", "gif", "webp"}:
                    return url
        return None

    def _extract_categories(self, entry: Dict[str, Any]) -> List[str]:
        cats: set[str] = set()
        if entry.get("tags"):
            for tag in entry["tags"]:
                term = tag.get("term") if isinstance(tag, dict) else str(tag)
                if term:
                    cats.add(self._clean_text(term))
        if entry.get("category"):
            cats.add(self._clean_text(entry["category"]))
        return list(cats)

    def _extract_domain(self, url: str) -> str:
        try:
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith("www.") else domain
        except Exception:
            return "Unknown Source"

# ---------------------------------------------------------------------------
# Public API – orchestrator
# ---------------------------------------------------------------------------


async def get_articles(*, force_refresh: bool = False) -> List[Article]:
    """Fetch, enrich, cache & store articles. Main entry point used by FastAPI."""
    cache_key = "articles"

    if not force_refresh:
        cached = await default_cache.get(cache_key)
        if cached:
            articles = [Article(**item) for item in cached]
            logger.info("Returning %s cached articles", len(articles))
            return articles

    logger.info("Fetching fresh articles…")
    async with FeedParser() as parser:
        tasks = [asyncio.create_task(fetch_and_parse_feed(parser, url)) for url in RSS_FEEDS]
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=120)
        except asyncio.TimeoutError:
            logger.error("Global timeout waiting for feeds – continuing with partial data")
            results = []

    all_articles: List[Article] = []
    for src, res in zip(RSS_FEEDS, results):
        if isinstance(res, Exception):
            logger.error("Error processing %s: %s", src, res)
        else:
            all_articles.extend(res)
    logger.info("Got %s raw articles", len(all_articles))

    # ------------------------------------------------------------------
    # Enrichment pipeline
    # ------------------------------------------------------------------
    articles = deduplicate_articles(all_articles)
    cron_ts = datetime.utcnow()
    for a in articles:
        a.id = md5(str(a.url).encode()).hexdigest()
        a.cron_timestamp = cron_ts

    analyzer = await get_or_load_sentiment_model()
    articles = await analyze_article_sentiments(articles, analyzer)

    _, labels = await cluster_with_labels(articles, distance_threshold=0.6)
    for a, lbl in zip(articles, labels):
        a.cluster_id = str(lbl)

    # ------------------------------------------------------------------
    # Cache + Mongo
    # ------------------------------------------------------------------
    if articles:
        await upsert_articles(articles)
        payload = [a.model_dump(mode="json") for a in articles] 
        await default_cache.set(cache_key, payload, ttl=CACHE_TTL_SECONDS)
        logger.info("Cached & stored %s articles", len(articles))

    return articles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def fetch_and_parse_feed(parser: FeedParser, url: str) -> List[Article]:
    content = await parser.fetch_feed_content(url)
    if not content:
        return []
    stripped = content.lstrip()
    if stripped.startswith(("{", "[")):          # quick heuristic
            try:
                data = json.loads(stripped)
                items = data if isinstance(data, list) else data.get("items", [])
                return [json_entry_to_article(i) for i in items]
            except Exception as e:
                logger.warning("Failed JSON parse for %s: %s — falling back to XML", url, e)

    # 2️⃣  Otherwise treat it as RSS / Atom XML
    return await parser.parse_feed_content(content, url)


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    seen: set[str] = set()
    uniq: List[Article] = []
    for art in articles:
        u = str(art.url)
        if u not in seen:
            seen.add(u)
            uniq.append(art)
    uniq.sort(key=lambda a: a.published_at, reverse=True)
    logger.info("Deduplicated %s → %s", len(articles), len(uniq))
    return uniq


# ---------------------------------------------------------------------------
# Background task (to be awaited from FastAPI lifespan or external scheduler)
# ---------------------------------------------------------------------------


async def periodic_feed_update(interval: int = CACHE_TTL_SECONDS) -> None:  # pragma: no cover
    while True:
        try:
            await get_articles(force_refresh=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("Periodic update failed: %s", exc)
        await asyncio.sleep(interval)

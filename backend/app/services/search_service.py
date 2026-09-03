"""Multi-source paper search service."""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper, parse_retry_after
from app.paper_sources.semantic_scholar import SemanticScholarSource
from app.paper_sources.arxiv import ArxivSource
from app.paper_sources.openalex import OpenAlexSource
from app.paper_sources.crossref import CrossrefSource
from app.paper_sources.ieee import IeeeSource
from app.paper_sources.core import CoreSource
from app.paper_sources.unpaywall import UnpaywallSource
from app.services.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

# Bounded LRU cache: (source_name, query_hash) -> (papers, timestamp)
# Uses OrderedDict to implement LRU eviction with max size.
_CACHE_MAX_SIZE = 1000
_CACHE_TTL = 3600  # 1 hour
_cache: OrderedDict[str, tuple[list[RawPaper], float]] = OrderedDict()


def _cache_key(source_name: str, query: str, limit: int) -> str:
    """Generate cache key for a search query."""
    qhash = hashlib.md5(f"{source_name}:{query}:{limit}".encode()).hexdigest()
    return f"{source_name}:{qhash}"


def _cache_get(key: str) -> tuple[list[RawPaper], float] | None:
    """Get from cache, evicting expired entries. Moves accessed entry to end (most recent)."""
    entry = _cache.get(key)
    if entry is None:
        return None
    papers, ts = entry
    if (time.time() - ts) >= _CACHE_TTL:
        # Expired — remove and return miss
        _cache.pop(key, None)
        return None
    # Move to end (mark as recently used)
    _cache.move_to_end(key)
    return entry


def _cache_put(key: str, papers: list[RawPaper]):
    """Insert into cache, evicting oldest if over capacity."""
    _cache[key] = (papers, time.time())
    _cache.move_to_end(key)
    # Evict oldest entries while over capacity
    while len(_cache) > _CACHE_MAX_SIZE:
        _cache.popitem(last=False)  # FIFO eviction (oldest first)


class SearchService:
    """Coordinates multi-source paper search with concurrency."""

    def __init__(self):
        # All candidate sources; only load those actually usable in the current
        # config so we don't fire wasted requests at key-gated sources (IEEE /
        # CORE) and can report an honest active-source count.
        all_sources: list[PaperSource] = [
            SemanticScholarSource(),
            ArxivSource(),
            OpenAlexSource(),
            CrossrefSource(),
            IeeeSource(),
            CoreSource(),
        ]
        self.sources: list[PaperSource] = [s for s in all_sources if s.is_available()]
        skipped = [s.name for s in all_sources if not s.is_available()]
        logger.info(
            "SearchService: %d/%d sources active (%s)%s",
            len(self.sources), len(all_sources),
            ", ".join(s.name for s in self.sources),
            f"; skipped (no key): {', '.join(skipped)}" if skipped else "",
        )
        self._unpaywall = UnpaywallSource()
        # 429 cooldown: source_name -> cooldown_until timestamp
        # When a source returns 429 (rate limited), skip it for cooldown_s
        self._cooldowns: dict[str, float] = {}
        self._cooldown_s = settings.rate_limit_cooldown_s

    def source_health(self) -> dict:
        """Report active/skipped sources for diagnostics and UI transparency."""
        active = [s.name for s in self.sources]
        return {
            "active_sources": active,
            "active_count": len(active),
            "cooling_down": [name for name in self._cooldowns
                             if self._is_cooled_down(name)],
        }

    def _is_cooled_down(self, source_name: str) -> bool:
        """Check if source is in 429 cooldown period."""
        until = self._cooldowns.get(source_name)
        if until and time.time() < until:
            return True
        if until:
            # Cooldown expired, clear it
            del self._cooldowns[source_name]
        return False

    def _mark_cooldown(self, source_name: str, retry_after: float | None = None):
        """Mark a source as rate-limited; skip for cooldown_s.

        A server-supplied Retry-After wins when it asks for longer than the
        default: it states when the source is actually usable again.
        """
        cooldown_s = self._cooldown_s
        if retry_after is not None and retry_after > cooldown_s:
            cooldown_s = retry_after
        self._cooldowns[source_name] = time.time() + cooldown_s
        logger.info("Source '%s' entered %.0fs cooldown (429)", source_name, cooldown_s)

    async def _search_with_cache(self, src: PaperSource, query: str, limit: int) -> list[RawPaper]:
        """Search one source with cache + rate limit + 429 cooldown."""
        # Skip if in cooldown
        if self._is_cooled_down(src.name):
            logger.debug("Skipping %s (in cooldown)", src.name)
            return []

        key = _cache_key(src.name, query, limit)
        cached = _cache_get(key)
        if cached:
            logger.debug("Cache hit for %s query '%s'", src.name, query[:30])
            return cached[0]

        # Acquire rate-limit token before hitting the API
        await rate_limiter.acquire(src.name)
        # Re-check the cooldown after queueing: acquire() is where concurrent
        # queries wait, so a sibling query may have hit 429 and cooled the source
        # down in the meantime. Without this the whole concurrent batch still
        # fires requests that are certain to fail, which is exactly what made a
        # rate-limited source cost every query its full retry budget.
        if self._is_cooled_down(src.name):
            logger.debug("Skipping %s (cooled down while awaiting rate limit)", src.name)
            return []
        try:
            result = await src.search(query, limit)
            _cache_put(key, result)
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._mark_cooldown(src.name, parse_retry_after(e.response))
            else:
                logger.warning("Source %s HTTP %d", src.name, e.response.status_code)
            return []
        except Exception as e:
            logger.warning("Source %s failed: %s", src.name, e)
            return []

    async def search_all_sources(self, query: str, limit: int = 15,
                                 allowed_sources: list[str] | None = None) -> list[RawPaper]:
        """Search all sources concurrently for a single query.

        allowed_sources (optional): restrict to the named sources (budgeted
        audit mode searches only S2/OpenAlex/arXiv by default); None = all.
        """
        sources = self.sources
        if allowed_sources:
            wanted = {s.strip().lower() for s in allowed_sources}
            sources = [src for src in self.sources if src.name.lower() in wanted]
        tasks = [self._search_with_cache(src, query, limit) for src in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[RawPaper] = []
        for src, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning("Source %s failed: %s", src.name, result)
                continue
            all_papers.extend(result)

        # Supplement empty pdf_url fields via Unpaywall (free, no key)
        try:
            await self._unpaywall.enrich(all_papers)
        except Exception as e:
            logger.warning("Unpaywall enrichment failed: %s", e)

        return all_papers

    async def search_multiple_queries(self, queries: list[str], limit: int = 15,
                                      allowed_sources: list[str] | None = None) -> list[RawPaper]:
        """Search all sources for multiple queries with bounded concurrency.

        Limits concurrent queries to search_query_concurrency to avoid
        overwhelming rate-limited sources (S2/OpenAlex).
        """
        sem = asyncio.Semaphore(settings.search_query_concurrency)

        async def _one(q: str) -> list[RawPaper]:
            async with sem:
                return await self.search_all_sources(q, limit, allowed_sources)

        tasks = [_one(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[RawPaper] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Query batch failed: %s", result)
                continue
            all_papers.extend(result)

        return all_papers


search_service = SearchService()

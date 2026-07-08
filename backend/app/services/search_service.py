"""Multi-source paper search service."""

import asyncio
import hashlib
import logging
import time

from app.paper_sources.base import PaperSource, RawPaper
from app.paper_sources.semantic_scholar import SemanticScholarSource
from app.paper_sources.arxiv import ArxivSource
from app.paper_sources.openalex import OpenAlexSource
from app.paper_sources.crossref import CrossrefSource
from app.paper_sources.ieee import IeeeSource
from app.paper_sources.core import CoreSource

logger = logging.getLogger(__name__)

# Simple in-memory cache: (source_name, query_hash) -> (papers, timestamp)
_cache: dict[str, tuple[list[RawPaper], float]] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_key(source_name: str, query: str, limit: int) -> str:
    """Generate cache key for a search query."""
    qhash = hashlib.md5(f"{source_name}:{query}:{limit}".encode()).hexdigest()
    return f"{source_name}:{qhash}"


class SearchService:
    """Coordinates multi-source paper search with concurrency."""

    def __init__(self):
        self.sources: list[PaperSource] = [
            SemanticScholarSource(),
            ArxivSource(),
            OpenAlexSource(),
            CrossrefSource(),
            IeeeSource(),
            CoreSource(),
        ]

    async def search_all_sources(self, query: str, limit: int = 15) -> list[RawPaper]:
        """Search all sources concurrently for a single query."""
        async def _search_with_cache(src: PaperSource) -> list[RawPaper]:
            key = _cache_key(src.name, query, limit)
            cached = _cache.get(key)
            if cached and (time.time() - cached[1]) < _CACHE_TTL:
                logger.debug("Cache hit for %s query '%s'", src.name, query[:30])
                return cached[0]
            result = await src.search(query, limit)
            _cache[key] = (result, time.time())
            return result

        tasks = [_search_with_cache(src) for src in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[RawPaper] = []
        for src, result in zip(self.sources, results):
            if isinstance(result, Exception):
                logger.warning("Source %s failed: %s", src.name, result)
                continue
            all_papers.extend(result)

        return all_papers

    async def search_multiple_queries(self, queries: list[str], limit: int = 15) -> list[RawPaper]:
        """Search all sources for multiple queries concurrently."""
        tasks = [self.search_all_sources(q, limit) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[RawPaper] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Query batch failed: %s", result)
                continue
            all_papers.extend(result)

        return all_papers


search_service = SearchService()

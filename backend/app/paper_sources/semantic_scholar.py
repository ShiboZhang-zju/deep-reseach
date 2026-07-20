"""Semantic Scholar paper source."""

import logging

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper, retry_with_backoff

logger = logging.getLogger(__name__)


class SemanticScholarSource(PaperSource):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"

    async def _do_search(self, query: str, limit: int, headers: dict) -> list[RawPaper]:
        fields = "title,abstract,authors,year,venue,externalIds,citationCount,url,openAccessPdf,references.paperId,references.title"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(f"{self.base_url}/paper/search", params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        try:
            data = await retry_with_backoff(
                self._do_search, query, limit, headers,
                max_retries=3, base_delay=2.0,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Semantic Scholar rate limited after retries")
            else:
                logger.error("Semantic Scholar error %d: %s", e.response.status_code, e.response.text[:200])
            return []
        except Exception as e:
            logger.error("Semantic Scholar request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        for item in data.get("data", []):
            ext_ids = item.get("externalIds") or {}
            papers.append(RawPaper(
                title=item.get("title", "") or "",
                abstract=item.get("abstract", "") or "",
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                year=item.get("year"),
                venue=item.get("venue", "") or "",
                doi=ext_ids.get("DOI", "") or "",
                arxiv_id=ext_ids.get("ArXiv", "") or "",
                semantic_scholar_id=item.get("paperId", "") or "",
                url=item.get("url", "") or "",
                pdf_url=(item.get("openAccessPdf") or {}).get("url", "") or "",
                citation_count=item.get("citationCount", 0) or 0,
                source=self.name,
                raw_data=item,
            ))
            # Parse references for citation graph (stored in raw_data)
            # references field: [{"paperId": "abc", "title": "..."}, ...]
            # Handled by search_papers.py via _extract_references()
        return papers

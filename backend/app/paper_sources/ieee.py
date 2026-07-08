"""IEEE Xplore paper source."""

import logging

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper

logger = logging.getLogger(__name__)


class IeeeSource(PaperSource):
    name = "ieee"
    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        if not settings.ieee_api_key:
            logger.warning("IEEE API key not configured, skipping")
            return []

        params = {
            "apikey": settings.ieee_api_key,
            "querytext": query,
            "max_records": min(limit, 200),
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(self.base_url, params=params)
                if resp.status_code == 429:
                    logger.warning("IEEE rate limited")
                    return []
                if resp.status_code != 200:
                    logger.error("IEEE error %d: %s", resp.status_code, resp.text[:200])
                    return []
                data = resp.json()
        except Exception as e:
            logger.error("IEEE request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        for item in data.get("articles", []):
            authors = []
            for au in item.get("authors", {}).get("authors", []):
                name = au.get("full_name", "")
                if name:
                    authors.append(name)

            doi = item.get("doi", "") or ""
            ieee_id = str(item.get("article_number", "")) or ""

            papers.append(RawPaper(
                title=item.get("title", "") or "",
                abstract=item.get("abstract", "") or "",
                authors=authors,
                year=item.get("publication_year"),
                venue=item.get("publication_title", "") or "",
                doi=doi,
                url=item.get("html_url", "") or "",
                pdf_url=item.get("pdf_url", "") or "",
                citation_count=item.get("citing_paper_count", 0) or 0,
                source=self.name,
                raw_data=item,
            ))
        return papers

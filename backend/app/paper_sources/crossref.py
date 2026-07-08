"""Crossref paper source."""

import logging

import httpx

from app.paper_sources.base import PaperSource, RawPaper, retry_with_backoff

logger = logging.getLogger(__name__)


class CrossrefSource(PaperSource):
    name = "crossref"
    base_url = "https://api.crossref.org/works"

    async def _do_search(self, query: str, limit: int, headers: dict) -> list[RawPaper]:
        params = {
            "query": query,
            "rows": min(limit, 100),
            "select": "title,abstract,author,published,DOI,URL,container-title,is-referenced-by-count",
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(self.base_url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        headers = {"User-Agent": "DeepResearch/1.0 (mailto:research@example.com)"}

        try:
            data = await retry_with_backoff(
                self._do_search, query, limit, headers,
                max_retries=3, base_delay=1.0,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Crossref rate limited after retries")
            else:
                logger.error("Crossref error %d", e.response.status_code)
            return []
        except Exception as e:
            logger.error("Crossref request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        for item in data.get("message", {}).get("items", []):
            titles = item.get("title", [])
            title = titles[0] if titles else ""

            abstract = item.get("abstract", "") or ""
            # Strip JATS XML tags
            if abstract:
                import re
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()

            authors = []
            for au in item.get("author", []):
                name_parts = []
                given = au.get("given", "")
                family = au.get("family", "")
                if given:
                    name_parts.append(given)
                if family:
                    name_parts.append(family)
                if name_parts:
                    authors.append(" ".join(name_parts))

            year = None
            date_parts = item.get("published", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            venues = item.get("container-title", [])
            venue = venues[0] if venues else ""

            doi = item.get("DOI", "") or ""
            url = item.get("URL", "") or ""

            papers.append(RawPaper(
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                citation_count=item.get("is-referenced-by-count", 0) or 0,
                source=self.name,
                raw_data={"doi": doi},
            ))
        return papers

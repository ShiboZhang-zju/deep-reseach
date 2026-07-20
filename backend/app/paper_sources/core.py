"""CORE (core.ac.uk) paper source."""

import logging

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper

logger = logging.getLogger(__name__)


class CoreSource(PaperSource):
    name = "core"
    base_url = "https://api.core.ac.uk/v3/search/works"

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        if not settings.core_api_key:
            logger.warning("CORE API key not configured, skipping")
            return []

        headers = {"Authorization": f"Bearer {settings.core_api_key}"}
        params = {
            "q": query,
            "limit": min(limit, 100),
        }

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(self.base_url, params=params, headers=headers)
                if resp.status_code == 429:
                    logger.warning("CORE rate limited")
                    raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
                if resp.status_code != 200:
                    logger.error("CORE error %d: %s", resp.status_code, resp.text[:200])
                    return []
                data = resp.json()
        except httpx.HTTPStatusError:
            raise  # let search_service handle cooldown
        except Exception as e:
            logger.error("CORE request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        for item in data.get("results", []):
            source = item.get("_source", item)

            doi = ""
            identifiers = source.get("identifiers", [])
            if isinstance(identifiers, list):
                for identifier in identifiers:
                    id_str = ""
                    if isinstance(identifier, dict):
                        id_str = identifier.get("identifier", "") or identifier.get("id", "")
                    elif isinstance(identifier, str):
                        id_str = identifier
                    if "doi" in id_str.lower():
                        doi = id_str.replace("doi:", "").replace("DOI:", "").strip()
                        break

            authors = []
            for au in source.get("authors", []):
                name = au.get("name", "") if isinstance(au, dict) else str(au)
                if name:
                    authors.append(name)

            year = None
            year_published = source.get("year_published", 0)
            if year_published:
                try:
                    year = int(year_published)
                except (ValueError, TypeError):
                    year = None

            download_url = source.get("download_url", "") or ""
            urls = source.get("source_fulltext_urls", [])
            if isinstance(urls, list) and urls and not download_url:
                download_url = urls[0] if isinstance(urls[0], str) else ""

            papers.append(RawPaper(
                title=source.get("title", "") or "",
                abstract=source.get("abstract", "") or "",
                authors=authors,
                year=year,
                venue=source.get("publisher", "") or "",
                doi=doi,
                url=download_url or source.get("id", ""),
                pdf_url=download_url,
                citation_count=0,
                source=self.name,
                raw_data=source,
            ))
        return papers

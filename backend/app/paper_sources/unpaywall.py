"""Unpaywall paper source — supplements open-access PDF URLs for known DOIs.

Unpaywall is free, requires no API key (only a contact email), and provides
~30M+ open-access PDF links. This source does NOT search by keyword; it only
enriches papers that already have a DOI with a legal open-access pdf_url.

Usage: call ``UnpaywallSource.enrich(papers)`` after keyword search to fill
``pdf_url`` on papers where it was empty.
"""

import asyncio
import logging

import httpx

from app.config import settings
from app.paper_sources.base import RawPaper

logger = logging.getLogger(__name__)


class UnpaywallSource:
    """Enrich existing RawPaper list with open-access PDF URLs via Unpaywall."""

    name = "unpaywall"
    base_url = "https://api.unpaywall.org/v2/{doi}"

    def __init__(self, email: str | None = None):
        self.email = email or settings.openalex_email or settings.crossref_email or "shoboz996@gmail.com"

    async def _lookup_doi(self, client: httpx.AsyncClient, doi: str) -> str:
        """Return best-effort open-access PDF URL for a DOI, or empty string."""
        try:
            resp = await client.get(
                self.base_url.format(doi=doi),
                params={"email": self.email},
                timeout=15,
            )
            if resp.status_code != 200:
                return ""
            data = resp.json()
            # Prefer best_oa_location.pdf_url; fall back to first OA location
            best = data.get("best_oa_location") or {}
            url = best.get("url_for_pdf", "") or best.get("url", "")
            if url:
                return url
            for loc in data.get("oa_locations", []) or []:
                u = loc.get("url_for_pdf", "") or loc.get("url", "")
                if u:
                    return u
        except Exception as e:
            logger.debug("Unpaywall lookup failed for %s: %s", doi, e)
        return ""

    async def enrich(self, papers: list[RawPaper], concurrency: int = 10) -> int:
        """Fill empty pdf_url fields via Unpaywall. Returns count enriched.

        Only papers with a DOI and empty pdf_url are queried.
        """
        candidates = [p for p in papers if p.doi and not p.pdf_url]
        if not candidates:
            return 0

        sem = asyncio.Semaphore(concurrency)
        enriched = 0

        async def _one(p: RawPaper):
            nonlocal enriched
            async with sem:
                async with httpx.AsyncClient(timeout=15) as client:
                    url = await self._lookup_doi(client, p.doi)
            if url:
                p.pdf_url = url
                enriched += 1

        await asyncio.gather(*[_one(p) for p in candidates], return_exceptions=True)
        logger.info("Unpaywall enriched %d/%d papers with PDF URLs", enriched, len(candidates))
        return enriched

"""OpenAlex paper source."""

import logging

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper, retry_with_backoff

logger = logging.getLogger(__name__)


class OpenAlexSource(PaperSource):
    name = "openalex"
    base_url = "https://api.openalex.org/works"

    async def _do_search(self, query: str, limit: int, headers: dict) -> list[RawPaper]:
        params = {
            "search": query,
            "per_page": min(limit, 200),
            "sort": "relevance_score:desc",
        }
        # OpenAlex accepts the polite-pool contact either as a mailto parameter
        # or in the User-Agent; send both, since only the parameter survives
        # proxies that rewrite headers. Note this does not lift an IP-level rate
        # limit: a controlled probe returned 429 for anonymous, User-Agent and
        # mailto-parameter requests alike, so cooldown remains the real defence.
        if settings.openalex_email:
            params["mailto"] = settings.openalex_email
        # API key gives $1/day budget (vs $0.10 without); free at openalex.org
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(self.base_url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        mailto = settings.openalex_email
        headers = {"User-Agent": f"DeepResearch/1.0 (mailto:{mailto})" if mailto
                   else "DeepResearch/1.0"}

        try:
            data = await retry_with_backoff(
                self._do_search, query, limit, headers,
                max_retries=3, base_delay=1.0,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAlex rate limited after retries")
                raise  # let search_service handle cooldown
            logger.error("OpenAlex error %d", e.response.status_code)
            return []
        except Exception as e:
            logger.error("OpenAlex request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        for item in data.get("results", []):
            ids = item.get("ids") or {}
            doi_raw = ids.get("doi", "") or ""
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

            authors = []
            for au in item.get("authorships", []):
                a = au.get("author") or {}
                name = a.get("display_name", "")
                if name:
                    authors.append(name)

            abstract_str = ""
            inv_idx = item.get("abstract_inverted_index")
            if inv_idx:
                word_positions = []
                for word, positions in inv_idx.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract_str = " ".join(w for _, w in word_positions)

            venue = ""
            venue_obj = item.get("primary_location", {}).get("source") if item.get("primary_location") else None
            if venue_obj:
                venue = venue_obj.get("display_name", "") or ""

            # Open-access info OpenAlex already returns for free: the is_oa flag
            # and the best OA PDF location. Capturing these lets scoring
            # deprioritise paywalled papers that cannot be fetched, and lets the
            # PDF download chain use the OA link directly instead of guessing.
            oa = item.get("open_access") or {}
            is_oa = oa.get("is_oa") if isinstance(oa, dict) else None
            best_oa = item.get("best_oa_location") or {}
            oa_pdf = ""
            if isinstance(best_oa, dict):
                oa_pdf = best_oa.get("pdf_url", "") or ""

            papers.append(RawPaper(
                title=item.get("display_name", "") or "",
                abstract=abstract_str,
                authors=authors,
                year=item.get("publication_year"),
                venue=venue,
                doi=doi,
                openalex_id=item.get("id", "").replace("https://openalex.org/", "") or "",
                url=item.get("id", "") or "",
                pdf_url=oa_pdf,
                is_oa=is_oa,
                citation_count=item.get("cited_by_count", 0) or 0,
                source=self.name,
                raw_data={"openalex_id": item.get("id", "")},
            ))
        return papers

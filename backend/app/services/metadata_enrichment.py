"""Metadata enrichment service (P1-8).

Problem: CORE/arXiv sources lack citation_count/venue/year for many papers,
leading to inaccurate authority scoring.

Solution: After papers are saved, asynchronously query Semantic Scholar / OpenAlex
by DOI or title to fill in missing citation_count, venue, year, and external IDs.

Usage:
    from app.services.metadata_enrichment import enrich_papers_metadata
    await enrich_papers_metadata(paper_ids, db)
"""

import asyncio
import logging
from typing import Optional

import httpx

from app.db.models import Paper
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# Only enrich papers missing these critical fields
_MISSING_METADATA_QUERY = (
    "((citation_count IS NULL OR citation_count = 0) "
    "OR (venue IS NULL OR venue = '') "
    "OR (year IS NULL))"
)

# S2 batch lookup endpoint (up to 500 papers per call)
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


async def _s2_lookup_single(doi: str = "", title: str = "") -> dict | None:
    """Query Semantic Scholar for a single paper by DOI or title.

    Returns dict with citationCount, year, venue, externalIds or None on failure.
    """
    fields = "title,year,venue,citationCount,externalIds,openAccessPdf"
    url = None
    if doi:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields={fields}"
    elif title:
        # Use search endpoint for title-based lookup
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search?query={title[:200]}"
            f"&limit=1&fields={fields}"
        )

    if not url:
        return None

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 429:
                logger.debug("S2 enrich rate limited")
                return None
            if resp.status_code != 200:
                return None
            data = resp.json()
            # For search endpoint, unwrap 'data' array
            if "data" in data and isinstance(data["data"], list):
                if not data["data"]:
                    return None
                data = data["data"][0]
            return data
    except Exception as e:
        logger.debug("S2 enrich lookup failed: %s", e)
        return None


async def _openalex_lookup(doi: str = "") -> dict | None:
    """Query OpenAlex for a single paper by DOI.

    Returns dict with cited_by_count, publication_year, venue_name or None.
    """
    if not doi:
        return None

    url = f"https://api.openalex.org/works/doi:{doi}?select=cited_by_count,publication_year,primary_location,locations"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "DeepResearch/1.0 (mailto:research@example.com)"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            result = {
                "citation_count": data.get("cited_by_count", 0),
                "year": data.get("publication_year"),
            }
            # Extract venue from primary_location
            primary = data.get("primary_location") or {}
            source = primary.get("source") or {}
            if source.get("display_name"):
                result["venue"] = source["display_name"]
            # Extract openalex_id
            if data.get("id"):
                result["openalex_id"] = data["id"].rsplit("/", 1)[-1]
            return result
    except Exception as e:
        logger.debug("OpenAlex enrich lookup failed: %s", e)
        return None


async def _enrich_one(paper: Paper) -> dict | None:
    """Enrich a single paper's metadata.

    Strategy:
    1. If has DOI → try OpenAlex (has citation + venue)
    2. If OpenAlex fails or no DOI → try S2 by DOI or title

    Returns dict of fields to update, or None if no enrichment possible.
    """
    result = {}

    # Try OpenAlex by DOI first (most reliable for citation counts)
    if paper.doi:
        oa_data = await _openalex_lookup(paper.doi)
        if oa_data:
            if oa_data.get("citation_count", 0) > (paper.citation_count or 0):
                result["citation_count"] = oa_data["citation_count"]
            if oa_data.get("year") and not paper.year:
                result["year"] = oa_data["year"]
            if oa_data.get("venue") and not paper.venue:
                result["venue"] = oa_data["venue"]
            if oa_data.get("openalex_id") and not paper.openalex_id:
                result["openalex_id"] = oa_data["openalex_id"]

    # Fallback: S2 by DOI or title
    needs_more = (
        not result.get("citation_count")
        or (not result.get("year") and not paper.year)
        or (not result.get("venue") and not paper.venue)
    )
    if needs_more:
        s2_data = await _s2_lookup_single(doi=paper.doi or "", title=paper.title or "")
        if s2_data:
            cc = s2_data.get("citationCount", 0) or 0
            if cc > result.get("citation_count", paper.citation_count or 0):
                result["citation_count"] = cc
            if s2_data.get("year") and not (result.get("year") or paper.year):
                result["year"] = s2_data["year"]
            if s2_data.get("venue") and not (result.get("venue") or paper.venue):
                result["venue"] = s2_data["venue"]
            ext = s2_data.get("externalIds") or {}
            if ext.get("DOI") and not paper.doi:
                result["doi"] = ext["DOI"]
            if ext.get("ArXiv") and not paper.arxiv_id:
                result["arxiv_id"] = ext["ArXiv"]
            s2_id = s2_data.get("paperId", "")
            if s2_id and not paper.semantic_scholar_id:
                result["semantic_scholar_id"] = s2_id

    return result if result else None


async def enrich_papers_metadata(paper_ids: list[str], db_session=None) -> dict:
    """Enrich metadata for a list of papers.

    Args:
        paper_ids: List of Paper IDs to enrich.
        db_session: Optional existing session. If None, creates a new one.

    Returns: {enriched, skipped, failed}
    """
    if not paper_ids:
        return {"enriched": 0, "skipped": 0, "failed": 0}

    own_session = db_session is None
    db = db_session or SessionLocal()

    try:
        papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
        # Filter to papers that actually need enrichment
        to_enrich = [
            p for p in papers
            if (p.citation_count or 0) == 0
            or not p.venue
            or p.year is None
        ]

        if not to_enrich:
            return {"enriched": 0, "skipped": len(papers), "failed": 0}

        logger.info("Enriching metadata for %d papers (of %d total)", len(to_enrich), len(papers))

        semaphore = asyncio.Semaphore(3)  # limit concurrent API calls
        enriched = 0
        failed = 0

        async def enrich_and_save(paper: Paper):
            nonlocal enriched, failed
            async with semaphore:
                try:
                    updates = await _enrich_one(paper)
                    if updates:
                        for field, value in updates.items():
                            setattr(paper, field, value)
                        enriched += 1
                        logger.debug("Enriched paper %s: %s", paper.id[:8],
                                    list(updates.keys()))
                    else:
                        # No enrichment available — not a failure
                        pass
                except Exception as e:
                    logger.debug("Enrich failed for paper %s: %s", paper.id[:8], e)
                    failed += 1

        # Process in small batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(to_enrich), batch_size):
            batch = to_enrich[i:i + batch_size]
            await asyncio.gather(*[enrich_and_save(p) for p in batch])
            # Commit after each batch
            if own_session:
                db.commit()
            # Small delay between batches to be polite
            await asyncio.sleep(0.5)

        if own_session:
            db.commit()

        logger.info("Metadata enrichment done: enriched=%d, failed=%d, skipped=%d",
                    enriched, failed, len(papers) - len(to_enrich))
        return {"enriched": enriched, "skipped": len(papers) - len(to_enrich), "failed": failed}

    except Exception as e:
        logger.error("Metadata enrichment batch failed: %s", e)
        if own_session:
            db.rollback()
        return {"enriched": 0, "skipped": 0, "failed": len(paper_ids)}
    finally:
        if own_session:
            db.close()

"""Paper normalization and deduplication service."""

import hashlib
import json
import re
from datetime import datetime

from app.paper_sources.base import RawPaper


def normalize_title(title: str) -> str:
    """Normalize a paper title for deduplication."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def title_hash(title: str) -> str:
    """SHA-256 hash of normalized title."""
    return hashlib.sha256(normalize_title(title).encode()).hexdigest()


def normalize_paper(raw: RawPaper, extra_sources: list[str] | None = None) -> dict:
    """Convert a RawPaper into a normalized dict suitable for DB insertion."""
    sources = [raw.source]
    if extra_sources:
        sources = list(set(sources + extra_sources))

    return {
        "title": raw.title,
        "abstract": raw.abstract,
        "authors_json": json.dumps(raw.authors, ensure_ascii=False),
        "year": raw.year,
        "venue": raw.venue,
        "doi": raw.doi or None,
        "arxiv_id": raw.arxiv_id or None,
        "semantic_scholar_id": raw.semantic_scholar_id or None,
        "openalex_id": raw.openalex_id or None,
        "url": raw.url,
        "pdf_url": raw.pdf_url,
        "is_oa": raw.is_oa,
        "citation_count": raw.citation_count,
        "sources_json": json.dumps(sources, ensure_ascii=False),
        "normalized_title": normalize_title(raw.title),
        "title_hash": title_hash(raw.title),
    }


def deduplicate_papers(raw_papers: list[RawPaper]) -> list[RawPaper]:
    """Deduplicate papers within a batch using priority-based matching."""
    seen: dict[str, RawPaper] = {}

    for paper in raw_papers:
        # Determine dedup key by priority
        key = None
        if paper.doi:
            key = f"doi:{paper.doi}"
        elif paper.arxiv_id:
            key = f"arxiv:{paper.arxiv_id}"
        elif paper.semantic_scholar_id:
            key = f"s2:{paper.semantic_scholar_id}"
        elif paper.openalex_id:
            key = f"openalex:{paper.openalex_id}"
        else:
            key = f"hash:{title_hash(paper.title)}"

        if key in seen:
            # Merge sources into existing
            existing = seen[key]
            existing_sources = set()
            if existing.raw_data.get("_merged_sources"):
                existing_sources.update(existing.raw_data["_merged_sources"])
            existing_sources.add(existing.source)
            existing_sources.add(paper.source)
            existing.raw_data["_merged_sources"] = list(existing_sources)
            # Fill in missing fields
            for attr in ["doi", "arxiv_id", "semantic_scholar_id", "openalex_id", "url", "pdf_url", "abstract", "venue"]:
                if not getattr(existing, attr) and getattr(paper, attr):
                    setattr(existing, attr, getattr(paper, attr))
            if paper.citation_count > existing.citation_count:
                existing.citation_count = paper.citation_count
        else:
            seen[key] = paper

    return list(seen.values())

"""Step: Search, deduplicate, and save papers for a round."""

import asyncio
import logging

from app.config import settings
from app.agent.state import ResearchState
from app.db.repositories import paper_repo
from app.services.search_service import search_service
from app.services.scoring_service import normalize_paper, deduplicate_papers
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def search_and_save_papers(db, state: ResearchState, queries: list[str], task_id: str,
                                  round_num: int) -> tuple[list[dict], int, int]:
    """Search papers for queries, deduplicate, and save to DB.

    Returns: (raw_papers_count, deduped_count, new_paper_ids)
    """
    raw_papers = await search_service.search_multiple_queries(
        queries, settings.papers_per_source_per_query
    )
    papers_found = len(raw_papers)
    logger.info("Round %d: found %d raw papers", round_num, papers_found)
    emit_event(task_id, "search_done", {"round": round_num, "found": papers_found})

    # Deduplicate within batch
    deduped = deduplicate_papers(raw_papers)
    logger.info("Round %d: %d papers after dedup", round_num, len(deduped))

    # Save to DB and track new vs existing
    # Use a set for O(1) membership check (collected_paper_ids can grow to hundreds)
    collected_set = set(state.collected_paper_ids)
    new_paper_ids = []
    for raw in deduped:
        normalized = normalize_paper(raw)
        paper, is_new = paper_repo.upsert_paper(db, normalized)
        paper_repo.create_task_paper(db, task_id, paper.id, round_num)
        if is_new:
            new_paper_ids.append(paper.id)
        if paper.id not in collected_set:
            state.collected_paper_ids.append(paper.id)
            collected_set.add(paper.id)

        # Fill citation relationships from S2 references (non-fatal)
        if is_new:
            try:
                _save_paper_citations(db, paper, raw, task_id)
            except Exception as e:
                logger.debug("Citation extraction failed for paper %s: %s", paper.id[:8], e)

    db.commit()
    logger.info("Round %d: %d new, %d dup", round_num, len(new_paper_ids), len(deduped) - len(new_paper_ids))

    # P1-8: Async metadata enrichment for newly found papers (non-blocking)
    if new_paper_ids:
        asyncio.create_task(_trigger_metadata_enrichment(new_paper_ids))

    return papers_found, len(deduped), new_paper_ids


async def _trigger_metadata_enrichment(paper_ids: list[str]):
    """Fire-and-forget metadata enrichment for newly discovered papers.

    Runs in background — does not block the search loop. Failures are logged
    but do not affect the main task.
    """
    try:
        from app.services.metadata_enrichment import enrich_papers_metadata
        result = await enrich_papers_metadata(paper_ids)
        if result.get("enriched", 0) > 0:
            logger.info("Metadata enrichment: %s", result)
    except Exception as e:
        logger.warning("Metadata enrichment failed (non-fatal): %s", e)


def _save_paper_citations(db, source_paper, raw, task_id: str):
    """Extract references from raw paper data and save as PaperCitation edges.

    Parses S2 'references' field (list of {paperId, title}) and
    OpenAlex 'referenced_works' (list of OpenAlex IDs).
    Only creates edges to papers already in our DB (avoid creating stub papers).
    """
    from app.db.models import Paper
    from app.services.scoring_service import normalize_title, title_hash

    raw_data = raw.raw_data or {}
    source_id = source_paper.id

    # S2 references: [{"paperId": "abc123", "title": "..."}, ...]
    references = raw_data.get("references") or []
    citation_edges = 0

    for ref in references[:50]:  # cap at 50 to avoid huge graphs
        ref_s2_id = ref.get("paperId", "") if isinstance(ref, dict) else ""
        ref_title = ref.get("title", "") if isinstance(ref, dict) else ""

        if not ref_s2_id and not ref_title:
            continue

        # Try to find the referenced paper in our DB
        target = None
        if ref_s2_id:
            target = db.query(Paper).filter(Paper.semantic_scholar_id == ref_s2_id).first()
        if not target and ref_title:
            th = title_hash(ref_title)
            target = db.query(Paper).filter(Paper.title_hash == th).first()

        if target and target.id != source_id:
            paper_repo.save_citation(
                db, source_id, target.id, "cites", weight=1.0, task_id=task_id
            )
            citation_edges += 1

    if citation_edges:
        logger.debug("Paper %s: saved %d citation edges", source_id[:8], citation_edges)

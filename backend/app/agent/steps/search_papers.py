"""Step: Search, deduplicate, and save papers for a round.

Phase 2.2A Hotfix:
- Accepts list[SearchQueryExecution] (not list[str])
- Per-query search with SearchQueryRecord lifecycle (pending → completed/failed)
- SearchQueryPaper mapping for provenance
- Partial query failure handling with minimum success threshold
"""

import asyncio
import logging
from dataclasses import dataclass

from app.config import settings
from app.agent.state import ResearchState
from app.db.repositories import paper_repo
from app.db.repositories.search_query_repo import update_query_results
from app.services.search_service import search_service
from app.services.scoring_service import normalize_paper, deduplicate_papers
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


@dataclass
class SearchRoundExecutionResult:
    """Result of executing all queries in a search round."""
    papers_found: int
    deduped_count: int
    new_paper_ids: list[str]
    completed_query_ids: list[str]
    failed_query_ids: list[str]


# Minimum success threshold: at least 1 query success and success rate >= 50%
MIN_SUCCESS_QUERIES = 1
MIN_SUCCESS_RATE = 0.5


async def search_and_save_papers(db, state: ResearchState,
                                  query_executions: list, task_id: str,
                                  round_num: int) -> tuple[int, int, list[str]]:
    """Search papers for each query, deduplicate, and save to DB.

    Args:
        query_executions: list[SearchQueryExecution] from generate_queries.

    Returns: (papers_found, deduped_count, new_paper_ids)
    """
    from app.db.models import SearchQueryPaper

    collected_set = set(state.collected_paper_ids)
    all_raw_papers = []
    completed_query_ids = []
    failed_query_ids = []
    new_paper_ids = []  # tracked as papers newly added for this task

    # Track all papers found per query for SearchQueryPaper mapping
    query_paper_mappings: list[tuple[str, str, int, str, bool]] = []
    # (query_id, paper_id, rank, source, is_new_for_task)

    for qe in query_executions:
        query_text = qe.query_text
        query_id = qe.query_id

        try:
            raw_papers = await search_service.search_multiple_queries(
                [query_text], settings.papers_per_source_per_query
            )
            raw_count = len(raw_papers)
            logger.info("Round %d query '%s': found %d raw papers",
                        round_num, query_text[:40], raw_count)

            # Save each paper and create SearchQueryPaper mapping
            new_paper_count_for_query = 0
            for rank, raw in enumerate(raw_papers):
                normalized = normalize_paper(raw)
                paper, is_new = paper_repo.upsert_paper(db, normalized)
                paper_repo.create_task_paper(db, task_id, paper.id, round_num)

                source_name = raw.source if hasattr(raw, 'source') else 'unknown'
                is_new_for_task = paper.id not in collected_set
                query_paper_mappings.append(
                    (query_id, paper.id, rank, source_name, is_new_for_task)
                )

                if is_new_for_task:
                    new_paper_count_for_query += 1
                    new_paper_ids.append(paper.id)
                if paper.id not in collected_set:
                    state.collected_paper_ids.append(paper.id)
                    collected_set.add(paper.id)

                if is_new:
                    try:
                        _save_paper_citations(db, paper, raw, task_id)
                    except Exception as e:
                        logger.debug("Citation extraction failed for paper %s: %s", paper.id[:8], e)

                all_raw_papers.append(raw)

            # Mark query as completed
            update_query_results(
                db, query_id,
                result_count=raw_count,
                new_paper_count=new_paper_count_for_query,
                status="completed",
            )
            completed_query_ids.append(query_id)
            logger.info("Round %d query '%s': %d new papers (total %d)",
                        round_num, query_text[:40], new_paper_count_for_query, raw_count)

        except Exception as e:
            logger.error("Round %d query '%s' failed: %s", round_num, query_text[:40], e)
            update_query_results(
                db, query_id,
                result_count=0,
                new_paper_count=0,
                status="failed",
                error=str(e)[:500],
            )
            failed_query_ids.append(query_id)

    db.flush()

    # Save SearchQueryPaper mappings
    for query_id, paper_id, rank, source_name, is_new_for_task in query_paper_mappings:
        existing_sqp = db.query(SearchQueryPaper).filter(
            SearchQueryPaper.query_id == query_id,
            SearchQueryPaper.paper_id == paper_id,
            SearchQueryPaper.source == source_name,
        ).first()
        if not existing_sqp:
            sqp = SearchQueryPaper(
                query_id=query_id,
                paper_id=paper_id,
                rank=rank,
                source=source_name,
                is_new_for_task=is_new_for_task,
            )
            db.add(sqp)

    db.commit()

    # Ensure no query stays pending
    for qe in query_executions:
        if qe.query_id not in completed_query_ids and qe.query_id not in failed_query_ids:
            update_query_results(db, qe.query_id, 0, 0,
                                  status="failed", error="threshold_not_met")

    # Check minimum success threshold
    total_queries = len(query_executions)
    successful_queries = len(completed_query_ids)
    if total_queries > 0:
        success_rate = successful_queries / total_queries
        if successful_queries < MIN_SUCCESS_QUERIES or success_rate < MIN_SUCCESS_RATE:
            logger.error("Round %d: search below threshold (%d/%d succeeded, rate=%.1f%%)",
                         round_num, successful_queries, total_queries, success_rate * 100)
            db.commit()
            raise RuntimeError(
                f"Search round {round_num} below minimum success threshold: "
                f"{successful_queries}/{total_queries} queries succeeded"
            )

    papers_found = len(all_raw_papers)
    # Deduplicate for reporting
    deduped = deduplicate_papers(all_raw_papers)
    # Dedupe new_paper_ids
    seen = set()
    unique_new_paper_ids = []
    for pid in new_paper_ids:
        if pid not in seen:
            seen.add(pid)
            unique_new_paper_ids.append(pid)
    new_paper_ids = unique_new_paper_ids

    emit_event(task_id, "search_done", {
        "round": round_num,
        "found": papers_found,
        "completed_queries": len(completed_query_ids),
        "failed_queries": len(failed_query_ids),
    })
    logger.info("Round %d: %d raw papers, %d after dedup, %d completed queries, %d failed",
                round_num, papers_found, len(deduped), len(completed_query_ids), len(failed_query_ids))

    # P1-8: Async metadata enrichment for newly found papers (non-blocking)
    if new_paper_ids:
        asyncio.create_task(_trigger_metadata_enrichment(list(new_paper_ids)))

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

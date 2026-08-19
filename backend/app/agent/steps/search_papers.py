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
    from app.db.models import SearchQueryPaper, SearchRawResult

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

            # Persist minimal raw-retrieval identity BEFORE the similarity
            # pre-filter, so the recall waterfall diagnostic can compare query
            # variants on raw external ids (not just the canonical survivors).
            # raw_rank is the paper's position within its source's result list
            # for this query, numbered here (not by the source) so the rank is
            # always unique per (query, source) even when a test/fake source
            # does not tag its papers.
            raw_result_rows: dict[tuple[str, int], SearchRawResult] = {}
            # Idempotency: a query may be re-executed across retries (same
            # SearchQueryRecord, e.g. a coverage retry re-runs the search phase).
            # Keep the first raw snapshot and don't duplicate rows for it.
            _already_has_raw = db.query(SearchRawResult.id).filter(
                SearchRawResult.query_id == query_id,
            ).limit(1).first() is not None
            if not _already_has_raw:
                _source_rank: dict[str, int] = {}
                for raw in raw_papers:
                    src = (getattr(raw, "source", "") or "unknown")
                    rk = _source_rank.get(src, 0)
                    _source_rank[src] = rk + 1
                    row = SearchRawResult(
                        query_id=query_id,
                        source=src,
                        raw_rank=rk,
                        external_paper_id=_external_paper_id(raw),
                        doi=getattr(raw, "doi", None) or None,
                        arxiv_id=getattr(raw, "arxiv_id", None) or None,
                        title=getattr(raw, "title", "") or "",
                        year=getattr(raw, "year", None),
                    )
                    db.add(row)
                    raw_result_rows[(src, rk)] = row

            # O7: prefilter by topic similarity before persisting, to keep
            # off-topic noise out of the evidence/gap pipeline.
            raw_papers = await _prefilter_by_similarity(raw_papers, state, round_num, query_text)

            # Save each paper and create SearchQueryPaper mapping
            new_paper_count_for_query = 0
            _source_rank2: dict[str, int] = {}
            for rank, raw in enumerate(raw_papers):
                normalized = normalize_paper(raw)
                paper, is_new = paper_repo.upsert_paper(db, normalized)
                paper_repo.create_task_paper(db, task_id, paper.id, round_num)

                source_name = raw.source if hasattr(raw, 'source') else 'unknown'
                # Backfill the canonical paper id onto the matching raw result
                # (same per-source ordering as the pre-filter numbering above),
                # so the diagnostic can join raw identity to final scoring.
                _rk = _source_rank2.get(source_name, 0)
                _source_rank2[source_name] = _rk + 1
                _rr = raw_result_rows.get((source_name, _rk))
                if _rr is not None:
                    _rr.canonical_paper_id = paper.id
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
    # NOTE: the same (query_id, paper_id, source) can appear multiple times in
    # query_paper_mappings within a single round (e.g. a paper returned under
    # the same source for a query more than once). The DB has a UNIQUE
    # constraint on (query_id, paper_id, source), and checking only committed
    # rows via `existing_sqp` misses duplicates already staged in this batch,
    # which caused an IntegrityError that rolled back the whole round. Track a
    # per-batch seen-set so we only add each unique triple once.
    seen_sqp: set[tuple] = set()
    for query_id, paper_id, rank, source_name, is_new_for_task in query_paper_mappings:
        key = (query_id, paper_id, source_name)
        if key in seen_sqp:
            continue
        seen_sqp.add(key)
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

    # P1-8: Async metadata enrichment for newly found papers (non-blocking).
    # O3: skippable — without an S2 key it competes with the main search loop
    # for the same rate-limited quota and slows it down for little gain.
    if new_paper_ids and settings.enable_metadata_enrichment:
        asyncio.create_task(_trigger_metadata_enrichment(list(new_paper_ids)))

    return papers_found, len(deduped), new_paper_ids


def _external_paper_id(raw) -> str | None:
    """Best available external identifier for a raw paper (S2/OpenAlex/arXiv/DOI).

    Used as the raw-layer identity in the overlap waterfall. Fallback to None
    when the source provided no external id; the diagnostic then falls back to
    a title hash so the paper still participates in raw overlap.
    """
    for attr in ("semantic_scholar_id", "openalex_id", "arxiv_id", "doi"):
        v = getattr(raw, attr, None)
        if v:
            return str(v)
    return None


async def _prefilter_by_similarity(raw_papers: list, state: ResearchState,
                round_num: int, query_text: str) -> list:
    """O7: drop papers whose title+abstract is semantically far from the topic.

    Uses the pluggable embedding backend. On any failure (embedding disabled,
    API error) it returns the input unchanged — prefiltering is best-effort and
    must never drop the whole round. The topic embedding is cached on `state`
    across queries to avoid recomputation.
    """
    threshold = settings.search_prefilter_min_similarity
    # Papers without an abstract can't feed the evidence pipeline (abstract is
    # the fallback when PDF fetch fails), so hold them to a stronger title-only
    # match instead of admitting weak no-abstract noise into the store.
    no_abstract_threshold = max(
        threshold,
        getattr(settings, "search_prefilter_no_abstract_min_similarity", threshold),
    )
    if threshold <= 0 or not raw_papers:
        return raw_papers

    try:
        from app.services.embedding_service import (
            embed_texts, cosine_similarity, embedding_enabled,
        )
        import asyncio

        if not embedding_enabled():
            return raw_papers

        topic = state.normalized_topic or state.user_input or query_text

        def _score() -> list:
            # Embed topic + all candidate texts in one batched call.
            texts = [topic]
            for rp in raw_papers:
                title = getattr(rp, "title", "") or ""
                abstract = getattr(rp, "abstract", "") or ""
                texts.append(f"{title}. {abstract}"[:2000])
            vecs = embed_texts(texts)
            if not vecs:
                return raw_papers
            topic_vec = vecs[0]
            kept = []
            dropped = 0
            for rp, vec in zip(raw_papers, vecs[1:]):
                sim = cosine_similarity(topic_vec, vec)
                has_abstract = bool(getattr(rp, "abstract", ""))
                effective = threshold if has_abstract else no_abstract_threshold
                if sim >= effective:
                    kept.append(rp)
                else:
                    dropped += 1
            if dropped:
                logger.info("Round %d query '%s': O7 prefilter dropped %d/%d off-topic papers",
                            round_num, query_text[:40], dropped, len(raw_papers))
            return kept

        return await asyncio.to_thread(_score)
    except Exception as e:
        logger.warning("O7 prefilter failed (non-fatal, keeping all): %s", e)
        return raw_papers


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

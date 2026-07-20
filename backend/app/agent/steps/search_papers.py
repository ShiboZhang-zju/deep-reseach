"""Step: Search, deduplicate, and save papers for a round."""

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
    new_paper_ids = []
    for raw in deduped:
        normalized = normalize_paper(raw)
        paper, is_new = paper_repo.upsert_paper(db, normalized)
        paper_repo.create_task_paper(db, task_id, paper.id, round_num)
        if is_new:
            new_paper_ids.append(paper.id)
        if paper.id not in state.collected_paper_ids:
            state.collected_paper_ids.append(paper.id)

    db.commit()
    logger.info("Round %d: %d new, %d dup", round_num, len(new_paper_ids), len(deduped) - len(new_paper_ids))

    return papers_found, len(deduped), new_paper_ids

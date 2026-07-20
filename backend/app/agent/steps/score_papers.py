"""Step: Score papers using LLM with concurrent semaphore."""

import asyncio
import logging

from app.agent.state import ResearchState
from app.agent.prompts import SCORE_SYSTEM, SCORE_USER
from app.db.models import TaskPaper, Paper
from app.db.repositories import paper_repo
from app.schemas.schemas import PaperScore
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

# Top-venue keywords for authority boost
TOP_VENUE_KEYWORDS = [
    "ICML", "NeurIPS", "ICLR", "CVPR", "ACL", "EMNLP",
    "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "TSE", "TACL",
    "Nature", "Science", "PAMI", "JMLR", "ICSE", "FSE",
    "ASE", "ISSTA", "OOPSLA", "PLDI",
]


async def score_papers(db, state: ResearchState, llm, task_id: str, round_num: int):
    """Score new papers from this round (concurrent with semaphore)."""
    # Get unscored task papers from this round
    unscored = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.discovered_round == round_num,
        TaskPaper.final_score.is_(None),
    ).all()

    # Pre-fetch all papers (avoid DB access during concurrent LLM calls)
    paper_map = {}
    for tp in unscored:
        paper = db.get(Paper, tp.paper_id)
        if paper:
            paper_map[tp.id] = (tp, paper)

    if not paper_map:
        return []

    logger.info("Scoring %d papers in round %d (concurrent, max 5)...", len(paper_map), round_num)

    # Concurrent scoring with semaphore
    semaphore = asyncio.Semaphore(5)

    async def score_one(tp_id: str):
        tp, paper = paper_map[tp_id]
        messages = [
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user", "content": SCORE_USER.format(
                topic=state.normalized_topic,
                title=paper.title,
                abstract=(paper.abstract or "")[:1000],
                authors=paper.authors_json or "",
                year=paper.year or "",
                venue=paper.venue or "",
                citations=paper.citation_count or 0,
            )},
        ]
        async with semaphore:
            try:
                score = await llm.chat_json(messages, PaperScore)
                return tp_id, score, None
            except Exception as e:
                logger.error("Failed to score paper %s: %s", paper.id, e)
                return tp_id, None, str(e)

    results = await asyncio.gather(*[score_one(tp_id) for tp_id in paper_map])

    # Process results sequentially (DB writes)
    scored = []
    for tp_id, score, error in results:
        if score is None:
            continue

        tp, paper = paper_map[tp_id]
        authority_adj = score.authority
        # Penalize papers with missing metadata (no citations + no year)
        if (paper.citation_count or 0) == 0 and paper.year is None:
            authority_adj = score.authority * 0.7
        # Boost papers from top venues
        venue_str = (paper.venue or "").upper()
        if any(kv in venue_str for kv in TOP_VENUE_KEYWORDS):
            authority_adj = min(1.0, authority_adj + 0.1)

        final_score = (
            0.30 * score.relevance + 0.25 * authority_adj + 0.15 * score.recency +
            0.15 * score.novelty + 0.15 * score.idea_potential
        )
        priority = "high" if final_score >= 0.75 else ("medium" if final_score >= 0.5 else "low")

        paper_repo.update_task_paper_scores(
            db, tp.id, score.model_dump(), final_score, priority,
            score.reason, f"{score.summary} | 方法: {score.method_extract}" if score.method_extract else score.summary
        )

        if priority == "high":
            if paper.id not in state.high_priority_paper_ids:
                state.high_priority_paper_ids.append(paper.id)
        elif priority == "medium":
            if paper.id not in state.medium_priority_paper_ids:
                state.medium_priority_paper_ids.append(paper.id)
        else:
            if paper.id not in state.low_priority_paper_ids:
                state.low_priority_paper_ids.append(paper.id)

        scored.append({
            "title": paper.title,
            "score": final_score,
            "priority": priority,
            "summary": score.summary,
        })

    logger.info("Scored %d/%d papers in round %d", len(scored), len(paper_map), round_num)

    # Record LLM token usage
    total_tokens = 0
    if hasattr(llm, "get_last_usage") and llm.get_last_usage():
        total_tokens = llm.get_last_usage().get("total_tokens", 0)

    paper_repo.save_trace(db, state.task_id, "score_papers", "action",
                          round_number=round_num,
                          output_data={"scored_count": len(scored)},
                          tokens=total_tokens)
    db.commit()
    return scored

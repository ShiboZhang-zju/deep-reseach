"""Step: Score papers using LLM with concurrent semaphore."""

import asyncio
import logging
import re

from app.config import settings
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


def _calibrate_batch_scores(raw_scores: list[float]) -> list[float]:
    """Widen priority separation via batch z-score, then clamp to [0, 1].

    Historically, per-paper independent LLM scoring produced near-identical
    final scores (spread ~0.05), making priority tiers meaningless. Here we add
    a fraction of each paper's deviation from the batch mean to its raw score so
    that above-average papers move up and below-average papers move down. The
    batch mean stays stable, so the overall 0.75/0.5 thresholds still apply.
    Disabled for small batches where a mean is not meaningful.
    """
    n = len(raw_scores)
    if n == 0:
        return raw_scores
    if settings.score_calibration_min_batch <= 0 or n < settings.score_calibration_min_batch:
        return raw_scores
    mean = sum(raw_scores) / n
    var = sum((s - mean) ** 2 for s in raw_scores) / n
    std = var ** 0.5
    if std == 0:
        return raw_scores
    strength = settings.score_calibration_strength
    # adjusted = s + strength * (s - mean): pulls apart around the mean.
    return [max(0.0, min(1.0, s + strength * (s - mean))) for s in raw_scores]


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

        # P1-7: Retrieve RAG passages to ground method_extract (avoids hallucination)
        # NOTE: RAG retrieval only makes sense AFTER PDF indexing (Phase 2.5).
        # During the search loop (Phase 2), ChromaDB is empty, so retrieval always
        # returns []. Skip it entirely to avoid 220 wasted to_thread calls that
        # exhaust the thread pool and cause process crashes.
        # We check SQLite (has_chunks) instead of ChromaDB to avoid triggering
        # embedding model loading on every paper.
        rag_context = ""
        try:
            from app.db.repositories.paper_repo import has_chunks
            if has_chunks(db, paper.id):
                # Paper has indexed chunks — retrieve from ChromaDB
                from app.services.rag_service import rag_retrieve
                rag_results = await rag_retrieve(
                    query=paper.title or state.normalized_topic,
                    top_k=3,
                    paper_ids=[paper.id],
                    section_filter=["method", "experiment"],
                )
                if rag_results:
                    _fig_pat = re.compile(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]')
                    rag_context = "\n".join(
                        f"({r['section']}) {_fig_pat.sub('', r['text'])[:400].strip()}"
                        for r in rag_results[:3]
                    )
        except Exception as e:
            logger.debug("RAG retrieve for scoring paper %s failed (non-fatal): %s",
                        paper.id[:8], e)

        user_content = SCORE_USER.format(
            topic=state.normalized_topic,
            title=paper.title,
            abstract=(paper.abstract or "")[:1000],
            authors=paper.authors_json or "",
            year=paper.year or "",
            venue=paper.venue or "",
            citations=paper.citation_count or 0,
        )
        if rag_context:
            user_content += f"\n\n## 论文全文段落（RAG检索，用于准确提取方法细节）\n{rag_context}"

        messages = [
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user", "content": user_content},
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
    # Pass 1: compute raw final scores (no DB write yet) so we can calibrate
    # across the whole batch before assigning priority tiers.
    computed = []  # (tp_id, score, final_score)
    for tp_id, score, error in results:
        if score is None:
            continue
        tp, paper = paper_map[tp_id]
        authority_adj = score.authority
        # P2: Citation-based authority adjustment with finer granularity
        citations = paper.citation_count or 0
        if citations == 0 and paper.year is None:
            authority_adj = score.authority * 0.6   # both missing: heavy penalty
        elif citations == 0:
            authority_adj = score.authority * 0.85  # citation=0 but year present: moderate penalty
        elif citations >= 100:
            authority_adj = min(1.0, score.authority + 0.1)  # high-citation: boost
        elif citations >= 10:
            authority_adj = min(1.0, score.authority + 0.05)  # medium-citation: small boost
        # Boost papers from top venues
        venue_str = (paper.venue or "").upper()
        if any(kv in venue_str for kv in TOP_VENUE_KEYWORDS):
            authority_adj = min(1.0, authority_adj + 0.1)

        # P1-7: use configurable weights (authority bumped to 0.30, relevance down to 0.25)
        final_score = (
            settings.score_weight_relevance * score.relevance
            + settings.score_weight_authority * authority_adj
            + settings.score_weight_recency * score.recency
            + settings.score_weight_novelty * score.novelty
            + settings.score_weight_idea_potential * score.idea_potential
        )
        computed.append((tp_id, score, final_score))

    # Cross-paper calibration: widen separation via batch z-score. Absolute
    # scores are nudged (not replaced) so downstream 0.75/0.5 thresholds still
    # apply, but near-tied papers get pulled apart.
    calibrated = _calibrate_batch_scores([c[2] for c in computed])

    scored = []
    for (tp_id, score, _raw), final_score in zip(computed, calibrated):
        tp, paper = paper_map[tp_id]
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

"""Step: Generate search queries driven by Research Questions.

Phase 2.2A:
- LLM explicitly outputs target_question_id for each query
- No round-robin assignment
- Validates target_question_id belongs to active questions
- Saves SearchQueryRecord with normalized_query_text
"""

import hashlib
import json
import logging
import re

from app.config import settings
from app.agent.state import ResearchState
from app.agent.prompts import QUERIES_SYSTEM, QUERIES_USER
from app.db.repositories import paper_repo
from app.db.repositories.search_query_repo import save_search_query
from app.schemas.schemas import QueryList, GeneratedQuery, SearchIntent
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def generate_queries(db, state: ResearchState, llm) -> list[str]:
    """Generate search queries for the current round.

    Phase 2.2A: Returns list[str] for backward compat with search_and_save_papers.
    Each query is also saved as SearchQueryRecord with target_question_id.
    """
    from app.agent.steps.decompose_research_space import select_target_questions

    target_questions = select_target_questions(db, state.task_id, limit=3)

    if target_questions and state.pipeline_version >= 2:
        return await _generate_structured_queries(db, state, llm, target_questions)
    else:
        return await _generate_legacy_queries(db, state, llm)


async def _generate_structured_queries(
    db, state: ResearchState, llm, target_questions
) -> list[str]:
    """Generate structured queries with LLM-assigned target_question_id."""
    from app.db.models import ResearchContract

    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    preferred = json.loads(contract.preferred_directions_json or "[]") if contract else []
    excluded = json.loads(contract.excluded_directions_json or "[]") if contract else []
    key_terms = json.loads(contract.key_terms_json or "[]") if contract else []

    # Build question context with FULL IDs (not truncated)
    question_contexts = []
    valid_question_ids = set()
    for q in target_questions:
        question_contexts.append(
            f"- Question ID: {q.id}\n  Type: {q.question_type}\n  Importance: {q.importance:.1f}\n  Question: {q.question}"
        )
        valid_question_ids.add(q.id)

    system_prompt = f"""You are a research search strategist. Generate {settings.queries_per_round} search queries targeting SPECIFIC Research Questions.

Each query MUST include:
- query_text: A concise search phrase (3-8 words) for academic search engines
- intent: One of survey, seminal, recent_work, benchmark, direct_neighbor, limitation, negative_result, question_answering, gap_falsification
- target_question_id: The FULL Question ID from the list below (must be one of the provided IDs)
- expected_evidence_type: What type of evidence you expect to find

Guidelines:
- Vary intent across queries
- Include alternative terminology and synonyms
- Avoid repeating previously used queries
- Focus on finding high-quality, relevant papers

Preferred directions: {', '.join(preferred) if preferred else '(none)'}
Excluded directions: {', '.join(excluded) if excluded else '(none)'}
Key terms: {', '.join(key_terms) if key_terms else '(none)'}"""

    user_prompt = f"""Research topic: {state.normalized_topic}

Available Research Questions (use these IDs for target_question_id):
{chr(10).join(question_contexts)}

Previous queries used:
{chr(10).join(state.used_queries[-20:]) if state.used_queries else '(none)'}

User feedback: {state.user_feedback or '(none)'}

Generate {settings.queries_per_round} structured queries. Each must target one of the provided Question IDs."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try up to 2 times (initial + 1 retry)
    valid_queries = []
    for attempt in range(2):
        result = await llm.chat_json(messages, QueryList)

        valid_queries = []
        invalid_queries = []

        for q in result.queries:
            if q.target_question_id in valid_question_ids:
                valid_queries.append(q)
            else:
                invalid_queries.append(q)
                logger.warning("Task %s: query '%s' has invalid target_question_id '%s' (not in %s)",
                             state.task_id[:8],
                             q.query_text[:30], q.target_question_id[:12],
                             [qid[:8] for qid in valid_question_ids])

        if not invalid_queries or attempt == 1:
            # All valid, or this was the retry
            if invalid_queries:
                logger.warning("Task %s: %d queries with invalid IDs after retry, rejecting them",
                             state.task_id[:8], len(invalid_queries))
            break

        logger.info("Task %s: retrying query generation (%d invalid IDs)", state.task_id[:8], len(invalid_queries))

    if not valid_queries:
        logger.error("Task %s: no valid queries generated (all target_question_ids invalid)", state.task_id[:8])
        raise ValueError("No valid queries generated — all target_question_ids were invalid")

    # Save each query as SearchQueryRecord
    saved_query_texts = []
    for q in valid_queries:
        normalized = _normalize_query_text(q.query_text)

        record = save_search_query(
            db, state.task_id, q.query_text, q.intent,
            q.target_question_id,
            q.expected_evidence_type,
            state.current_round,
        )
        saved_query_texts.append(q.query_text)

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={
                              "queries": [q.query_text for q in valid_queries],
                              "structured_queries": [
                                  {
                                      "query_text": q.query_text,
                                      "intent": q.intent,
                                      "target_question_id": q.target_question_id,
                                      "expected_evidence_type": q.expected_evidence_type,
                                  }
                                  for q in valid_queries
                              ],
                              "target_question_ids": [q.id for q in target_questions],
                              "intent": "structured_llm_bound",
                              "invalid_count": len(invalid_queries) if 'invalid_queries' in dir() else 0,
                          })
    db.commit()
    return saved_query_texts


async def _generate_legacy_queries(db, state: ResearchState, llm) -> list[str]:
    """Legacy query generation (fallback)."""
    messages = [
        {"role": "system", "content": QUERIES_SYSTEM.format(num_queries=settings.queries_per_round)},
        {"role": "user", "content": QUERIES_USER.format(
            topic=state.normalized_topic,
            keywords=", ".join(state.keywords),
            used_queries="\n".join(state.used_queries[-20:]) if state.used_queries else "(none)",
            gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
            feedback=state.user_feedback or "(none)",
            num_queries=settings.queries_per_round,
        )},
    ]

    # Legacy returns list[str] — wrap as GeneratedQuery for DB saving
    from app.schemas.schemas import QueryList as LegacyQueryList
    # Try structured output first; if LLM returns strings, adapt
    try:
        result = await llm.chat_json(messages, QueryList)
        query_texts = [q.query_text for q in result.queries]
    except Exception:
        # Fallback: try old QueryList that returns list[str]
        legacy_result = await llm.chat_json(messages, LegacyQueryList)
        query_texts = legacy_result.queries if hasattr(legacy_result, 'queries') else []

    for query_text in query_texts:
        save_search_query(db, state.task_id, query_text, "survey",
                          None, None, state.current_round)

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={"queries": query_texts, "intent": "legacy"})
    db.commit()
    return query_texts


def _normalize_query_text(text: str) -> str:
    """Normalize query text for uniqueness checking."""
    return re.sub(r'\s+', ' ', (text or '').strip().lower())

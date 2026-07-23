"""Step: Generate search queries driven by Research Questions.

Phase 2.2A Closure:
- Returns list[SearchQueryExecution] (not list[str])
- Proper retry: preserve valid, only re-request invalid
- min_valid_queries threshold
- SearchQueryRecord lifecycle: pending → completed/failed
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, asdict

from app.config import settings
from app.agent.state import ResearchState
from app.agent.prompts import QUERIES_SYSTEM, QUERIES_USER
from app.db.repositories import paper_repo
from app.db.repositories.search_query_repo import save_search_query, update_query_results
from app.schemas.schemas import QueryList, GeneratedQuery, SearchIntent
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


@dataclass
class SearchQueryExecution:
    """Structured query execution record returned by generate_queries."""
    query_id: str
    query_text: str
    intent: str
    target_question_id: str
    expected_evidence_type: str | None


async def generate_queries(db, state: ResearchState, llm) -> list[SearchQueryExecution]:
    """Generate search queries for the current round.

    Phase 2.2A Closure: Returns list[SearchQueryExecution].
    """
    from app.agent.steps.decompose_research_space import select_target_questions

    target_questions = select_target_questions(db, state.task_id, limit=3)

    if target_questions and state.pipeline_version >= 2:
        return await _generate_structured_queries(db, state, llm, target_questions)
    else:
        return await _generate_legacy_queries(db, state, llm)


async def _generate_structured_queries(
    db, state: ResearchState, llm, target_questions
) -> list[SearchQueryExecution]:
    """Generate structured queries with LLM-assigned target_question_id."""
    from app.db.models import ResearchContract

    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    preferred = json.loads(contract.preferred_directions_json or "[]") if contract else []
    excluded = json.loads(contract.excluded_directions_json or "[]") if contract else []
    key_terms = json.loads(contract.key_terms_json or "[]") if contract else []

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

    # Phase 2.2A Closure (#4): Proper retry — preserve valid, only re-request invalid
    min_valid_queries = min(3, settings.queries_per_round)
    valid_queries: list[GeneratedQuery] = []
    seen_texts: set[str] = set()

    for attempt in range(2):
        result = await llm.chat_json(messages, QueryList)

        new_valid = []
        for q in result.queries:
            norm = _normalize_query_text(q.query_text)
            if q.target_question_id in valid_question_ids and norm not in seen_texts:
                new_valid.append(q)
                seen_texts.add(norm)
            elif q.target_question_id not in valid_question_ids:
                logger.warning("Task %s: query '%s' has invalid target_question_id '%s'",
                             state.task_id[:8], q.query_text[:30], q.target_question_id[:12])

        valid_queries.extend(new_valid)

        if len(valid_queries) >= settings.queries_per_round:
            break  # Got enough

        if attempt == 0 and len(valid_queries) < min_valid_queries:
            logger.info("Task %s: retrying query generation (%d valid, need %d)",
                       state.task_id[:8], len(valid_queries), min_valid_queries)
            # Continue to retry
        else:
            break  # Good enough or last attempt

    if len(valid_queries) < min_valid_queries:
        logger.error("Task %s: only %d valid queries (minimum %d)",
                     state.task_id[:8], len(valid_queries), min_valid_queries)
        raise ValueError(
            f"Only {len(valid_queries)} valid queries generated (minimum {min_valid_queries})"
        )

    # Save each query as SearchQueryRecord
    executions: list[SearchQueryExecution] = []
    for q in valid_queries[:settings.queries_per_round]:
        record = save_search_query(
            db, state.task_id, q.query_text, q.intent,
            q.target_question_id,
            q.expected_evidence_type,
            state.current_round,
        )
        executions.append(SearchQueryExecution(
            query_id=record.id,
            query_text=q.query_text,
            intent=q.intent,
            target_question_id=q.target_question_id,
            expected_evidence_type=q.expected_evidence_type,
        ))

    # Update state.used_queries for backward compat
    state.used_queries.extend(eq.query_text for eq in executions)

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={
                              "queries": [eq.query_text for eq in executions],
                              "structured_queries": [asdict(eq) for eq in executions],
                              "target_question_ids": [q.id for q in target_questions],
                              "intent": "structured_llm_bound",
                              "valid_count": len(valid_queries),
                          })
    db.commit()
    return executions


async def _generate_legacy_queries(db, state: ResearchState, llm) -> list[SearchQueryExecution]:
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

    try:
        result = await llm.chat_json(messages, QueryList)
        executions = []
        for q in result.queries:
            record = save_search_query(db, state.task_id, q.query_text, "survey",
                                        None, None, state.current_round)
            executions.append(SearchQueryExecution(
                query_id=record.id, query_text=q.query_text,
                intent="survey", target_question_id="",
                expected_evidence_type=None,
            ))
        state.used_queries.extend(eq.query_text for eq in executions)
    except Exception:
        return []

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={"queries": [eq.query_text for eq in executions], "intent": "legacy"})
    db.commit()
    return executions


def _normalize_query_text(text: str) -> str:
    """Normalize query text for uniqueness checking."""
    return re.sub(r'\s+', ' ', (text or '').strip().lower())

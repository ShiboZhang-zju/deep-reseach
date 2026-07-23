"""Step: Generate search queries driven by Research Questions.

Phase 2.1 (#3): Each query is saved as a SearchQueryRecord with
target_question_id, intent, and expected_evidence_type.
"""

import json
import logging

from app.config import settings
from app.agent.state import ResearchState
from app.agent.prompts import QUERIES_SYSTEM, QUERIES_USER
from app.db.repositories import paper_repo
from app.db.repositories.search_query_repo import save_search_query
from app.schemas.schemas import QueryList
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def generate_queries(db, state: ResearchState, llm) -> list[str]:
    """Generate search queries for the current round.

    Phase 2.1: Saves SearchQueryRecord for each query with target_question_id.
    """
    from app.agent.steps.decompose_research_space import select_target_questions
    from app.db.models import ResearchContract

    target_questions = select_target_questions(db, state.task_id, limit=3)

    if target_questions:
        return await _generate_question_driven_queries(db, state, llm, target_questions)
    else:
        return await _generate_legacy_queries(db, state, llm)


async def _generate_question_driven_queries(
    db, state: ResearchState, llm, target_questions
) -> list[str]:
    """Generate queries targeted at specific Research Questions."""
    from app.db.models import ResearchContract

    contract = None
    if state.contract_id:
        contract = db.get(ResearchContract, state.contract_id)

    preferred = json.loads(contract.preferred_directions_json or "[]") if contract else []
    excluded = json.loads(contract.excluded_directions_json or "[]") if contract else []
    key_terms = json.loads(contract.key_terms_json or "[]") if contract else []

    question_contexts = []
    for q in target_questions:
        question_contexts.append(
            f"- Question [{q.id[:8]}] (type={q.question_type}, importance={q.importance:.1f}): {q.question}"
        )

    system_prompt = f"""You are a research search strategist. Generate {settings.queries_per_round} search queries targeting SPECIFIC Research Questions.

Each query must target one of the provided questions. Vary intent: some for seminal works, some for recent advances, some for benchmarks.

Guidelines:
- Each query should be a concise phrase (3-8 words) for academic search
- Include alternative terminology and synonyms
- Avoid repeating previously used queries
- Focus on finding high-quality, relevant papers

Preferred directions: {', '.join(preferred) if preferred else '(none)'}
Excluded directions: {', '.join(excluded) if excluded else '(none)'}
Key terms: {', '.join(key_terms) if key_terms else '(none)'}"""

    user_prompt = f"""Research topic: {state.normalized_topic}

Target Research Questions:
{chr(10).join(question_contexts)}

Previous queries used:
{chr(10).join(state.used_queries[-20:]) if state.used_queries else '(none)'}

User feedback: {state.user_feedback or '(none)'}

Generate {settings.queries_per_round} search queries. Output as a JSON list of query strings."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = await llm.chat_json(messages, QueryList)

    # Phase 2.1 (#3): Save each query as a SearchQueryRecord
    saved_queries = []
    for i, query_text in enumerate(result.queries):
        # Assign to questions in round-robin
        target_q = target_questions[i % len(target_questions)] if target_questions else None
        intent = _determine_intent(i, len(result.queries))
        evidence_type = _determine_evidence_type(target_q.question_type if target_q else None)

        record = save_search_query(
            db, state.task_id, query_text, intent,
            target_q.id if target_q else None,
            evidence_type,
            state.current_round,
        )
        saved_queries.append(record)

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={
                              "queries": result.queries,
                              "target_question_ids": [q.id for q in target_questions],
                              "saved_query_ids": [r.id for r in saved_queries],
                              "intent": "question_driven",
                          })
    db.commit()
    return result.queries


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
    result = await llm.chat_json(messages, QueryList)

    # Save without target question
    for query_text in result.queries:
        save_search_query(db, state.task_id, query_text, "survey",
                          None, None, state.current_round)

    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={"queries": result.queries, "intent": "legacy"})
    db.commit()
    return result.queries


def _determine_intent(index: int, total: int) -> str:
    """Determine query intent based on position in the batch."""
    if index == 0:
        return "seminal"
    elif index == 1:
        return "recent_work"
    elif index == 2:
        return "benchmark"
    elif index == 3:
        return "limitation"
    else:
        return "direct_neighbor"


def _determine_evidence_type(question_type: str | None) -> str | None:
    """Map question type to expected evidence type."""
    if not question_type:
        return None
    mapping = {
        "problem": "problem",
        "method": "method",
        "evaluation": "result",
        "dataset": "dataset",
        "resource": "limitation",
        "failure": "negative_result",
        "application": "comparison",
    }
    return mapping.get(question_type)

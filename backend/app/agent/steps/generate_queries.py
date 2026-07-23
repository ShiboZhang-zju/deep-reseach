"""Step: Generate search queries driven by Research Questions.

Phase 1.5 refactor:
- Queries are now targeted at specific Research Questions
- Each query has intent, target_question_id, expected_evidence_type
- Falls back to old behavior if no Research Questions exist
"""

import json
import logging

from app.config import settings
from app.agent.state import ResearchState
from app.agent.prompts import QUERIES_SYSTEM, QUERIES_USER
from app.db.repositories import paper_repo
from app.schemas.schemas import QueryList, GeneratedQuery
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def generate_queries(db, state: ResearchState, llm) -> list[str]:
    """Generate search queries for the current round.

    Phase 1.5: If Research Questions exist, queries are targeted at them.
    Returns list[str] for backward compat with search_and_save_papers.
    """
    from app.agent.steps.decompose_research_space import select_target_questions
    from app.db.models import ResearchContract

    # Try to get target questions
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

    # Get contract for context
    contract = None
    if state.contract_id:
        contract = db.get(ResearchContract, state.contract_id)

    # Build context for each question
    question_contexts = []
    for q in target_questions:
        question_contexts.append(
            f"- Question [{q.id[:8]}] (type={q.question_type}, importance={q.importance:.1f}): {q.question}"
        )

    # Get contract constraints
    preferred = json.loads(contract.preferred_directions_json or "[]") if contract else []
    excluded = json.loads(contract.excluded_directions_json or "[]") if contract else []
    key_terms = json.loads(contract.key_terms_json or "[]") if contract else []

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

    # Save trace with target question info
    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={
                              "queries": result.queries,
                              "target_question_ids": [q.id for q in target_questions],
                              "target_questions": [
                                  {"id": q.id, "question": q.question[:80], "type": q.question_type}
                                  for q in target_questions
                              ],
                              "intent": "question_driven",
                          })
    db.commit()
    return result.queries


async def _generate_legacy_queries(db, state: ResearchState, llm) -> list[str]:
    """Legacy query generation (fallback when no Research Questions exist)."""
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
    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={"queries": result.queries, "intent": "legacy"})
    db.commit()
    return result.queries

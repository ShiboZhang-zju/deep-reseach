"""Step: Decompose research space into structured questions.

Phase 1.5 fixes:
- Only reuse questions belonging to the current contract_id (not just task_id)
- Supersede old questions when contract changes
- Persist state.active_question_ids
"""

import json
import logging

from app.agent.state import ResearchState
from app.agent.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_USER
from app.db.models import ResearchContract, ResearchQuestion
from app.db.repositories import paper_repo, task_repo
from app.schemas.schemas import ResearchDecompositionSchema

logger = logging.getLogger(__name__)


async def decompose_research_space(db, state: ResearchState, llm, task_id: str) -> list[ResearchQuestion]:
    """Decompose the research contract into structured research questions.

    Phase 1.5: Only reuses questions from the current active contract.
    """
    if not state.contract_id:
        logger.warning("Task %s: no contract_id in state, using fallback", task_id[:8])
        return await _decompose_fallback(db, state, llm, task_id)

    contract = db.get(ResearchContract, state.contract_id)
    if not contract:
        logger.warning("Task %s: contract %s not found, using fallback", task_id[:8], state.contract_id[:8])
        return await _decompose_fallback(db, state, llm, task_id)

    # Phase 1.5: Only reuse active questions from THIS contract
    existing_qs = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract.id,
        ResearchQuestion.status != "superseded",
    ).all()

    if existing_qs:
        logger.info("Task %s: %d active questions for contract v%d, skipping",
                    task_id[:8], len(existing_qs), contract.version)
        state.active_question_ids = [q.id for q in existing_qs]
        task_repo.save_state(db, task_id, state)
        db.commit()
        return existing_qs

    # Build prompt
    user_content = DECOMPOSE_USER.format(
        topic=contract.topic,
        target_problem=contract.target_problem or "(not specified)",
        target_setting=contract.target_setting or "(not specified)",
        desired_output=contract.desired_output or "method",
        preferred_directions=json.loads(contract.preferred_directions_json or "[]"),
        excluded_directions=json.loads(contract.excluded_directions_json or "[]"),
        key_terms=json.loads(contract.key_terms_json or "[]"),
    )

    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    result = await llm.chat_json(messages, ResearchDecompositionSchema)

    # Save questions to database
    saved_questions = []
    for q in result.questions:
        rq = ResearchQuestion(
            task_id=task_id,
            contract_id=contract.id,
            question=q.question,
            question_type=q.question_type,
            importance=q.importance,
            searchability=q.searchability,
            status="open",
            axis_name=q.axis_name,
            version=contract.version,
        )
        db.add(rq)
        saved_questions.append(rq)

    db.flush()

    # Phase 1.5: Persist active_question_ids to state
    state.active_question_ids = [q.id for q in saved_questions]
    # Also update deprecated research_questions for backward compat
    state.research_questions = [q.question for q in result.questions]
    task_repo.save_state(db, task_id, state)

    paper_repo.save_trace(db, task_id, "decompose_research_space", "action",
                          output_data={
                              "contract_id": contract.id,
                              "contract_version": contract.version,
                              "question_count": len(result.questions),
                              "axes": [a.axis_name for a in result.axes],
                              "question_types": [q.question_type for q in result.questions],
                          })
    db.commit()

    logger.info("Task %s: decomposed into %d questions for contract v%d",
               task_id[:8], len(result.questions), contract.version)
    return saved_questions


async def _decompose_fallback(db, state: ResearchState, llm, task_id: str) -> list[ResearchQuestion]:
    """Fallback when no contract exists — uses normalized_topic directly."""
    user_content = DECOMPOSE_USER.format(
        topic=state.normalized_topic or state.user_input,
        target_problem="(not specified)",
        target_setting="(not specified)",
        desired_output="method",
        preferred_directions=[],
        excluded_directions=[],
        key_terms=state.keywords,
    )

    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    result = await llm.chat_json(messages, ResearchDecompositionSchema)

    saved_questions = []
    for q in result.questions:
        rq = ResearchQuestion(
            task_id=task_id,
            question=q.question,
            question_type=q.question_type,
            importance=q.importance,
            searchability=q.searchability,
            status="open",
            axis_name=q.axis_name,
        )
        db.add(rq)
        saved_questions.append(rq)

    db.flush()
    state.active_question_ids = [q.id for q in saved_questions]
    state.research_questions = [q.question for q in result.questions]
    task_repo.save_state(db, task_id, state)

    paper_repo.save_trace(db, task_id, "decompose_research_space", "action",
                          output_data={"question_count": len(result.questions), "fallback": True})
    db.commit()

    return saved_questions


def select_target_questions(db, task_id: str, limit: int = 3) -> list[ResearchQuestion]:
    """Select research questions to target in the next search round.

    Phase 1.5: Selection criteria:
    1. status == 'open' or 'partially_covered' (not covered/unavailable/superseded)
    2. Sort by importance × searchability (higher first)
    3. When CoverageRecords exist, prefer low-coverage questions
    """
    from app.db.models import CoverageRecord

    # Get active questions
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.status.in_(["open", "partially_covered"]),
    ).all()

    if not questions:
        return []

    # Try to get coverage records
    coverage_map = {}
    for cr in db.query(CoverageRecord).filter(
        CoverageRecord.task_id == task_id,
    ).all():
        coverage_map[cr.question_id] = cr

    # Sort by: (1 - coverage_score) × importance × searchability
    def score_q(q):
        cr = coverage_map.get(q.id)
        coverage = cr.coverage_score if cr else 0.0
        return (1.0 - coverage) * q.importance * q.searchability

    questions.sort(key=score_q, reverse=True)
    return questions[:limit]

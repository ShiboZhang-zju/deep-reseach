"""Step: Decompose research space into structured questions.

Phase 1: Takes a ResearchContract and produces 5-12 specific, searchable,
answerable Research Questions organized by research axes.

This replaces the old `knowledge_gaps: list[str]` approach with structured
questions that can be tracked, covered, and used to drive targeted search.
"""

import json
import logging

from app.agent.state import ResearchState
from app.agent.prompts import (
    DECOMPOSE_SYSTEM, DECOMPOSE_USER,
)
from app.db.models import ResearchContract, ResearchQuestion
from app.db.repositories import paper_repo
from app.schemas.schemas import ResearchDecompositionSchema

logger = logging.getLogger(__name__)


async def decompose_research_space(db, state: ResearchState, llm, task_id: str) -> list[ResearchQuestion]:
    """Decompose the research contract into structured research questions.

    Produces 5-12 questions organized by research axes:
    - problem axis: what problems exist?
    - method axis: what methods are used?
    - evaluation axis: how are methods evaluated?
    - dataset axis: what datasets are available?
    - resource axis: what resources are needed?
    - failure axis: what failure modes exist?
    - application axis: what applications exist?
    """
    # Get the contract
    contract = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()

    if not contract:
        # Fallback: use normalized_topic as a minimal contract
        logger.warning("Task %s: no contract found, using normalized_topic as fallback", task_id[:8])
        return await _decompose_fallback(db, state, llm, task_id)

    # Check if questions already exist
    existing_qs = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
    ).count()
    if existing_qs > 0:
        logger.info("Task %s: %d research questions already exist, skipping", task_id[:8], existing_qs)
        return db.query(ResearchQuestion).filter(
            ResearchQuestion.task_id == task_id,
        ).all()

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
        )
        db.add(rq)
        saved_questions.append(rq)

    db.flush()

    # Update state with question IDs for backward compatibility
    state.research_questions = [q.question for q in result.questions]

    paper_repo.save_trace(db, task_id, "decompose_research_space", "action",
                          output_data={
                              "question_count": len(result.questions),
                              "axes": [a.axis_name for a in result.axes],
                              "question_types": [q.question_type for q in result.questions],
                          })
    db.commit()

    logger.info("Task %s: decomposed into %d research questions across %d axes",
               task_id[:8], len(result.questions), len(result.axes))
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
    state.research_questions = [q.question for q in result.questions]

    paper_repo.save_trace(db, task_id, "decompose_research_space", "action",
                          output_data={"question_count": len(result.questions), "fallback": True})
    db.commit()

    return saved_questions

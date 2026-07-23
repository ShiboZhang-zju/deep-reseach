"""Step: Build structured Research Contract from user input + clarifications.

Phase 2.2A:
- Single compute_contract_input_version() function with all 7 components
- Proper Contract invalidation (supersede old + questions)
- state.active_question_ids cleared on new Contract
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

from app.agent.state import ResearchState
from app.agent.prompts import BUILD_CONTRACT_SYSTEM, BUILD_CONTRACT_USER
from app.db.models import ResearchContract, ResearchTask, ResearchQuestion, UserFeedback
from app.db.repositories import paper_repo, task_repo
from app.schemas.schemas import ResearchContractSchema

logger = logging.getLogger(__name__)


def _normalize_str(s: str) -> str:
    """Normalize string for stable hashing."""
    return (s or "").strip()


def compute_contract_input_version(db, task: ResearchTask, state: ResearchState) -> str:
    """Compute a stable SHA-256 hash of ALL inputs that affect the Contract.

    Includes (in stable sorted order):
    1. task.user_input (original input)
    2. state.user_input (with clarifications appended)
    3. state.user_feedback
    4. state.clarification_questions (JSON, sorted keys)
    5. All clarification_answer feedback records (JSON content)
    6. All research_feedback records (JSON content)
    7. state.pipeline_version
    """
    # Gather clarification_answer feedback
    clarification_answers = []
    research_feedbacks = []
    feedbacks = db.query(UserFeedback).filter(
        UserFeedback.task_id == task.id,
    ).order_by(UserFeedback.created_at).all()
    for fb in feedbacks:
        if fb.feedback_type == "clarification_answer":
            clarification_answers.append(fb.content or "")
        elif fb.feedback_type == "research_feedback":
            research_feedbacks.append(fb.content or "")

    # Build stable JSON representation
    components = {
        "task_user_input": _normalize_str(task.user_input),
        "state_user_input": _normalize_str(state.user_input),
        "state_user_feedback": _normalize_str(state.user_feedback),
        "clarification_questions": state.clarification_questions or [],
        "clarification_answers": clarification_answers,
        "research_feedbacks": research_feedbacks,
        "pipeline_version": state.pipeline_version,
    }

    # Stable serialization: sort_keys=True, fixed separators
    combined = json.dumps(components, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# Backward compat alias
def compute_input_hash(task: ResearchTask, state: ResearchState) -> str:
    """Deprecated: use compute_contract_input_version instead."""
    # This is a fallback that doesn't have DB access — only includes basic fields
    parts = [
        _normalize_str(task.user_input),
        _normalize_str(state.user_feedback),
        json.dumps(state.clarification_questions or [], sort_keys=True, ensure_ascii=False),
    ]
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


async def build_research_contract(db, state: ResearchState, llm, task_id: str) -> ResearchContract:
    """Build a structured Research Contract from user input and clarifications.

    Phase 2.2A: Uses compute_contract_input_version() for proper invalidation.
    """
    task = db.get(ResearchTask, task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    current_hash = compute_contract_input_version(db, task, state)

    # Check for existing active contract
    existing = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()

    if existing and existing.input_hash == current_hash:
        logger.info("Task %s: contract v%d reuse (hash=%s)", task_id[:8], existing.version, current_hash[:12])
        state.contract_id = existing.id
        state.normalized_topic = existing.topic
        state.keywords = json.loads(existing.key_terms_json or "[]")
        task_repo.save_state(db, task_id, state)
        db.commit()
        return existing

    # Input changed or no contract — supersede old ones
    if existing:
        existing.status = "superseded"
        existing.superseded_at = datetime.now(timezone.utc).replace(tzinfo=None)
        # Supersede old questions belonging to THIS contract only
        old_questions = db.query(ResearchQuestion).filter(
            ResearchQuestion.contract_id == existing.id,
            ResearchQuestion.status != "superseded",
        ).all()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for q in old_questions:
            q.status = "superseded"
            q.superseded_at = now
        db.flush()
        logger.info("Task %s: superseded contract v%d and %d questions",
                    task_id[:8], existing.version, len(old_questions))

    # Build new contract via LLM
    user_content = BUILD_CONTRACT_USER.format(
        user_input=state.user_input,
        previous_topic=state.normalized_topic or "(none)",
        keywords=", ".join(state.keywords) if state.keywords else "(none)",
    )

    messages = [
        {"role": "system", "content": BUILD_CONTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    result = await llm.chat_json(messages, ResearchContractSchema)

    new_version = (existing.version + 1) if existing else 1

    contract = ResearchContract(
        task_id=task_id,
        topic=result.topic,
        target_problem=result.target_problem,
        target_setting=result.target_setting,
        desired_output=result.desired_output,
        novelty_bar=result.novelty_bar,
        preferred_directions_json=json.dumps(result.preferred_directions, sort_keys=True, ensure_ascii=False),
        excluded_directions_json=json.dumps(result.excluded_directions, sort_keys=True, ensure_ascii=False),
        gpu_available=result.gpu_available,
        max_gpu_hours=result.max_gpu_hours,
        max_api_budget=result.max_api_budget,
        max_runtime_minutes=result.max_runtime_minutes,
        allow_large_benchmark=result.allow_large_benchmark,
        allow_model_training=result.allow_model_training,
        experiment_preferences_json=json.dumps(result.experiment_preferences, sort_keys=True, ensure_ascii=False),
        key_terms_json=json.dumps(result.key_terms, sort_keys=True, ensure_ascii=False),
        time_scope_start=result.time_scope_start,
        time_scope_end=result.time_scope_end,
        confidence=result.confidence,
        status="active",
        version=new_version,
        input_hash=current_hash,
    )
    db.add(contract)
    db.flush()

    # Phase 2.2A: Clear active_question_ids on new Contract
    state.contract_id = contract.id
    state.active_question_ids = []
    state.normalized_topic = result.topic
    state.keywords = result.key_terms
    task_repo.update_normalized_topic(db, task_id, result.topic)
    task_repo.save_state(db, task_id, state)

    paper_repo.save_trace(db, task_id, "build_research_contract", "action",
                          output_data={
                              "contract_id": contract.id,
                              "version": new_version,
                              "input_hash": current_hash,
                              "topic": result.topic,
                              "desired_output": result.desired_output,
                              "novelty_bar": result.novelty_bar,
                              "num_key_terms": len(result.key_terms),
                          })
    db.commit()

    logger.info("Task %s: research contract v%d built (topic=%s, confidence=%.2f)",
               task_id[:8], new_version, result.topic[:50], result.confidence)
    return contract

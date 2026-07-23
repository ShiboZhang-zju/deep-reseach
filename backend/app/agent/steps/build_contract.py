"""Step: Build structured Research Contract from user input + clarifications.

Phase 1.5 fixes:
- Persist state.contract_id via task_repo.save_state()
- Add version/input_hash for contract versioning
- Supersede old contracts when input changes
- Remove meaningless state.user_input = state.user_input
"""

import hashlib
import json
import logging

from app.agent.state import ResearchState
from app.agent.prompts import BUILD_CONTRACT_SYSTEM, BUILD_CONTRACT_USER
from app.db.models import ResearchContract, ResearchTask, UserFeedback
from app.db.repositories import paper_repo, task_repo
from app.schemas.schemas import ResearchContractSchema

logger = logging.getLogger(__name__)


def compute_input_hash(task: ResearchTask, state: ResearchState) -> str:
    """Compute a stable SHA-256 hash of all inputs that affect the Contract.

    Includes:
    - task.user_input (original + clarifications)
    - state.user_feedback
    - state.clarification_questions (Phase 2.1: #14)
    """
    parts = [
        task.user_input or "",
        state.user_feedback or "",
        json.dumps(state.clarification_questions or [], ensure_ascii=False),
    ]
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


async def build_research_contract(db, state: ResearchState, llm, task_id: str) -> ResearchContract:
    """Build a structured Research Contract from user input and clarifications.

    Phase 1.5: Implements versioning — if input_hash changed, supersede old contract.
    """
    task = db.get(ResearchTask, task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    current_hash = compute_input_hash(task, state)

    # Check for existing active contract
    existing = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()

    if existing and existing.input_hash == current_hash:
        logger.info("Task %s: contract v%d reuse (hash=%s)", task_id[:8], existing.version, current_hash)
        # Persist contract_id to state
        state.contract_id = existing.id
        state.normalized_topic = existing.topic
        state.keywords = json.loads(existing.key_terms_json or "[]")
        task_repo.save_state(db, task_id, state)
        db.commit()
        return existing

    # Input changed or no contract — supersede old ones
    if existing:
        from app.db.models import ResearchQuestion
        existing.status = "superseded"
        existing.superseded_at = datetime_now()
        # Also supersede old questions
        old_questions = db.query(ResearchQuestion).filter(
            ResearchQuestion.contract_id == existing.id,
            ResearchQuestion.status != "superseded",
        ).all()
        for q in old_questions:
            q.status = "superseded"
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
        preferred_directions_json=json.dumps(result.preferred_directions, ensure_ascii=False),
        excluded_directions_json=json.dumps(result.excluded_directions, ensure_ascii=False),
        gpu_available=result.gpu_available,
        max_gpu_hours=result.max_gpu_hours,
        max_api_budget=result.max_api_budget,
        max_runtime_minutes=result.max_runtime_minutes,
        allow_large_benchmark=result.allow_large_benchmark,
        allow_model_training=result.allow_model_training,
        experiment_preferences_json=json.dumps(result.experiment_preferences, ensure_ascii=False),
        key_terms_json=json.dumps(result.key_terms, ensure_ascii=False),
        time_scope_start=result.time_scope_start,
        time_scope_end=result.time_scope_end,
        confidence=result.confidence,
        status="active",
        version=new_version,
        input_hash=current_hash,
    )
    db.add(contract)
    db.flush()

    # Phase 1.5: Persist contract info to state
    state.contract_id = contract.id
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


def datetime_now():
    """Helper to get UTC now (avoids circular import with models)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)

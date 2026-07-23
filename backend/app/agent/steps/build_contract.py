"""Step: Build structured Research Contract from user input + clarifications.

Phase 1: Replaces the old `normalized_topic` + `keywords` approach with a
rich structured contract that captures user intent, constraints, and preferences.

The contract is the source of truth for all downstream steps — queries,
evidence extraction, gap mining, and idea synthesis all reference the contract.
"""

import json
import logging

from app.agent.state import ResearchState
from app.agent.prompts import BUILD_CONTRACT_SYSTEM, BUILD_CONTRACT_USER
from app.db.models import ResearchContract
from app.db.repositories import paper_repo
from app.schemas.schemas import ResearchContractSchema

logger = logging.getLogger(__name__)


async def build_research_contract(db, state: ResearchState, llm, task_id: str) -> ResearchContract:
    """Build a structured Research Contract from user input and clarifications.

    This replaces the old clarify_topic → normalized_topic flow.
    The user's original input + any clarification answers are compiled
    into a structured contract by the LLM.
    """
    # Check if contract already exists for this task
    existing = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()
    if existing:
        logger.info("Task %s: contract already exists (id=%s), skipping", task_id[:8], existing.id[:8])
        return existing

    # Build the user prompt from state
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

    # Save to database
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
        key_terms_json=json.dumps(result.key_terms, ensure_ascii=False),
        time_scope_start=result.time_scope_start,
        time_scope_end=result.time_scope_end,
        confidence=result.confidence,
        status="active",
    )
    db.add(contract)
    db.flush()

    # Update state with contract info (backward compatibility)
    state.normalized_topic = result.topic
    state.keywords = result.key_terms
    state.user_input = state.user_input  # preserve original input

    paper_repo.save_trace(db, task_id, "build_research_contract", "action",
                          output_data={
                              "contract_id": contract.id,
                              "topic": result.topic,
                              "desired_output": result.desired_output,
                              "novelty_bar": result.novelty_bar,
                              "num_key_terms": len(result.key_terms),
                              "num_preferred": len(result.preferred_directions),
                              "num_excluded": len(result.excluded_directions),
                          })
    db.commit()

    logger.info("Task %s: research contract built (topic=%s, confidence=%.2f)",
               task_id[:8], result.topic[:50], result.confidence)
    return contract

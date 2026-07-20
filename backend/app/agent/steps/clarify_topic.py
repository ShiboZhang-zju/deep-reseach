"""Step: Topic clarification."""

from app.agent.state import ResearchState
from app.agent.prompts import CLARIFY_SYSTEM, CLARIFY_USER
from app.db.repositories import paper_repo
from app.schemas.schemas import ClarityResult


async def clarify_topic(db, state: ResearchState, llm) -> ClarityResult:
    """Ask LLM whether the research direction is clear enough."""
    messages = [
        {"role": "system", "content": CLARIFY_SYSTEM},
        {"role": "user", "content": CLARIFY_USER.format(user_input=state.user_input)},
    ]
    result = await llm.chat_json(messages, ClarityResult)
    paper_repo.save_trace(db, state.task_id, "clarify_topic", "action",
                          output_data=result.model_dump())
    db.commit()
    return result

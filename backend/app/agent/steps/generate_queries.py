"""Step: Generate search queries for the current round."""

from app.config import settings
from app.agent.state import ResearchState
from app.agent.prompts import QUERIES_SYSTEM, QUERIES_USER
from app.db.repositories import paper_repo
from app.schemas.schemas import QueryList


async def generate_queries(db, state: ResearchState, llm) -> list[str]:
    """Generate search queries based on topic, keywords, used queries, and gaps."""
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
                          output_data={"queries": result.queries})
    db.commit()
    return result.queries

"""Step: Generate round summary and knowledge gaps."""

from app.agent.state import ResearchState
from app.agent.prompts import ROUND_SUMMARY_SYSTEM, ROUND_SUMMARY_USER
from app.db.repositories import paper_repo
from app.schemas.schemas import RoundSummary


async def summarize_round(db, state: ResearchState, llm, round_num: int, scored_papers: list):
    """Summarize the round and identify knowledge gaps."""
    papers_text = "\n".join(
        f"- {p['title']} (score: {p['score']:.2f}, {p['priority']}): {p['summary']}"
        for p in scored_papers[:30]
    )
    messages = [
        {"role": "system", "content": ROUND_SUMMARY_SYSTEM},
        {"role": "user", "content": ROUND_SUMMARY_USER.format(
            topic=state.normalized_topic,
            round_num=round_num,
            papers_summary=papers_text or "(no papers)",
            previous_gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
        )},
    ]
    result = await llm.chat_json(messages, RoundSummary)
    paper_repo.save_trace(db, state.task_id, "summarize_round", "action",
                          round_number=round_num,
                          output_data=result.model_dump())
    db.commit()
    return result.summary, result.knowledge_gaps

"""Step: Generate experiment plans for selected ideas."""

import json
import logging

from app.agent.state import ResearchState
from app.agent.prompts import EXPERIMENT_SYSTEM, EXPERIMENT_USER
from app.db.models import ResearchIdea
from app.db.repositories import paper_repo
from app.schemas.schemas import ExperimentPlanSchema
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def score_idea(db, state: ResearchState, llm, idea) -> "IdeaScore":
    """Score a single idea (re-exported from generate_ideas for deep evaluation)."""
    from app.agent.steps.generate_ideas import _score_idea
    return await _score_idea(db, state, llm, idea)


async def generate_experiments(db, state: ResearchState, llm, task_id: str, idea_ids: list[str]):
    """Deep evaluate selected ideas and generate experiment plans."""
    # Deep evaluate selected ideas
    ideas = db.query(ResearchIdea).filter(ResearchIdea.id.in_(idea_ids)).all()

    good_ideas = []
    for idea in ideas:
        try:
            scores = await score_idea(db, state, llm, idea)
        except Exception as e:
            logger.error("Failed to deep-score idea %s: %s", idea.id, e)
            if idea.final_score and idea.final_score >= 0.75:
                good_ideas.append(idea)
            continue

        idea_score = (
            0.20 * scores.novelty + 0.20 * scores.feasibility + 0.20 * scores.significance +
            0.20 * scores.evidence_support + 0.10 * scores.differentiation +
            0.05 * scores.experimentability + 0.05 * scores.potential_impact
        )
        final_score = idea_score - 0.08 * scores.risk

        if final_score >= 0.70:
            decision = "go"
        elif final_score >= 0.50:
            decision = "revise"
        else:
            decision = "reject"

        paper_repo.update_idea_scores(db, idea.id, scores.model_dump(), final_score, decision)
        idea.user_selected = True
        db.flush()
        db.commit()

        # Phase 0 fix: conditional_go is no longer auto-promoted.
        # Only genuinely "go" ideas proceed to experiment generation.
        if decision == "go":
            good_ideas.append(idea)
        logger.info("Idea '%s' deep-scored: %.3f -> %s", idea.title[:40], final_score, decision)

    emit_event(task_id, "ideas_judged", {"total": len(ideas), "go": len(good_ideas)})

    if not good_ideas:
        return {"status": "need_more_research", "reason": "No selected idea is ready for experiment."}

    # Generate experiment plans
    logger.info("Task %s: generating experiments for %d ideas...", task_id[:8], len(good_ideas))

    # Get wiki context for experiment generation
    wiki_ctx = ""
    try:
        from app.services.wiki_service import get_wiki_context
        wiki_ctx = get_wiki_context(db, task_id, page_types=["method", "dataset", "model"], max_chars=8000)
        if wiki_ctx:
            logger.info("Task %s: wiki context loaded for experiments (%d chars)", task_id[:8], len(wiki_ctx))
    except Exception as e:
        logger.warning("Task %s: wiki context for experiments failed (non-fatal): %s", task_id[:8], e)

    for idea in good_ideas:
        messages = [
            {"role": "system", "content": EXPERIMENT_SYSTEM},
            {"role": "user", "content": EXPERIMENT_USER.format(
                topic=state.normalized_topic,
                title=idea.title or "",
                description=idea.description or "",
                method=idea.method_sketch or "",
                contribution=idea.expected_contribution or "",
                related_papers=idea.related_paper_ids_json or "",
                wiki_context=wiki_ctx or "(no wiki available)",
            )},
        ]
        try:
            plan = await llm.chat_json(messages, ExperimentPlanSchema)
        except Exception as e:
            logger.error("Failed to generate experiment for idea %s: %s", idea.id, e)
            continue

        plan_data = {
            "hypothesis": plan.hypothesis,
            "dataset": plan.dataset,
            "baselines": plan.baselines,
            "metrics": plan.metrics,
            "steps_markdown": "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps)),
            "steps_json": json.dumps(plan.steps, ensure_ascii=False),
            "risks": plan.risks,
        }
        paper_repo.save_experiment(db, task_id, idea.id, plan_data)
        db.commit()
        emit_event(task_id, "experiment_generated", {"idea_id": idea.id, "title": idea.title})
        logger.info("Experiment generated for idea '%s'", idea.title[:40])

    return {"status": "done", "experiments_generated": len(good_ideas)}

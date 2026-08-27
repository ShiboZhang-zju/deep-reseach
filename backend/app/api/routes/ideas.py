"""Idea API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import ResearchIdea, isoformat_utc
from app.schemas.schemas import IdeaOut

router = APIRouter()


@router.get("/tasks/{task_id}/ideas", response_model=list[IdeaOut])
def get_ideas(task_id: str, include_superseded: bool = False, db: Session = Depends(get_db_session)):
    """Get ideas for a task.

    P1-5: By default only returns 'active' ideas (excludes soft-deleted 'superseded').
    Set include_superseded=true to see all historical ideas.
    """
    query = db.query(ResearchIdea).filter(ResearchIdea.task_id == task_id)
    if not include_superseded:
        query = query.filter(ResearchIdea.idea_status == "active")
    ideas = query.order_by(ResearchIdea.final_score.desc().nullslast()).all()

    return [_to_idea_out(i) for i in ideas]


def _to_idea_out(idea: ResearchIdea) -> IdeaOut:
    return IdeaOut(
        id=idea.id,
        task_id=idea.task_id,
        title=idea.title,
        description=idea.description,
        motivation=idea.motivation,
        method_sketch=idea.method_sketch,
        expected_contribution=idea.expected_contribution,
        novelty=idea.novelty,
        feasibility=idea.feasibility,
        significance=idea.significance,
        evidence_support=idea.evidence_support,
        differentiation=idea.differentiation,
        experimentability=idea.experimentability,
        potential_impact=idea.potential_impact,
        risk=idea.risk,
        final_score=idea.final_score,
        decision=idea.decision,
        score_status=idea.score_status or ("scored" if idea.final_score is not None else "unscored"),
        score_error=idea.score_error,
        related_paper_ids_json=idea.related_paper_ids_json,
        quality_reason_codes_json=idea.quality_reason_codes_json,
        confidence_tier=idea.confidence_tier,
        user_selected=idea.user_selected,
        idea_status=idea.idea_status or "active",
        created_at=isoformat_utc(idea.created_at),
    )

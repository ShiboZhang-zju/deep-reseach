"""Idea API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import ResearchIdea
from app.schemas.schemas import IdeaOut

router = APIRouter()


@router.get("/tasks/{task_id}/ideas", response_model=list[IdeaOut])
def get_ideas(task_id: str, db: Session = Depends(get_db_session)):
    ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task_id
    ).order_by(ResearchIdea.final_score.desc().nullslast()).all()

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
        related_paper_ids_json=idea.related_paper_ids_json,
        user_selected=idea.user_selected,
        created_at=idea.created_at.isoformat() if idea.created_at else "",
    )

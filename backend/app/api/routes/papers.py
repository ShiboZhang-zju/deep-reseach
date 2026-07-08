"""Paper API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.deps import get_db_session
from app.db.models import Paper, TaskPaper, ResearchRound
from app.schemas.schemas import PaperOut, RoundOut

router = APIRouter()


@router.get("/tasks/{task_id}/papers", response_model=list[PaperOut])
def get_papers(
    task_id: str,
    priority: str | None = Query(None, description="Filter by: high/medium/low"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db_session),
):
    query = (
        db.query(Paper, TaskPaper)
        .join(TaskPaper, TaskPaper.paper_id == Paper.id)
        .filter(TaskPaper.task_id == task_id)
    )
    if priority:
        query = query.filter(TaskPaper.priority == priority)

    results = query.order_by(TaskPaper.final_score.desc().nullslast()).offset(offset).limit(limit).all()

    return [
        PaperOut(
            id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            authors_json=paper.authors_json,
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            url=paper.url,
            citation_count=paper.citation_count or 0,
            sources_json=paper.sources_json,
            final_score=tp.final_score,
            priority=tp.priority,
            reason=tp.reason,
            summary=tp.summary,
        )
        for paper, tp in results
    ]


@router.get("/tasks/{task_id}/rounds", response_model=list[RoundOut])
def get_rounds(task_id: str, db: Session = Depends(get_db_session)):
    rounds = db.query(ResearchRound).filter(
        ResearchRound.task_id == task_id
    ).order_by(ResearchRound.round_number).all()

    return [
        RoundOut(
            id=r.id,
            round_number=r.round_number,
            queries_json=r.queries_json,
            papers_found=r.papers_found,
            new_papers=r.new_papers,
            duplicate_rate=r.duplicate_rate,
            summary=r.summary,
            knowledge_gaps_json=r.knowledge_gaps_json,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rounds
    ]

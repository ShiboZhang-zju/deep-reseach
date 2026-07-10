"""Paper API routes."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.deps import get_db_session
from app.db.models import Paper, TaskPaper, ResearchRound, WikiPage, isoformat_utc
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
            created_at=isoformat_utc(r.created_at),
        )
        for r in rounds
    ]


@router.get("/tasks/{task_id}/wiki")
def get_wiki_pages(
    task_id: str,
    page_type: str | None = Query(None, description="Filter by: concept/method/dataset/model/synthesis"),
    db: Session = Depends(get_db_session),
):
    """Get wiki pages for a task (LLM Wiki knowledge base)."""
    query = db.query(WikiPage).filter(WikiPage.task_id == task_id)
    if page_type:
        query = query.filter(WikiPage.page_type == page_type)
    else:
        query = query.filter(WikiPage.page_type != "index")

    pages = query.order_by(WikiPage.page_type, WikiPage.title).all()

    return [
        {
            "id": p.id,
            "page_type": p.page_type,
            "title": p.title,
            "content_markdown": p.content_markdown,
            "paper_ids": json.loads(p.paper_ids_json or "[]"),
            "links": json.loads(p.links_json or "[]"),
            "contradictions": json.loads(p.contradictions_json or "[]"),
            "created_at": isoformat_utc(p.created_at),
            "updated_at": isoformat_utc(p.updated_at),
        }
        for p in pages
    ]


@router.get("/tasks/{task_id}/wiki/stats")
def get_wiki_stats_api(task_id: str, db: Session = Depends(get_db_session)):
    """Get wiki statistics for a task."""
    from app.services.wiki_service import get_wiki_stats
    return get_wiki_stats(db, task_id)

"""Report API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import Report
from app.schemas.schemas import ReportOut

router = APIRouter()


@router.get("/tasks/{task_id}/report", response_model=ReportOut)
def get_report(task_id: str, db: Session = Depends(get_db_session)):
    report = db.query(Report).filter(
        Report.task_id == task_id
    ).order_by(Report.created_at.desc()).first()

    if not report:
        return ReportOut(
            id="",
            task_id=task_id,
            content_markdown="",
            content_json=None,
            created_at="",
        )

    return ReportOut(
        id=report.id,
        task_id=report.task_id,
        content_markdown=report.content_markdown or "",
        content_json=report.content_json,
        created_at=report.created_at.isoformat() if report.created_at else "",
    )

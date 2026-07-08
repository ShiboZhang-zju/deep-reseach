"""Agent trace API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import AgentTrace
from app.schemas.schemas import TraceOut

router = APIRouter()


@router.get("/tasks/{task_id}/traces", response_model=list[TraceOut])
def get_traces(task_id: str, db: Session = Depends(get_db_session)):
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id
    ).order_by(AgentTrace.created_at).all()

    return [
        TraceOut(
            id=t.id,
            step_name=t.step_name,
            step_type=t.step_type,
            round_number=t.round_number,
            input_json=t.input_json,
            output_json=t.output_json,
            llm_tokens_used=t.llm_tokens_used,
            duration_ms=t.duration_ms,
            created_at=t.created_at.isoformat() if t.created_at else "",
        )
        for t in traces
    ]

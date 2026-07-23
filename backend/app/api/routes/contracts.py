"""Contract and Research Question API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import ResearchContract, ResearchQuestion, isoformat_utc
from app.schemas.schemas import ContractOut, ResearchQuestionOut

router = APIRouter()


@router.get("/tasks/{task_id}/contract", response_model=ContractOut | None)
def get_contract(task_id: str, db: Session = Depends(get_db_session)):
    """Get the research contract for a task."""
    contract = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()
    if not contract:
        return None
    return _to_contract_out(contract)


@router.get("/tasks/{task_id}/questions", response_model=list[ResearchQuestionOut])
def get_questions(task_id: str, db: Session = Depends(get_db_session)):
    """Get research questions for a task."""
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
    ).order_by(ResearchQuestion.importance.desc()).all()
    return [_to_question_out(q) for q in questions]


def _to_contract_out(contract: ResearchContract) -> ContractOut:
    return ContractOut(
        id=contract.id,
        task_id=contract.task_id,
        topic=contract.topic,
        target_problem=contract.target_problem,
        target_setting=contract.target_setting,
        desired_output=contract.desired_output,
        novelty_bar=contract.novelty_bar,
        preferred_directions_json=contract.preferred_directions_json,
        excluded_directions_json=contract.excluded_directions_json,
        gpu_available=contract.gpu_available,
        max_gpu_hours=contract.max_gpu_hours,
        max_api_budget=contract.max_api_budget,
        max_runtime_minutes=contract.max_runtime_minutes,
        allow_large_benchmark=contract.allow_large_benchmark,
        allow_model_training=contract.allow_model_training,
        key_terms_json=contract.key_terms_json,
        time_scope_start=contract.time_scope_start,
        time_scope_end=contract.time_scope_end,
        status=contract.status,
        confidence=contract.confidence,
        created_at=isoformat_utc(contract.created_at),
        updated_at=isoformat_utc(contract.updated_at),
    )


def _to_question_out(q: ResearchQuestion) -> ResearchQuestionOut:
    return ResearchQuestionOut(
        id=q.id,
        task_id=q.task_id,
        contract_id=q.contract_id,
        question=q.question,
        question_type=q.question_type,
        importance=q.importance,
        searchability=q.searchability,
        status=q.status,
        axis_name=q.axis_name,
        created_at=isoformat_utc(q.created_at),
    )

"""Contract, Questions, Evidence, Coverage, and Phase API routes."""

import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import (
    ResearchContract, ResearchQuestion, EvidenceUnit, CoverageRecord,
    PaperRole, PhaseRun, isoformat_utc,
)
from app.schemas.schemas import (
    ContractOut, ResearchQuestionOut, EvidenceUnitOut,
    CoverageRecordOut, PaperRoleOut, PhaseRunOut,
)

router = APIRouter()


# === Contract ===

@router.get("/tasks/{task_id}/contract", response_model=ContractOut | None)
def get_contract(task_id: str, db: Session = Depends(get_db_session)):
    """Get the active research contract for a task."""
    contract = db.query(ResearchContract).filter(
        ResearchContract.task_id == task_id,
        ResearchContract.status == "active",
    ).first()
    if not contract:
        return None
    return _to_contract_out(contract)


# === Questions ===

@router.get("/tasks/{task_id}/questions", response_model=list[ResearchQuestionOut])
def get_questions(task_id: str, db: Session = Depends(get_db_session)):
    """Get research questions for a task (excludes superseded)."""
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.status != "superseded",
    ).order_by(ResearchQuestion.importance.desc()).all()
    return [_to_question_out(q) for q in questions]


# === Evidence ===

@router.get("/tasks/{task_id}/evidence", response_model=list[EvidenceUnitOut])
def get_evidence(
    task_id: str,
    paper_id: str | None = None,
    evidence_type: str | None = None,
    verification_status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db_session),
):
    """Get evidence units for a task with optional filters."""
    query = db.query(EvidenceUnit).filter(EvidenceUnit.task_id == task_id)
    if paper_id:
        query = query.filter(EvidenceUnit.paper_id == paper_id)
    if evidence_type:
        query = query.filter(EvidenceUnit.evidence_type == evidence_type)
    if verification_status:
        query = query.filter(EvidenceUnit.verification_status == verification_status)

    total = query.count()
    results = query.order_by(EvidenceUnit.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return [_to_evidence_out(e) for e in results]


@router.get("/tasks/{task_id}/evidence/{evidence_id}", response_model=EvidenceUnitOut | None)
def get_evidence_unit(task_id: str, evidence_id: str, db: Session = Depends(get_db_session)):
    """Get a single evidence unit."""
    eu = db.query(EvidenceUnit).filter(
        EvidenceUnit.id == evidence_id,
        EvidenceUnit.task_id == task_id,
    ).first()
    if not eu:
        return None
    return _to_evidence_out(eu)


# === Coverage ===

@router.get("/tasks/{task_id}/coverage", response_model=list[CoverageRecordOut])
def get_coverage(task_id: str, db: Session = Depends(get_db_session)):
    """Get coverage records for a task."""
    records = db.query(CoverageRecord).filter(
        CoverageRecord.task_id == task_id,
    ).all()
    return [_to_coverage_out(c) for c in records]


# === Paper Roles ===

@router.get("/tasks/{task_id}/paper-roles", response_model=list[PaperRoleOut])
def get_paper_roles(task_id: str, db: Session = Depends(get_db_session)):
    """Get paper role classifications for a task."""
    roles = db.query(PaperRole).filter(
        PaperRole.task_id == task_id,
    ).all()
    return [_to_role_out(r) for r in roles]


# === Phases ===

@router.get("/tasks/{task_id}/phases", response_model=list[PhaseRunOut])
def get_phases(task_id: str, db: Session = Depends(get_db_session)):
    """Get phase execution records for a task."""
    from app.db.repositories import phase_repo
    phases = phase_repo.get_all_phases(db, task_id)
    return [_to_phase_out(p) for p in phases]


# === Helpers ===

def _to_contract_out(contract: ResearchContract) -> ContractOut:
    return ContractOut(
        id=contract.id,
        task_id=contract.task_id,
        topic=contract.topic,
        target_problem=contract.target_problem,
        target_setting=contract.target_setting,
        desired_output=contract.desired_output,
        novelty_bar=contract.novelty_bar,
        preferred_directions=json.loads(contract.preferred_directions_json or "[]"),
        excluded_directions=json.loads(contract.excluded_directions_json or "[]"),
        gpu_available=contract.gpu_available,
        max_gpu_hours=contract.max_gpu_hours,
        max_api_budget=contract.max_api_budget,
        max_runtime_minutes=contract.max_runtime_minutes,
        allow_large_benchmark=contract.allow_large_benchmark if contract.allow_large_benchmark is not None else True,
        allow_model_training=contract.allow_model_training if contract.allow_model_training is not None else True,
        key_terms=json.loads(contract.key_terms_json or "[]"),
        experiment_preferences=json.loads(contract.experiment_preferences_json or "{}"),
        time_scope_start=contract.time_scope_start,
        time_scope_end=contract.time_scope_end,
        status=contract.status,
        confidence=contract.confidence if contract.confidence is not None else 0.5,
        version=contract.version,
        input_hash=contract.input_hash,
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
        importance=q.importance if q.importance is not None else 0.5,
        searchability=q.searchability if q.searchability is not None else 0.5,
        status=q.status,
        axis_name=q.axis_name,
        version=q.version,
        created_at=isoformat_utc(q.created_at),
    )


def _to_evidence_out(e: EvidenceUnit) -> EvidenceUnitOut:
    return EvidenceUnitOut(
        id=e.id,
        task_id=e.task_id,
        paper_id=e.paper_id,
        evidence_type=e.evidence_type,
        normalized_claim=e.normalized_claim,
        original_span=e.original_span,
        section=e.section,
        page_number=e.page_number,
        dataset_name=e.dataset_name,
        metric_name=e.metric_name,
        result_value=e.result_value,
        extraction_method=e.extraction_method or "llm",
        extraction_confidence=e.extraction_confidence if e.extraction_confidence is not None else 0.5,
        verification_status=e.verification_status or "unverified",
        created_at=isoformat_utc(e.created_at),
    )


def _to_coverage_out(c: CoverageRecord) -> CoverageRecordOut:
    return CoverageRecordOut(
        id=c.id,
        task_id=c.task_id,
        question_id=c.question_id,
        coverage_score=c.coverage_score if c.coverage_score is not None else 0.0,
        confidence=c.confidence if c.confidence is not None else 0.0,
        supporting_evidence_count=c.supporting_evidence_count or 0,
        contradicting_evidence_count=c.contradicting_evidence_count or 0,
        direct_neighbor_count=c.direct_neighbor_count or 0,
        unresolved_aspects=json.loads(c.unresolved_aspects_json or "[]"),
        unavailable_reason=c.unavailable_reason,
        updated_at=isoformat_utc(c.updated_at),
        created_at=isoformat_utc(c.created_at),
    )


def _to_role_out(r: PaperRole) -> PaperRoleOut:
    return PaperRoleOut(
        id=r.id,
        task_id=r.task_id,
        paper_id=r.paper_id,
        role=r.role,
        confidence=r.confidence if r.confidence is not None else 0.5,
        reason=r.reason,
        created_at=isoformat_utc(r.created_at),
    )


def _to_phase_out(p: PhaseRun) -> PhaseRunOut:
    return PhaseRunOut(
        id=p.id,
        task_id=p.task_id,
        phase_name=p.phase_name,
        status=p.status,
        attempt_count=p.attempt_count or 0,
        started_at=isoformat_utc(p.started_at) if p.started_at else None,
        completed_at=isoformat_utc(p.completed_at) if p.completed_at else None,
        input_version=p.input_version,
        output_version=p.output_version,
        error_message=p.error_message,
        round_number=p.round_number,
        output_summary=p.output_summary,
        created_at=isoformat_utc(p.created_at),
        updated_at=isoformat_utc(p.updated_at),
    )

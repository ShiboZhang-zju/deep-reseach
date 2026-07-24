"""Persistence helpers for lightweight intervention candidates."""

import json

from sqlalchemy.orm import Session

from app.db.models import GapCandidate, InterventionCandidate
from app.db.repositories.gap_repo import CrossTaskValidationError


def create_intervention_candidate(db: Session, task_id: str, gap_id: str, data: dict) -> InterventionCandidate:
    gap = db.get(GapCandidate, gap_id)
    if not gap or gap.task_id != task_id:
        raise CrossTaskValidationError(f"Gap {gap_id} does not belong to task {task_id}")
    item = InterventionCandidate(
        task_id=task_id,
        gap_id=gap_id,
        intervention_type=data["intervention_type"],
        failure_mechanism=data["failure_mechanism"],
        proposed_intervention=data["proposed_intervention"],
        intermediate_effect=data["intermediate_effect"],
        measurable_outcome=data["measurable_outcome"],
        required_components_json=json.dumps(data.get("required_components", []), ensure_ascii=False),
        dependency_paper_ids_json=json.dumps(data.get("dependency_paper_ids", []), ensure_ascii=False),
        implementation_cost=data.get("implementation_cost"),
        mechanism_confidence=data.get("mechanism_confidence"),
        evidence_gate=data.get("evidence_gate", "UNKNOWN"),
        novelty_gate=data.get("novelty_gate", "UNKNOWN"),
        feasibility_gate=data.get("feasibility_gate", "UNKNOWN"),
        gate_rationale_json=json.dumps(data.get("gate_rationale", {}), ensure_ascii=False),
        status=data.get("status", "candidate"),
    )
    db.add(item)
    db.flush()
    return item


def list_interventions_for_task(db: Session, task_id: str) -> list[InterventionCandidate]:
    return db.query(InterventionCandidate).filter(
        InterventionCandidate.task_id == task_id,
    ).order_by(InterventionCandidate.created_at).all()

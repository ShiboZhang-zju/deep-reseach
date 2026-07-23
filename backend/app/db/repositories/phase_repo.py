"""Phase repository — CRUD for PhaseRun records.

Phase 2.2A:
- should_skip_phase checks for completed with matching input_version (not just latest)
- attempt_count increments based on previous max
- No skip_phase() (removed to avoid masking completed records)
"""

from datetime import datetime, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import PhaseRun


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def start_phase(db: Session, task_id: str, phase_name: str,
                input_version: str = "", round_number: int = None) -> PhaseRun:
    """Create a new PhaseRun in running state.

    Phase 2.2A: attempt_count = max previous attempt + 1 for same
    task_id + phase_name + input_version.
    """
    # Find max attempt_count for same task/phase/input
    max_attempt = db.query(func.max(PhaseRun.attempt_count)).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == phase_name,
        PhaseRun.input_version == input_version,
    ).scalar() or 0

    pr = PhaseRun(
        task_id=task_id,
        phase_name=phase_name,
        status="running",
        started_at=_utcnow(),
        attempt_count=max_attempt + 1,
        input_version=input_version,
        round_number=round_number,
    )
    db.add(pr)
    db.flush()
    return pr


def complete_phase(db: Session, phase_run_id: str, output_version: str = "",
                   output_summary: str = ""):
    """Mark a PhaseRun as completed."""
    pr = db.get(PhaseRun, phase_run_id)
    if pr:
        pr.status = "completed"
        pr.completed_at = _utcnow()
        pr.output_version = output_version
        pr.output_summary = output_summary
        db.flush()


def fail_phase(db: Session, phase_run_id: str, error_message: str):
    """Mark a PhaseRun as failed."""
    pr = db.get(PhaseRun, phase_run_id)
    if pr:
        pr.status = "failed"
        pr.completed_at = _utcnow()
        pr.error_message = error_message[:2000]
        db.flush()


def should_skip_phase(db: Session, task_id: str, phase_name: str,
                      input_version: str = "") -> bool:
    """Check if a phase should be skipped.

    Phase 2.2A: Query for ANY completed PhaseRun with matching input_version,
    not just the latest record. This prevents a 'skipped' PhaseRun from
    masking a previously completed one.
    """
    if not input_version:
        return False

    completed = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == phase_name,
        PhaseRun.status == "completed",
        PhaseRun.input_version == input_version,
    ).first()

    return completed is not None


def get_latest_completed_phase(db: Session, task_id: str, phase_name: str,
                                input_version: str = "") -> PhaseRun | None:
    """Get the latest completed PhaseRun for a given task and phase."""
    query = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == phase_name,
        PhaseRun.status == "completed",
    )
    if input_version:
        query = query.filter(PhaseRun.input_version == input_version)
    return query.order_by(PhaseRun.created_at.desc()).first()


def get_latest_phase(db: Session, task_id: str, phase_name: str) -> PhaseRun | None:
    """Get the latest PhaseRun for a given task and phase (any status)."""
    return db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == phase_name,
    ).order_by(PhaseRun.created_at.desc()).first()


def mark_interrupted_phases(db: Session, task_id: str):
    """Mark any 'running' phases as interrupted (for crash recovery)."""
    running = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.status == "running",
    ).all()
    for pr in running:
        pr.status = "failed"
        pr.error_message = "interrupted_by_restart"
        pr.completed_at = _utcnow()
    db.flush()
    return len(running)


def get_all_phases(db: Session, task_id: str) -> list[PhaseRun]:
    """Get all PhaseRun records for a task, ordered by creation."""
    return db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
    ).order_by(PhaseRun.created_at).all()

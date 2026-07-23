"""Phase repository — CRUD for PhaseRun records."""

from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.db.models import PhaseRun


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def start_phase(db: Session, task_id: str, phase_name: str,
                input_version: str = "", round_number: int = None) -> PhaseRun:
    """Create a new PhaseRun in running state."""
    pr = PhaseRun(
        task_id=task_id,
        phase_name=phase_name,
        status="running",
        started_at=_utcnow(),
        attempt_count=1,
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


def skip_phase(db: Session, task_id: str, phase_name: str, reason: str = ""):
    """Record a skipped phase."""
    pr = PhaseRun(
        task_id=task_id,
        phase_name=phase_name,
        status="skipped",
        attempt_count=0,
        output_summary=f"skipped: {reason}",
    )
    db.add(pr)
    db.flush()
    return pr


def get_latest_phase(db: Session, task_id: str, phase_name: str) -> PhaseRun | None:
    """Get the latest PhaseRun for a given task and phase."""
    return db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == phase_name,
    ).order_by(PhaseRun.created_at.desc()).first()


def should_skip_phase(db: Session, task_id: str, phase_name: str,
                      input_version: str = "") -> bool:
    """Check if a phase should be skipped (already completed with same input)."""
    latest = get_latest_phase(db, task_id, phase_name)
    if not latest:
        return False
    if latest.status == "completed":
        if not input_version or latest.input_version == input_version:
            return True
    return False


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

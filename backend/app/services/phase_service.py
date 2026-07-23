"""Phase service — wraps phases with PhaseRun tracking.

Phase 2.1 fixes:
- (#16) PhaseRun covers search round, source prep, evidence, coverage, summary
- (#17) Use SHA-256 for output_version, not Python hash()
- (#18) Fix skip-then-reexecute problem: should_skip checks input_version properly
"""

import hashlib
import json
import logging

from app.db.session import SessionLocal
from app.db.repositories import phase_repo

logger = logging.getLogger(__name__)


def compute_output_version(result) -> str:
    """(#17) Compute stable SHA-256 output version, not Python hash()."""
    if result is None:
        return "none"
    try:
        if isinstance(result, (list, dict)):
            content = json.dumps(result, ensure_ascii=False, default=str)
        elif hasattr(result, '__dict__'):
            content = json.dumps(result.__dict__, ensure_ascii=False, default=str)
        else:
            content = str(result)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "computed"


async def execute_phase(db, task_id: str, phase_name: str, operation,
                        input_version: str = "", round_number: int = None):
    """Execute a phase with PhaseRun tracking.

    Args:
        db: SQLAlchemy session
        task_id: Task ID
        phase_name: Name of the phase
        operation: Async callable(db) -> result
        input_version: Hash/version of input data (for skip detection)
        round_number: Optional round number for search rounds

    Returns:
        Result of operation, or None if skipped.
    """
    # (#18) Check if should skip — only if completed AND same input_version
    if phase_repo.should_skip_phase(db, task_id, phase_name, input_version):
        logger.info("Task %s: phase '%s' skipped (completed, same input)",
                    task_id[:8], phase_name)
        phase_repo.skip_phase(db, task_id, phase_name, "already_completed_same_input")
        db.commit()
        return None

    # Start phase
    pr = phase_repo.start_phase(db, task_id, phase_name, input_version, round_number)
    db.commit()

    try:
        result = await operation(db)
        output_ver = compute_output_version(result)
        phase_repo.complete_phase(db, pr.id,
                                  output_version=output_ver,
                                  output_summary=str(result)[:500] if result else "completed")
        db.commit()
        return result
    except Exception as e:
        phase_repo.fail_phase(db, pr.id, str(e))
        db.commit()
        raise


def mark_interrupted(task_id: str):
    """Mark interrupted phases on startup recovery."""
    db = SessionLocal()
    try:
        count = phase_repo.mark_interrupted_phases(db, task_id)
        if count:
            logger.warning("Task %s: marked %d interrupted phases as failed", task_id[:8], count)
            db.commit()
    except Exception as e:
        logger.error("Failed to mark interrupted phases: %s", e)
        db.rollback()
    finally:
        db.close()

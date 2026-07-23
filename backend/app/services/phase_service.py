"""Phase service — wraps phases with PhaseRun tracking.

Provides execute_phase() async context manager that:
- Checks if phase should be skipped (already completed, same input)
- Records start/completion/failure
- Supports retry on failure
"""

import logging
from contextlib import asynccontextmanager

from app.db.session import SessionLocal
from app.db.repositories import phase_repo

logger = logging.getLogger(__name__)


async def execute_phase(db, task_id: str, phase_name: str, operation, input_version: str = ""):
    """Execute a phase with PhaseRun tracking.

    Args:
        db: SQLAlchemy session
        task_id: Task ID
        phase_name: Name of the phase
        operation: Async callable that takes db and returns a result
        input_version: Hash/version of input data (for skip detection)

    Returns:
        The result of operation, or None if skipped.
    """
    # Check if should skip
    if phase_repo.should_skip_phase(db, task_id, phase_name, input_version):
        logger.info("Task %s: phase '%s' skipped (already completed, same input)",
                    task_id[:8], phase_name)
        phase_repo.skip_phase(db, task_id, phase_name, "already_completed")
        db.commit()
        return None

    # Start phase
    pr = phase_repo.start_phase(db, task_id, phase_name, input_version)
    db.commit()

    try:
        result = await operation(db)
        phase_repo.complete_phase(db, pr.id,
                                  output_version=str(hash(str(result)))[:16] if result else "",
                                  output_summary=str(result)[:500] if result else "")
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

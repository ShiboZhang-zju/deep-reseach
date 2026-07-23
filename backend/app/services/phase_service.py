"""Phase service — wraps phases with PhaseRun tracking.

Phase 2.2A:
- (#3) Fixed skip logic: queries for completed with matching input_version
- (#3) Removed skip_phase() — no longer creates masking skipped records
- (#3) attempt_count increments based on previous max
- (#3) All versions use full SHA-256 (64 hex chars)
"""

import hashlib
import json
import logging

from app.db.session import SessionLocal
from app.db.repositories import phase_repo

logger = logging.getLogger(__name__)


def compute_output_version(result) -> str:
    """Compute stable SHA-256 output version (64 hex chars)."""
    if result is None:
        return hashlib.sha256(b"none").hexdigest()
    try:
        if isinstance(result, (list, dict)):
            content = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
        elif hasattr(result, '__dict__'):
            content = json.dumps(result.__dict__, sort_keys=True, ensure_ascii=False, default=str)
        else:
            content = str(result)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(b"error").hexdigest()


async def execute_phase(db, task_id: str, phase_name: str, operation,
                        input_version: str = "", round_number: int = None):
    """Execute a phase with PhaseRun tracking.

    Phase 2.2A:
    - Checks should_skip_phase (queries for completed with matching input_version)
    - Does NOT create a 'skipped' PhaseRun (to avoid masking completed records)
    - Records start/completion/failure
    - attempt_count increments properly
    """
    # Check if should skip
    if phase_repo.should_skip_phase(db, task_id, phase_name, input_version):
        logger.info("Task %s: phase '%s' skipped (completed with same input_version)",
                    task_id[:8], phase_name)
        # Do NOT create a skipped PhaseRun — just return None
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

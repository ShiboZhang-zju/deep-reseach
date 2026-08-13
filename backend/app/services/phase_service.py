"""Phase service — wraps phases with PhaseRun tracking.

Phase 2.2A Closure:
- (#6) Stable output version: filter SQLAlchemy _ fields, use dataclass/Pydantic
"""

import dataclasses
import hashlib
import json
import logging
import time

from pydantic import BaseModel

from app.db.session import SessionLocal
from app.db.repositories import paper_repo, phase_repo

logger = logging.getLogger(__name__)


def stable_phase_payload(result) -> dict | str:
    """Convert result to a stable serializable payload.

    Phase 2.2A Closure (#6):
    - Filter all _-prefixed fields from SQLAlchemy __dict__
    - Use to_phase_payload() if available
    - Use dataclasses.asdict for dataclasses
    - Use model_dump for Pydantic models
    """
    if result is None:
        return "none"

    if hasattr(result, "to_phase_payload"):
        return result.to_phase_payload()

    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)

    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")

    if isinstance(result, dict):
        return result

    if isinstance(result, (list, tuple)):
        return [_stable_item(item) for item in result]

    if hasattr(result, '__dict__'):
        # Filter SQLAlchemy internal fields
        return {k: v for k, v in result.__dict__.items()
                if not k.startswith('_') and not callable(v)}

    return str(result)


def _stable_item(item):
    """Convert a single item to a stable form."""
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    if dataclasses.is_dataclass(item):
        return dataclasses.asdict(item)
    if isinstance(item, dict):
        return item
    return str(item)


def compute_output_version(result) -> str:
    """Compute stable SHA-256 output version (64 hex chars)."""
    payload = stable_phase_payload(result)
    try:
        content = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), default=str)
    except Exception:
        content = str(payload)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def execute_phase(db, task_id: str, phase_name: str, operation,
                        input_version: str = "", round_number: int = None):
    """Execute a phase with PhaseRun tracking."""
    if phase_repo.should_skip_phase(db, task_id, phase_name, input_version):
        logger.info("Task %s: phase '%s' skipped (completed with same input_version)",
                    task_id[:8], phase_name)
        return None

    pr = phase_repo.start_phase(db, task_id, phase_name, input_version, round_number)
    db.commit()

    started = time.perf_counter()
    try:
        result = await operation(db)
        duration_ms = int((time.perf_counter() - started) * 1000)
        output_ver = compute_output_version(result)
        payload = stable_phase_payload(result)
        output_json_str = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        summary = output_json_str[:500]
        phase_repo.complete_phase(db, pr.id,
                                  output_version=output_ver,
                                  output_summary=summary,
                                  output_json=output_json_str)
        # Phase-level duration trace so runtime is queryable. The per-step
        # traces inside a phase historically never populated duration_ms, which
        # left no structured way to see how long each phase actually took.
        paper_repo.save_trace(db, task_id, phase_name, "phase_duration",
                              round_number=round_number, duration_ms=duration_ms)
        db.commit()
        return result
    except Exception as e:
        duration_ms = int((time.perf_counter() - started) * 1000)
        phase_repo.fail_phase(db, pr.id, str(e))
        try:
            paper_repo.save_trace(db, task_id, phase_name, "phase_duration",
                                  round_number=round_number, duration_ms=duration_ms,
                                  output_data={"error": str(e)[:500]})
        except Exception:
            pass
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

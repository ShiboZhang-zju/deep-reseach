"""Database repository for research tasks."""

import json
import time

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.models import ResearchTask
from app.agent.state import ResearchState


def create_task(db: Session, user_input: str, max_rounds: int | None = None) -> ResearchTask:
    """Create a pending task.

    max_rounds defaults to settings.max_rounds (MAX_ROUNDS in .env) instead of a
    hard-coded constant, otherwise the configured round budget is silently
    ignored for every task created through the API.
    """
    if max_rounds is None:
        from app.config import settings
        max_rounds = settings.max_rounds
    task = ResearchTask(user_input=user_input, status="pending", max_rounds=max_rounds)
    state = ResearchState(user_input=user_input)
    task.state_json = state.to_json()
    db.add(task)
    db.flush()
    return task


def get_task(db: Session, task_id: str) -> ResearchTask | None:
    return db.get(ResearchTask, task_id)


def list_tasks(db: Session, limit: int = 50) -> list[ResearchTask]:
    return db.query(ResearchTask).order_by(ResearchTask.created_at.desc()).limit(limit).all()


def _standalone_status_write(task_id: str, **fields) -> None:
    """Write status fields through a dedicated short transaction, with lock
    retries (0.5s/1s/2s backoff).

    Fallback for `flush` raising OperationalError(database is locked) inside
    the caller's session: after such a failure the session needs rollback()
    (dropping its pending writes), and without this fallback the task DIES —
    observed 2026-09-07, task 789bec4a failed on the bare
    `UPDATE research_tasks` while another task held the write lock. A status
    flip is independent bookkeeping, so a standalone write is semantically
    safe and keeps the task alive.
    """
    from app.db.session import SessionLocal

    for attempt in range(1, 4):
        session = SessionLocal()
        try:
            task = session.get(ResearchTask, task_id)
            if task:
                for key, value in fields.items():
                    setattr(task, key, value)
                session.commit()
            return
        except OperationalError as exc:
            session.rollback()
            if "database is locked" not in str(exc) or attempt == 3:
                raise
            time.sleep(0.5 * attempt)
        finally:
            session.close()


def _flush_with_lock_fallback(db: Session, task_id: str, **fields) -> None:
    """Flush `fields` in the caller's session; on lock contention fall back to
    a standalone retried write. The rollback after a failed flush drops the
    caller's OTHER pending writes — acceptable here because they were already
    uncommittable (the same lock would break the caller's eventual commit
    anyway), and it keeps the task alive instead of failing it.
    """
    task = db.get(ResearchTask, task_id)
    if not task:
        return
    for key, value in fields.items():
        setattr(task, key, value)
    try:
        db.flush()
    except OperationalError as exc:
        if "database is locked" not in str(exc):
            raise
        db.rollback()
        _standalone_status_write(task_id, **fields)


def update_status(db: Session, task_id: str, status: str):
    _flush_with_lock_fallback(db, task_id, status=status)


def update_stop_reason(db: Session, task_id: str, reason: str):
    _flush_with_lock_fallback(db, task_id, stop_reason=reason)


def get_state(db: Session, task_id: str) -> ResearchState:
    task = db.get(ResearchTask, task_id)
    if not task or not task.state_json:
        return ResearchState(task_id=task_id)
    state = ResearchState.from_json(task.state_json)
    state.task_id = task_id
    return state


def save_state(db: Session, task_id: str, state: ResearchState):
    task = db.get(ResearchTask, task_id)
    if task:
        task.state_json = state.to_json()
        task.current_round = state.current_round
        db.flush()


def update_normalized_topic(db: Session, task_id: str, topic: str):
    task = db.get(ResearchTask, task_id)
    if task:
        task.normalized_topic = topic
        db.flush()

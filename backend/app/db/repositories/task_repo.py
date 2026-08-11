"""Database repository for research tasks."""

import json

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


def update_status(db: Session, task_id: str, status: str):
    task = db.get(ResearchTask, task_id)
    if task:
        task.status = status
        db.flush()


def update_stop_reason(db: Session, task_id: str, reason: str):
    task = db.get(ResearchTask, task_id)
    if task:
        task.stop_reason = reason
        db.flush()


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

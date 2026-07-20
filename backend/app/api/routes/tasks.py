"""Task API routes."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.repositories import task_repo
from app.db.models import isoformat_utc
from app.agent.runner import start_agent, stop_agent, run_experiment_generation
from app.schemas.schemas import TaskCreate, TaskOut, ClarifyRequest, FeedbackRequest, IdeaSelectRequest

router = APIRouter()


async def _deferred_start_agent(task_id: str):
    """Start agent after yielding control so HTTP response is sent first.

    If start_agent rejects (capacity full), emit an error event so the
    frontend can surface it instead of waiting indefinitely.
    """
    await asyncio.sleep(0.05)
    started = start_agent(task_id)
    if not started:
        from app.services.event_service import emit_event
        from app.config import settings
        emit_event(task_id, "error", {
            "message": f"已达最大并发任务数 ({settings.max_concurrent_agents})，请等待已有任务完成后再启动",
        })


@router.post("/tasks", response_model=TaskOut)
def create_task(body: TaskCreate, db: Session = Depends(get_db_session)):
    task = task_repo.create_task(db, body.user_input)
    db.commit()
    return _to_task_out(task)


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(limit: int = 50, db: Session = Depends(get_db_session)):
    tasks = task_repo.list_tasks(db, limit)
    return [_to_task_out(t) for t in tasks]


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return _to_task_out(task)


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: str, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    # P1-11: start_agent now does atomic capacity-check-and-register internally,
    # closing the race window. The route just needs to handle the reject case.
    asyncio.create_task(_deferred_start_agent(task_id))
    return {"status": "started"}


@router.post("/tasks/{task_id}/stop")
def stop_task(task_id: str, db: Session = Depends(get_db_session)):
    stop_agent(task_id)
    task_repo.update_status(db, task_id, "stopped")
    db.commit()
    # Clean up SSE event queue for stopped task
    from app.services.event_service import cleanup_task_events
    cleanup_task_events(task_id)
    return {"status": "stopped"}


@router.post("/tasks/{task_id}/clarify")
async def submit_clarification(task_id: str, body: ClarifyRequest, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    state = task_repo.get_state(db, task_id)
    state.user_input = task.user_input + "\nClarifications: " + " | ".join(body.answers)
    state.research_questions = []  # Clear old questions
    task_repo.save_state(db, task_id, state)
    db.commit()

    # Restart agent AFTER response is sent (avoids blocking event loop)
    asyncio.create_task(_deferred_start_agent(task_id))
    return {"status": "restarted"}


@router.post("/tasks/{task_id}/feedback")
async def submit_feedback(task_id: str, body: FeedbackRequest, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    from app.db.repositories.paper_repo import save_feedback
    save_feedback(db, task_id, "research_feedback", body.content, None, body.need_more_research)

    state = task_repo.get_state(db, task_id)
    state.user_feedback = body.content
    task_repo.save_state(db, task_id, state)
    db.commit()

    if body.need_more_research:
        asyncio.create_task(_deferred_start_agent(task_id))

    return {"status": "feedback_saved"}


@router.post("/tasks/{task_id}/ideas/select")
def select_ideas(task_id: str, body: IdeaSelectRequest, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    state = task_repo.get_state(db, task_id)
    state.selected_idea_ids = body.idea_ids
    task_repo.save_state(db, task_id, state)

    from app.db.repositories.paper_repo import save_feedback
    save_feedback(db, task_id, "idea_selection", "", body.idea_ids, False)
    db.commit()

    return {"status": "ideas_selected"}


@router.post("/tasks/{task_id}/ideas/judge")
async def judge_selected_ideas(task_id: str, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    state = task_repo.get_state(db, task_id)
    if not state.selected_idea_ids:
        raise HTTPException(400, "No ideas selected")

    # Start experiment generation asynchronously (avoids HTTP timeout)
    async def _deferred_judge():
        await asyncio.sleep(0.05)
        await run_experiment_generation(task_id, state.selected_idea_ids)

    asyncio.create_task(_deferred_judge())
    return {"status": "started", "idea_count": len(state.selected_idea_ids)}


@router.post("/tasks/{task_id}/experiments")
async def generate_experiments(task_id: str, db: Session = Depends(get_db_session)):
    task = task_repo.get_task(db, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    state = task_repo.get_state(db, task_id)
    if not state.selected_idea_ids:
        raise HTTPException(400, "No ideas selected")

    result = await run_experiment_generation(task_id, state.selected_idea_ids)
    return result


def _to_task_out(task) -> TaskOut:
    return TaskOut(
        id=task.id,
        user_input=task.user_input,
        normalized_topic=task.normalized_topic,
        status=task.status,
        current_round=task.current_round,
        max_rounds=task.max_rounds,
        stop_reason=task.stop_reason,
        state_json=task.state_json,
        created_at=isoformat_utc(task.created_at),
        updated_at=isoformat_utc(task.updated_at),
    )

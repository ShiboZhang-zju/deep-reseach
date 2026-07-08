"""SSE event streaming routes."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.services.event_service import event_stream

router = APIRouter()


@router.get("/tasks/{task_id}/events")
async def stream_events(task_id: str):
    return StreamingResponse(
        event_stream(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

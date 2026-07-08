"""SSE event service for real-time progress push."""

import asyncio
import json
from collections import defaultdict

# Per-task event queues
_event_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)


def emit_event(task_id: str, event_type: str, data: dict):
    """Push an event to the task's SSE queue."""
    event = {"event": event_type, "data": json.dumps(data, ensure_ascii=False, default=str)}
    _event_queues[task_id].put_nowait(event)


async def event_stream(task_id: str):
    """Async generator yielding SSE events for a task."""
    queue = _event_queues[task_id]
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30)
            yield f"event: {event['event']}\ndata: {event['data']}\n\n"
        except asyncio.TimeoutError:
            yield f": keepalive\n\n"

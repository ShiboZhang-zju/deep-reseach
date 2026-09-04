"""SSE event service for real-time progress push.

Improvements (P0-3):
- Bounded queue (maxsize=200) to prevent memory leak when frontend disconnects
- cleanup_task_events() called when task reaches terminal state
- emit_event drops oldest event if queue is full (non-blocking)
"""

import asyncio
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Maximum events buffered per task (prevents unbounded memory growth)
_MAX_QUEUE_SIZE = 200

# Per-task event queues (bounded)
_event_queues: dict[str, asyncio.Queue] = {}


def _get_or_create_queue(task_id: str) -> asyncio.Queue:
    """Get or create a bounded queue for a task."""
    if task_id not in _event_queues:
        _event_queues[task_id] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
    return _event_queues[task_id]


def emit_event(task_id: str, event_type: str, data: dict):
    """Push an event to the task's SSE queue.

    Uses put_nowait; if queue is full, drops the oldest event to make room.
    This prevents the agent from blocking if the frontend is disconnected.
    """
    queue = _get_or_create_queue(task_id)
    event = {"event": event_type, "data": json.dumps(data, ensure_ascii=False, default=str)}

    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        # Queue full — drop oldest to make room for newest
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue for task %s still full after drop, event lost", task_id[:8])


def cleanup_task_events(task_id: str):
    """Remove all queued events for a task (call when task reaches terminal state)."""
    if task_id in _event_queues:
        del _event_queues[task_id]
        logger.debug("Cleaned up event queue for task %s", task_id[:8])


# Terminal statuses that trigger cleanup. Beyond the obvious three this also
# covers the newer honest terminal states — without them a task ending in
# waiting_for_user_review / insufficient_evidence / more_research_required /
# abstained left its queue (up to 200 events) resident until process exit.
_TERMINAL_STATUSES = {
    "done", "stopped", "failed", "abstained", "insufficient_evidence",
    "more_research_required", "waiting_for_user_review",
}


def emit_event_with_cleanup(task_id: str, event_type: str, data: dict):
    """Emit event and auto-cleanup queue if task reaches terminal state."""
    emit_event(task_id, event_type, data)
    # Check if this is a terminal status event
    if event_type == "status" and isinstance(data, dict):
        status = data.get("status")
        if status in _TERMINAL_STATUSES:
            # Schedule cleanup after a short delay (allow SSE clients to receive final events)
            asyncio.create_task(_delayed_cleanup(task_id))


async def _delayed_cleanup(task_id: str, delay: float = 10.0):
    """Clean up event queue after a delay (let clients receive final events)."""
    await asyncio.sleep(delay)
    cleanup_task_events(task_id)


async def event_stream(task_id: str):
    """Async generator yielding SSE events for a task."""
    queue = _get_or_create_queue(task_id)
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30)
            yield f"event: {event['event']}\ndata: {event['data']}\n\n"
        except asyncio.TimeoutError:
            yield f": keepalive\n\n"

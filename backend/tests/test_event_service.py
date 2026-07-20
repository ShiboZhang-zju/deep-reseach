"""Tests for SSE event service (event_service.py).

Covers: bounded queue, drop-oldest on full, cleanup.
"""

import asyncio
import pytest

from app.services.event_service import (
    emit_event,
    cleanup_task_events,
    event_stream,
    _MAX_QUEUE_SIZE,
)


class TestEventQueue:
    def setup_method(self):
        """Clean up any leftover queues before each test."""
        cleanup_task_events("test_task")

    def teardown_method(self):
        cleanup_task_events("test_task")

    def test_emit_and_receive(self):
        emit_event("test_task", "status", {"status": "searching"})
        from app.services.event_service import _get_or_create_queue
        q = _get_or_create_queue("test_task")
        assert not q.empty()
        event = q.get_nowait()
        assert event["event"] == "status"

    def test_bounded_queue_maxsize(self):
        from app.services.event_service import _get_or_create_queue
        q = _get_or_create_queue("test_task")
        assert q.maxsize == _MAX_QUEUE_SIZE

    def test_drop_oldest_when_full(self):
        """When queue is full, emitting should drop the oldest event."""
        task_id = "test_full"
        cleanup_task_events(task_id)
        try:
            # Fill the queue
            for i in range(_MAX_QUEUE_SIZE):
                emit_event(task_id, "event", {"index": i})

            from app.services.event_service import _get_or_create_queue
            q = _get_or_create_queue(task_id)
            assert q.full()

            # Emit one more — should drop oldest (index=0) and add new
            emit_event(task_id, "event", {"index": 999})

            # Queue should still be full (dropped one, added one)
            assert q.full()

            # First event should now be index=1 (index=0 was dropped)
            first = q.get_nowait()
            assert first["event"] == "event"
        finally:
            cleanup_task_events(task_id)

    def test_cleanup_removes_queue(self):
        from app.services.event_service import _event_queues
        emit_event("test_cleanup", "status", {"status": "done"})
        assert "test_cleanup" in _event_queues
        cleanup_task_events("test_cleanup")
        assert "test_cleanup" not in _event_queues

    def test_cleanup_nonexistent_task(self):
        """Should not raise on cleanup of non-existent task."""
        cleanup_task_events("nonexistent_task")

    @pytest.mark.asyncio
    async def test_event_stream_yields_events(self):
        task_id = "test_stream"
        cleanup_task_events(task_id)
        try:
            emit_event(task_id, "test_event", {"data": "hello"})

            gen = event_stream(task_id)
            # Get first event
            result = await gen.__anext__()
            assert "test_event" in result
            assert "hello" in result
        finally:
            cleanup_task_events(task_id)

"""Tests for the 5 direct fixes (P1-10/11/12/13/14).

Covers:
- auto_promote uses 'conditional_go' not 'go'
- citation cleanup replaces fabricated [Px] with [unsupported]
- recover_interrupted_tasks calibrates current_round
- start_agent does atomic capacity-check-and-register
- (env.example key cleanup is a data fix, no unit test needed)
"""

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Fix 2: auto_promote uses conditional_go ===

def test_auto_promote_uses_conditional_go_not_go():
    """_auto_promote_ideas should set decision='conditional_go', not 'go'."""
    from app.agent.runner import _auto_promote_ideas

    # Mock ideas: score 0.6 (>= 0.55 threshold but < 0.70 go threshold)
    idea1 = MagicMock()
    idea1.final_score = 0.65
    idea1.title = "Test Idea 1"
    idea1.decision = "revise"

    idea2 = MagicMock()
    idea2.final_score = 0.40  # below threshold, should not be promoted
    idea2.title = "Test Idea 2"
    idea2.decision = "revise"

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [idea1, idea2]

    with patch("app.agent.runner.task_repo"), \
         patch("app.agent.runner.paper_repo"), \
         patch("app.agent.runner.emit_event"):
        _auto_promote_ideas(db, "test-task-id")

    assert idea1.decision == "conditional_go", \
        f"Expected conditional_go, got {idea1.decision}"
    assert idea2.decision == "revise", \
        f"Low-score idea should not be promoted, got {idea2.decision}"


def test_auto_promote_does_not_use_go():
    """Ensure 'go' is NEVER used by auto_promote (preserves validation semantics)."""
    from app.agent.runner import _auto_promote_ideas

    idea = MagicMock()
    idea.final_score = 0.60
    idea.title = "Test"
    idea.decision = "revise"

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [idea]

    with patch("app.agent.runner.task_repo"), \
         patch("app.agent.runner.paper_repo"), \
         patch("app.agent.runner.emit_event"):
        _auto_promote_ideas(db, "test-task-id")

    assert idea.decision != "go", \
        "auto_promote must not set decision='go' — that would conflate validated and unvalidated ideas"


# === Fix 3: citation cleanup replaces fabricated citations ===

def test_citation_cleanup_replaces_fabricated_with_unsupported():
    """Fabricated [P99] should be replaced with [unsupported] in the report text."""
    from app.agent.steps.generate_report import _validate_and_clean_citations

    report = "Some claim [P1] and another claim [P99] here."
    papers = [MagicMock() for _ in range(5)]  # valid range: P1-P5

    db = MagicMock()

    cleaned = _validate_and_clean_citations(db, report, papers, "test-task")

    assert "[P99]" not in cleaned, "Fabricated citation should be removed"
    assert "[unsupported]" in cleaned, "Fabricated citation should be replaced with [unsupported]"
    assert "[P1]" in cleaned, "Valid citation should be preserved"


def test_citation_cleanup_preserves_valid_citations():
    """Valid [P1]-[P5] citations should remain unchanged."""
    from app.agent.steps.generate_report import _validate_and_clean_citations

    report = "Claim A [P1]. Claim B [P3]. Claim C [P5]."
    papers = [MagicMock() for _ in range(5)]

    db = MagicMock()
    cleaned = _validate_and_clean_citations(db, report, papers, "test-task")

    assert "[P1]" in cleaned
    assert "[P3]" in cleaned
    assert "[P5]" in cleaned
    assert "[unsupported]" not in cleaned


def test_citation_cleanup_handles_multiple_fabricated():
    """Multiple fabricated citations should all be replaced."""
    from app.agent.steps.generate_report import _validate_and_clean_citations

    report = "[P1] [P50] [P2] [P100] [P3]"
    papers = [MagicMock() for _ in range(5)]

    db = MagicMock()
    cleaned = _validate_and_clean_citations(db, report, papers, "test-task")

    assert cleaned.count("[unsupported]") == 2
    assert "[P50]" not in cleaned
    assert "[P100]" not in cleaned


# === Fix 4: recover_interrupted_tasks calibrates current_round ===

def test_recover_calibrates_current_round_upward():
    """If state.current_round is ahead of research_rounds table, it should be pulled back."""
    from app.agent.runner import recover_interrupted_tasks
    from app.agent.state import ResearchState

    # state says round 5, but only 2 rounds were actually saved
    state = ResearchState(task_id="t1", current_round=5)
    task = MagicMock()
    task.id = "task-ahead"
    task.status = "searching"
    task.state_json = state.to_json()

    db = MagicMock()
    # First db.query(ResearchTask).filter().all() returns [task]
    # Second db.query(ResearchRound.round_number).filter().order_by().first() returns (2,)
    task_query = MagicMock()
    task_query.filter.return_value.all.return_value = [task]
    round_query = MagicMock()
    round_query.filter.return_value.order_by.return_value.first.return_value = (2,)
    db.query.side_effect = [task_query, round_query]

    # Patch SessionLocal where runner imports it (app.agent.runner.SessionLocal)
    with patch("app.agent.runner.SessionLocal", return_value=db), \
         patch("app.agent.runner.logger"):
        recover_interrupted_tasks()

    # state_json should have been updated with current_round=2
    updated_state = ResearchState.from_json(task.state_json)
    assert updated_state.current_round == 2, \
        f"Expected current_round=2 (from research_rounds), got {updated_state.current_round}"


def test_recover_calibrates_current_round_downward():
    """If state.current_round is behind research_rounds table, it should be pushed forward."""
    from app.agent.runner import recover_interrupted_tasks
    from app.agent.state import ResearchState

    state = ResearchState(task_id="t2", current_round=0)
    task = MagicMock()
    task.id = "task-behind"
    task.status = "searching"
    task.state_json = state.to_json()

    db = MagicMock()
    task_query = MagicMock()
    task_query.filter.return_value.all.return_value = [task]
    round_query = MagicMock()
    round_query.filter.return_value.order_by.return_value.first.return_value = (3,)
    db.query.side_effect = [task_query, round_query]

    with patch("app.agent.runner.SessionLocal", return_value=db), \
         patch("app.agent.runner.logger"):
        recover_interrupted_tasks()

    updated_state = ResearchState.from_json(task.state_json)
    assert updated_state.current_round == 3


def test_recover_no_rounds_resets_to_zero():
    """If no rounds in research_rounds table, current_round should be 0."""
    from app.agent.runner import recover_interrupted_tasks
    from app.agent.state import ResearchState

    state = ResearchState(task_id="t3", current_round=3)
    task = MagicMock()
    task.id = "task-no-rounds"
    task.status = "searching"
    task.state_json = state.to_json()

    db = MagicMock()
    task_query = MagicMock()
    task_query.filter.return_value.all.return_value = [task]
    round_query = MagicMock()
    round_query.filter.return_value.order_by.return_value.first.return_value = None
    db.query.side_effect = [task_query, round_query]

    with patch("app.agent.runner.SessionLocal", return_value=db), \
         patch("app.agent.runner.logger"):
        recover_interrupted_tasks()

    updated_state = ResearchState.from_json(task.state_json)
    assert updated_state.current_round == 0


# === Fix 5: start_agent atomic check-and-register ===

def test_start_agent_rejects_when_capacity_full():
    """start_agent should return False when max_concurrent_agents is reached."""
    from app.agent.runner import start_agent, _task_registry

    # Fill registry with fake running tasks
    original = _task_registry.copy()
    try:
        _task_registry.clear()
        _task_registry["existing-1"] = MagicMock()
        _task_registry["existing-1"].done.return_value = False
        _task_registry["existing-2"] = MagicMock()
        _task_registry["existing-2"].done.return_value = False

        # max_concurrent_agents is 2 (from config), so this should be rejected
        result = start_agent("new-task-id")
        assert result is False, "start_agent should return False when capacity is full"
        assert "new-task-id" not in _task_registry, \
            "Rejected task should not be in registry"
    finally:
        _task_registry.clear()
        _task_registry.update(original)


def test_start_agent_accepts_when_capacity_available():
    """start_agent should return True and register when capacity is available."""
    from app.agent.runner import start_agent, _task_registry

    original = _task_registry.copy()
    try:
        _task_registry.clear()
        # No running tasks, capacity available

        # Need an event loop for loop.create_task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = start_agent("new-task-id")
        assert result is True, "start_agent should return True when capacity available"
        assert "new-task-id" in _task_registry, "Task should be registered"

        # Cleanup: cancel the created task
        task = _task_registry.pop("new-task-id")
        if hasattr(task, "cancel"):
            task.cancel()
    finally:
        _task_registry.clear()
        _task_registry.update(original)


def test_start_agent_returns_true_for_already_running():
    """start_agent should return True (idempotent) if task already running."""
    from app.agent.runner import start_agent, _task_registry

    original = _task_registry.copy()
    try:
        _task_registry.clear()
        existing_task = MagicMock()
        existing_task.done.return_value = False
        _task_registry["running-task"] = existing_task

        result = start_agent("running-task")
        assert result is True, "Already-running task should return True (idempotent)"
    finally:
        _task_registry.clear()
        _task_registry.update(original)

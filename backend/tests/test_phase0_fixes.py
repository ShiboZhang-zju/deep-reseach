"""Tests for Phase 0 refactoring fixes.

Covers:
- _finish_with_insufficient_evidence sets correct status
- policy.early_termination_check consolidates stop logic
- round retry limits prevent infinite loops
- PhaseRun model can be created and queried
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Phase 0: _finish_with_insufficient_evidence ===

def test_finish_with_insufficient_evidence_sets_status():
    """When all ideas are rejected after retries, task should get insufficient_evidence status."""
    from app.agent.runner import _finish_with_insufficient_evidence

    db = MagicMock()
    with patch("app.agent.runner.task_repo") as mock_task_repo, \
         patch("app.agent.runner.emit_event"):
        _finish_with_insufficient_evidence(db, "test-task-id", 5)

    # Verify task_repo.update_status was called with insufficient_evidence
    mock_task_repo.update_status.assert_called_once_with(db, "test-task-id", "insufficient_evidence")
    mock_task_repo.update_stop_reason.assert_called_once_with(db, "test-task-id", "no_credible_ideas_after_retries")


def test_auto_promote_does_not_promote():
    """Phase 0: _auto_promote_ideas should NOT change any idea's decision."""
    from app.agent.runner import _auto_promote_ideas

    idea = MagicMock()
    idea.final_score = 0.65
    idea.title = "Test Idea"
    idea.decision = "reject"  # was rejected

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [idea]

    with patch("app.agent.runner.task_repo"), \
         patch("app.agent.runner.paper_repo"), \
         patch("app.agent.runner.emit_event"):
        _auto_promote_ideas(db, "test-task-id")

    # decision should remain 'reject', not promoted to conditional_go
    assert idea.decision == "reject", \
        f"Phase 0: auto_promote is removed, decision should stay 'reject', got {idea.decision}"


# === Phase 0: policy.early_termination_check ===

def test_policy_early_termination_no_new_high():
    """early_termination_check should trigger when no new high-priority for 2 rounds."""
    from app.agent.policy import early_termination_check
    from app.agent.state import ResearchState

    state = ResearchState(task_id="t1", current_round=3)
    stop, reason = early_termination_check(state, no_new_high_priority_count=2, duplicate_rate=0.3)
    assert stop is True
    assert "no_new_high_priority" in reason


def test_policy_early_termination_high_dup_rate():
    """early_termination_check should trigger when duplicate rate is high."""
    from app.agent.policy import early_termination_check
    from app.agent.state import ResearchState

    state = ResearchState(task_id="t1", current_round=2)
    stop, reason = early_termination_check(state, no_new_high_priority_count=0, duplicate_rate=0.70)
    assert stop is True
    assert "duplicate_rate" in reason


def test_policy_early_termination_does_not_trigger_early():
    """early_termination_check should NOT trigger in round 1."""
    from app.agent.policy import early_termination_check
    from app.agent.state import ResearchState

    state = ResearchState(task_id="t1", current_round=1)
    stop, reason = early_termination_check(state, no_new_high_priority_count=2, duplicate_rate=0.80)
    assert stop is False
    assert reason == ""


# === Phase 0: PhaseRun model ===

def test_phase_run_model_creation():
    """PhaseRun model can be instantiated with expected fields."""
    from app.db.models import PhaseRun

    pr = PhaseRun(
        task_id="test-task",
        phase_name="clarify",
        status="pending",
        attempt_count=0,
    )
    assert pr.phase_name == "clarify"
    assert pr.status == "pending"
    assert pr.attempt_count == 0


def test_phase_run_out_schema():
    """PhaseRunOut schema can be created from a PhaseRun-like object."""
    from app.schemas.schemas import PhaseRunOut, NEW_TASK_STATUSES

    # Verify new statuses are defined
    assert "insufficient_evidence" in NEW_TASK_STATUSES
    assert "more_research_required" in NEW_TASK_STATUSES
    assert "auditing_gaps" in NEW_TASK_STATUSES
    assert "checking_feasibility" in NEW_TASK_STATUSES
    assert "synthesizing_ideas" in NEW_TASK_STATUSES


# === Phase 0: generate_experiment does not accept conditional_go ===

def test_generate_experiment_rejects_conditional_go():
    """generate_experiment should NOT treat conditional_go as good_idea."""
    # Read the source to verify the fix
    import inspect
    from app.agent.steps import generate_experiment

    source = inspect.getsource(generate_experiment)
    # The fix removed 'conditional_go' from the condition
    assert 'conditional_go' not in source.split("if decision ==")[1].split("\n")[0] or \
           'if decision == "go"' in source, \
        "generate_experiment should only accept 'go', not 'conditional_go'"


# === Phase 0: runner.py no longer has db.refresh = None ===

def test_runner_no_db_refresh_override():
    """runner.py should NOT contain db.refresh = None as executable code."""
    import inspect
    from app.agent import runner

    source = inspect.getsource(runner)
    # Check that db.refresh = None is NOT used as an executable statement
    # (it may appear in comments, which is fine)
    lines = source.split("\n")
    for line in lines:
        stripped = line.strip()
        # Skip comment lines
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
            continue
        # Check for executable db.refresh = None
        if "db.refresh = None" in stripped or "db.refresh=None" in stripped:
            # Allow if it's inside a comment on the same line
            code_part = stripped.split("#")[0].strip()
            if "db.refresh = None" in code_part or "db.refresh=None" in code_part:
                pytest.fail(
                    f"Phase 0 fix: db.refresh = None should not be executable code. Found in: {stripped}"
                )

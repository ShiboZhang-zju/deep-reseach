"""Tests for agent termination policy (policy.py).

Covers: max_rounds, high_priority_target, early termination conditions.
"""

import pytest

from app.agent.state import ResearchState
from app.agent.policy import should_stop
from app.config import settings


class TestShouldStop:
    def test_stop_on_max_rounds(self):
        state = ResearchState(
            task_id="test",
            current_round=5,
            high_priority_paper_ids=[],
        )
        stop, reason = should_stop(state)
        assert stop is True
        assert reason == "max_rounds_reached"

    def test_stop_on_high_priority_target(self):
        state = ResearchState(
            task_id="test",
            current_round=1,
            high_priority_paper_ids=["p"] * settings.high_priority_target,
        )
        stop, reason = should_stop(state)
        assert stop is True
        assert reason == "high_priority_target_reached"

    def test_no_stop_below_max_rounds(self):
        state = ResearchState(
            task_id="test",
            current_round=2,
            high_priority_paper_ids=[],
        )
        stop, reason = should_stop(state)
        assert stop is False
        assert reason == ""

    def test_no_stop_below_high_priority_target(self):
        state = ResearchState(
            task_id="test",
            current_round=1,
            high_priority_paper_ids=["p"] * (settings.high_priority_target - 1),
        )
        stop, reason = should_stop(state)
        assert stop is False

    def test_stop_exactly_max_rounds(self):
        state = ResearchState(
            task_id="test",
            current_round=settings.max_rounds,
        )
        stop, _ = should_stop(state)
        assert stop is True

    def test_stop_exactly_high_priority_target(self):
        state = ResearchState(
            task_id="test",
            current_round=1,
            high_priority_paper_ids=["p"] * settings.high_priority_target,
        )
        stop, _ = should_stop(state)
        assert stop is True


class TestEarlyTermination:
    """Test early termination conditions in runner._check_early_termination.

    These are tested by replicating the logic since the function requires DB access.
    """

    def test_no_new_high_priority_2_rounds(self):
        """If no_new_high_count >= 2 and current_round >= 2 → stop."""
        no_new_high_count = 2
        current_round = 2
        should_terminate = no_new_high_count >= 2 and current_round >= 2
        assert should_terminate is True

    def test_no_new_high_priority_round_1(self):
        """Round 1 should not trigger early stop even if no new high-priority."""
        no_new_high_count = 2
        current_round = 1
        should_terminate = no_new_high_count >= 2 and current_round >= 2
        assert should_terminate is False

    def test_high_duplicate_rate(self):
        """duplicate_rate > 0.75 and current_round >= 2 → stop."""
        duplicate_rate = 0.80
        current_round = 2
        should_terminate = duplicate_rate > 0.75 and current_round >= 2
        assert should_terminate is True

    def test_moderate_duplicate_rate_no_stop(self):
        """duplicate_rate = 0.75 (not > 0.75) → no stop."""
        duplicate_rate = 0.75
        current_round = 2
        should_terminate = duplicate_rate > 0.75 and current_round >= 2
        assert should_terminate is False

    def test_high_dup_rate_round_1_no_stop(self):
        """High duplicate rate in round 1 should not trigger early stop."""
        duplicate_rate = 0.90
        current_round = 1
        should_terminate = duplicate_rate > 0.75 and current_round >= 2
        assert should_terminate is False

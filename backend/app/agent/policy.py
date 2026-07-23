"""Termination policy for the agent loop.

Phase 0 refactor: ALL stop logic consolidated here.
runner.py should call should_stop() and early_termination_check() from this module only.
"""

from app.agent.state import ResearchState
from app.config import settings


def should_stop(state: ResearchState) -> tuple[bool, str]:
    """Check if the agent should stop searching.

    Primary stop conditions (checked at the start of each round):
    1. Max rounds reached
    2. High-priority target reached
    """
    # 1. Max rounds reached
    if state.current_round >= settings.max_rounds:
        return True, "max_rounds_reached"

    # 2. Enough high-priority papers
    if len(state.high_priority_paper_ids) >= settings.high_priority_target:
        return True, "high_priority_target_reached"

    return False, ""


def early_termination_check(
    state: ResearchState,
    no_new_high_priority_count: int,
    duplicate_rate: float,
) -> tuple[bool, str]:
    """Check early termination conditions. Returns (should_stop, reason).

    Called after each round completes. This replaces the inline
    _check_early_termination in runner.py.

    Conditions (checked in order):
    1. No new high-priority papers for >= 2 rounds (with min round guard)
    2. High duplicate rate (>= 0.65 for 2+ rounds)
    """
    # 1. No new high-priority for 2 consecutive rounds
    if no_new_high_priority_count >= 2 and state.current_round >= 2:
        return True, "no_new_high_priority_2_rounds"

    # 2. High duplicate rate (P4-1: 0.75 -> 0.65)
    if duplicate_rate > 0.65 and state.current_round >= 2:
        return True, f"high_duplicate_rate ({duplicate_rate:.2f})"

    return False, ""

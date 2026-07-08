"""Termination policy for the agent loop."""

from app.agent.state import ResearchState
from app.config import settings


def should_stop(state: ResearchState) -> tuple[bool, str]:
    """Check if the agent should stop searching."""
    # 1. Max rounds reached
    if state.current_round >= settings.max_rounds:
        return True, "max_rounds_reached"

    # 2. Enough high-priority papers
    if len(state.high_priority_paper_ids) >= settings.high_priority_target:
        return True, "high_priority_target_reached"

    # 3. No new high-priority papers for >= 2 rounds (with min round guard)
    if state.current_round >= 2:
        # If the last 2 rounds added 0 high-priority papers
        # We check via round_summaries length vs high_priority growth
        # Simple heuristic: if current_round >= 2 and high_priority hasn't grown
        # This is checked in runner by comparing before/after each round
        pass

    return False, ""

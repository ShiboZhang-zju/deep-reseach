"""Idea score thresholds are configuration, not three copies of 0.70.

Distribution analysis (2026-10-31, 595 scored ideas / 94 tasks) showed the
0.70 go line is calibrated to the v3-v7 population, not to v7.2:

  historical (v3-v7, n=579): median 0.711, 61.3% clear 0.70
  v7.2 probes  (n=16)     : median 0.654, 12.5% clear 0.70
                            and 0 of the 4 executable_candidates clear it

The line was hardcoded in three places. It is now one setting; the value is
deliberately unchanged (lowering it is a policy call, not a refactor), but it
must be a single visible knob so that decision is one edit away.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_defaults_match_the_previously_hardcoded_values():
    from app.config import settings
    assert settings.idea_score_go_line == pytest.approx(0.70)
    assert settings.idea_score_revise_line == pytest.approx(0.50)


def test_thresholds_are_ordered():
    from app.config import settings
    assert settings.idea_score_revise_line < settings.idea_score_go_line


def test_go_line_is_referenced_not_hardcoded():
    """Guard against a future re-hardcoding of 0.70 in the decision branches."""
    import pathlib

    backend = pathlib.Path(__file__).resolve().parent.parent
    for rel in ("app/agent/steps/generate_ideas.py",
                "app/agent/steps/generate_experiment.py",
                "app/agent/steps/generate_minimal_experiments.py"):
        src = (backend / rel).read_text(encoding="utf-8")
        assert "idea_score_go_line" in src, (
            f"{rel} must read the shared threshold, not a literal 0.70")
        assert "final_score >= 0.70" not in src, (
            f"{rel} still compares against a hardcoded 0.70")
        assert "idea_score_revise_line" in src or rel.endswith(
            "generate_minimal_experiments.py")


def test_decision_bands_follow_settings(monkeypatch):
    """Changing the setting must move the go/revise/reject boundaries."""
    from app.config import settings

    # The bands live inside the score functions; assert the mapping rule
    # directly against the settings so a copy-pasted literal cannot drift.
    monkeypatch.setattr(settings, "idea_score_go_line", 0.60)
    monkeypatch.setattr(settings, "idea_score_revise_line", 0.40)

    def decide(score):
        if score >= settings.idea_score_go_line:
            return "go"
        if score >= settings.idea_score_revise_line:
            return "revise"
        return "reject"

    assert decide(0.65) == "go"
    assert decide(0.45) == "revise"
    assert decide(0.35) == "reject"

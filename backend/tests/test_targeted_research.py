"""Tests for O2 targeted remediation decision logic (no LLM / no network)."""

from app.agent.state import ResearchState
from app.agent.steps.targeted_research import (
    can_remediate,
    _reason_key,
    _build_seed_queries,
    REMEDIABLE_REASONS,
    _REASON_PLAYBOOK,
)
from app.config import settings


def test_reason_key_normalizes_prefixed_reasons():
    assert _reason_key("no_evidence_backed_gap_candidates") == "no_evidence_backed_gap_candidates"
    # readiness reason is emitted with a suffix — must still resolve
    assert _reason_key("readiness_more_research: some detail") == "readiness_more_research"
    assert _reason_key("totally_unknown_reason") is None
    assert _reason_key("") is None


def test_all_remediable_reasons_have_playbook():
    for reason in REMEDIABLE_REASONS:
        assert reason in _REASON_PLAYBOOK
        intent, templates = _REASON_PLAYBOOK[reason]
        assert isinstance(intent, str) and intent
        assert templates and all("{topic}" in t or t for t in templates)


def test_can_remediate_respects_per_reason_limit():
    state = ResearchState(task_id="t", normalized_topic="agent memory")
    reason = "no_evidence_backed_gap_candidates"
    # fresh state — allowed
    assert can_remediate(state, reason) is True
    # exhaust per-reason budget
    state.remediation_attempts = {reason: settings.max_remediation_attempts}
    assert can_remediate(state, reason) is False


def test_can_remediate_respects_global_budget():
    state = ResearchState(task_id="t", normalized_topic="x")
    reason = "no_surviving_gap_after_audit"
    state.remediation_attempts = {"__total__": settings.max_remediation_rounds_total}
    assert can_remediate(state, reason) is False


def test_can_remediate_rejects_unknown_reason():
    state = ResearchState(task_id="t", normalized_topic="x")
    assert can_remediate(state, "identical_error_streak (boom)") is False


def test_build_seed_queries_injects_topic():
    seeds = _build_seed_queries("no_evidence_backed_gap_candidates", "graph neural networks")
    assert any("graph neural networks" in s for s in seeds)
    assert any("limitation" in s.lower() for s in seeds)


def test_remediation_disabled_when_attempts_zero(monkeypatch):
    monkeypatch.setattr(settings, "max_remediation_attempts", 0)
    state = ResearchState(task_id="t", normalized_topic="x")
    try:
        assert can_remediate(state, "no_evidence_backed_gap_candidates") is False
    finally:
        monkeypatch.undo()

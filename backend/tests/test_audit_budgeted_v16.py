"""Tests for v16 Budgeted Falsification mode (flag-gated, default OFF).

The budgeted audit reuses the v14 machinery under hard caps; these tests pin
the cap arithmetic and the honest timeout outcome (budget exhausted = gap
closed as unproven, never re-entering the loop).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_audit_gaps import _seed_gap, temp_db  # noqa: E402,F401


def test_audit_budget_legacy_values(temp_db, monkeypatch):
    from app.agent.steps.audit_gaps import _audit_budget

    monkeypatch.setattr("app.config.settings.gap_audit_budgeted", False)
    budget = _audit_budget()
    assert budget["sources"] is None
    assert budget["queries"] == 12 and budget["candidates"] == 20
    assert budget["neighbors"] == 5 and budget["fulltext"] == 5


def test_audit_budget_hard_caps(temp_db, monkeypatch):
    from app.agent.steps.audit_gaps import _audit_budget

    monkeypatch.setattr("app.config.settings.gap_audit_budgeted", True)
    budget = _audit_budget()
    assert budget["queries"] == 4 and budget["candidates"] == 10
    assert budget["neighbors"] == 3 and budget["fulltext"] == 2
    assert budget["killer_queries"] == 3
    assert budget["sources"] == ["semantic_scholar", "openalex", "arxiv"]


def test_timeout_budgeted_closes_gap_as_inconclusive(temp_db, monkeypatch):
    """v16: a wall-clock timeout closes the gap as unproven — terminal status
    `inconclusive` (budget ran out), never `rejected` (disproven), and the
    outcome is traced like the legacy timeout is."""
    from app.agent.steps.audit_gaps import _record_audit_timeout
    from app.db.models import AgentTrace
    from app.db.repositories import gap_repo

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    monkeypatch.setattr("app.config.settings.gap_audit_budgeted", True)
    result = _record_audit_timeout(db, gap.task_id, gap, audit_round=1)
    db.commit()

    assert result.recommended_action == "reject"
    assert gap.status == "inconclusive"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert audit.recommended_action == "reject"
    assert "audit_budget_exhausted" in (audit.rejection_reason or "")
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == gap.task_id,
        AgentTrace.step_name == "gap_audit_timeout").one()
    assert "budgeted" in (trace.output_json or "")
    db.close()


@pytest.mark.asyncio
async def test_killer_search_respects_budgeted_sources(temp_db, monkeypatch):
    """v16: the killer round must run under the budget's source whitelist —
    without it the killer phase queried every source (benchmark family
    included) even though the main audit restricted itself to the budget's
    three sources."""
    from types import SimpleNamespace

    from app.config import settings
    from app.agent.steps import audit_gaps

    db = temp_db()
    _, gap, _ = _seed_gap(db)

    captured: dict = {}

    async def fake_search(db_, state, executions, task_id, round_, **kwargs):
        captured["kwargs"] = kwargs
        return {}, [], []

    monkeypatch.setattr(audit_gaps, "search_and_save_papers", fake_search)

    decision = SimpleNamespace(
        killer_query_terms=["contamination velocity"],
        killer_found=False,
        closest_killer_work="A paper that kills this gap",
        residual_uncertainty="",
        audit_result="confirmed",
    )

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    await audit_gaps._run_killer_search(db, None, gap.task_id, gap, decision, [], 1)
    assert captured["kwargs"] == {
        "allowed_sources": ["semantic_scholar", "openalex", "arxiv"]}

    monkeypatch.setattr(settings, "gap_audit_budgeted", False)
    await audit_gaps._run_killer_search(db, None, gap.task_id, gap, decision, [], 1)
    # Legacy must not pass the kwarg at all (zero signature pressure on the
    # existing FakeSearchService doubles).
    assert captured["kwargs"] == {}
    db.close()


def test_timeout_legacy_keeps_more_search(temp_db, monkeypatch):
    from app.agent.steps.audit_gaps import _record_audit_timeout
    from app.db.repositories import gap_repo

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    monkeypatch.setattr("app.config.settings.gap_audit_budgeted", False)
    result = _record_audit_timeout(db, gap.task_id, gap, audit_round=1)
    db.commit()

    assert result.recommended_action == "more_search"
    assert gap.status == "auditing"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert audit.recommended_action == "more_search"

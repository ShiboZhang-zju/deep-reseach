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


def test_timeout_budgeted_closes_gap_as_unproven(temp_db, monkeypatch):
    from app.agent.steps.audit_gaps import _record_audit_timeout
    from app.db.repositories import gap_repo

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    monkeypatch.setattr("app.config.settings.gap_audit_budgeted", True)
    result = _record_audit_timeout(db, gap.task_id, gap, audit_round=1)
    db.commit()

    assert result.recommended_action == "reject"
    assert gap.status == "rejected"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert audit.recommended_action == "reject"
    assert "audit_budget_exhausted" in (audit.rejection_reason or "")


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

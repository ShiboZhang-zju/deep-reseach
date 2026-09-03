"""Pytest configuration — isolate tests from the development database.

`AUTO_CREATE_SCHEMA` lets `app.main` create tables for tests.

`DATABASE_URL` is pinned to a throwaway file BEFORE anything imports
`app.config`, so that importing the app (e.g. `from app.main import app` in the
API tests) can never touch `backend/deep_research.db`. Without this, running the
suite from `backend/` hits the real database and can interrupt a task that is
currently running.
"""
import os
import tempfile

os.environ["AUTO_CREATE_SCHEMA"] = "true"

_test_db = os.path.join(tempfile.gettempdir(), "deep_research_pytest.db")
os.environ["DATABASE_URL"] = "sqlite:///" + _test_db.replace("\\", "/")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_audit_mode_flags(monkeypatch):
    """Pin audit-mode flags to legacy defaults for every test.

    The developer .env may carry GAP_AUDIT_BUDGETED/GAP_AUDIT_PROGRESSIVE=true
    for live runs; without this pin, every legacy-behavior test silently runs
    under the new semantics (observed 2026-09-03: 8 failures after activating
    budgeted mode). Mode-specific tests override these via their own
    monkeypatch.setattr calls, which apply after this fixture.
    """
    from app.config import settings
    from app.agent.steps import audit_gaps

    monkeypatch.setattr(settings, "gap_audit_budgeted", False)
    monkeypatch.setattr(settings, "gap_audit_progressive", False)
    # GAP_SEARCH_POLICY_VERSION is bound at import time from .env — rebind it
    # so version-stamping assertions are independent of the live flag state.
    monkeypatch.setattr(audit_gaps, "GAP_SEARCH_POLICY_VERSION",
                        "gap-search-admission-v14")

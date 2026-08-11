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

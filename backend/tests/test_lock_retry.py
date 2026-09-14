"""Lock-retry contract regression tests.

These pin the behaviour that the 2026-09-14 batch failure exposed: SQLite
``database is locked`` is a TRANSIENT condition on the agent path (the lock can
be held past ``busy_timeout`` by a sibling task's write-then-await transaction
or by an OS-level scanner reading the DB file), so every write must survive it
instead of failing the task.

History the tests encode:

* 2026-09-10  task ea57b3c0 lost 5 papers when the evidence INSERT hit the lock
* 2026-09-14  tasks 60607a47 / 10c75f49 died on
              ``INSERT INTO search_query_records`` — a path that had no retry
              because lock handling had been added ad hoc, per function

The guarded invariant is: a lock error retries and eventually succeeds, while a
NON-lock OperationalError still fails fast so real bugs are not masked.
"""

import sqlite3

import pytest
from sqlalchemy.exc import OperationalError

from app.db.lock_retry import (
    commit_with_retry,
    flush_with_retry,
    is_lock_error,
    retry_on_locked,
)


def _locked_error() -> OperationalError:
    """Build a real OperationalError carrying the SQLite lock wording."""
    orig = sqlite3.OperationalError("database is locked")
    return OperationalError("INSERT INTO t VALUES (1)", {}, orig)


def _non_lock_error() -> OperationalError:
    orig = sqlite3.OperationalError("no such table: t")
    return OperationalError("INSERT INTO t VALUES (1)", {}, orig)


class _FakeSession:
    """Minimal Session stand-in recording the call sequence.

    ``flush_errors`` is consumed one entry per flush call; ``None`` means the
    flush succeeds, an exception instance means that call raises.
    """

    def __init__(self, flush_errors=None, commit_errors=None):
        self.flush_errors = list(flush_errors or [])
        self.commit_errors = list(commit_errors or [])
        self.flushes = 0
        self.commits = 0
        self.rollbacks = 0

    def flush(self):
        self.flushes += 1
        if self.flush_errors:
            exc = self.flush_errors.pop(0)
            if exc is not None:
                raise exc

    def commit(self):
        self.commits += 1
        if self.commit_errors:
            exc = self.commit_errors.pop(0)
            if exc is not None:
                raise exc

    def rollback(self):
        self.rollbacks += 1


class TestIsLockError:
    def test_matches_sqlite_lock_wording(self):
        assert is_lock_error(_locked_error())

    def test_rejects_unrelated_operational_error(self):
        assert not is_lock_error(_non_lock_error())

    def test_rejects_non_operational_error(self):
        assert not is_lock_error(ValueError("database is locked"))


class TestFlushWithRetry:
    def test_retries_until_lock_clears(self):
        db = _FakeSession(flush_errors=[_locked_error(), _locked_error(), None])
        flush_with_retry(db)
        assert db.flushes == 3
        # A retry MUST roll back first: the failed flush left the transaction
        # unusable, and without the rollback every subsequent call would raise
        # PendingRollbackError instead of re-attempting the write.
        assert db.rollbacks == 2

    def test_non_lock_error_propagates_without_retry(self):
        db = _FakeSession(flush_errors=[_non_lock_error()])
        with pytest.raises(OperationalError, match="no such table"):
            flush_with_retry(db)
        assert db.flushes == 1
        assert db.rollbacks == 0

    def test_exhausted_retries_reraise_lock_error(self):
        # max_attempts=3 -> the third lock error is surfaced rather than looping
        # forever against a permanently held lock.
        db = _FakeSession(flush_errors=[_locked_error()] * 3)
        with pytest.raises(OperationalError, match="database is locked"):
            flush_with_retry(db, max_attempts=3)
        assert db.flushes == 3


class TestCommitWithRetry:
    def test_retries_until_lock_clears(self):
        db = _FakeSession(commit_errors=[_locked_error(), None])
        commit_with_retry(db)
        assert db.commits == 2
        assert db.rollbacks == 1

    def test_non_lock_error_propagates_without_retry(self):
        db = _FakeSession(commit_errors=[_non_lock_error()])
        with pytest.raises(OperationalError, match="no such table"):
            commit_with_retry(db)
        assert db.commits == 1


class TestRetryOnLocked:
    def test_wrapped_function_is_reinvoked_after_lock(self):
        db = _FakeSession()
        calls = []

        @retry_on_locked
        def write(db):
            calls.append(1)
            if len(calls) < 3:
                raise _locked_error()
            return "done"

        assert write(db) == "done"
        assert len(calls) == 3
        assert db.rollbacks == 2

    def test_wrapped_function_reraises_non_lock_error(self):
        db = _FakeSession()

        @retry_on_locked
        def write(db):
            raise _non_lock_error()

        with pytest.raises(OperationalError, match="no such table"):
            write(db)


class TestSearchQueryInsertIsGuarded:
    """The exact path that killed the 2026-09-14 batch tasks.

    ``save_search_query`` previously did a bare ``db.flush()``, so one locked
    INSERT lost the whole round and the task. This test drives the real
    repository function against a session whose flush raises the lock error
    once, and asserts the write still lands.
    """

    def test_lock_then_success_returns_record(self):
        from app.db.repositories import search_query_repo

        db = _FakeSession(flush_errors=[_locked_error(), None])
        # get the query().filter()...first() chain to report "no existing row"
        recorded = {}

        class _Query:
            def filter(self, *a, **k):
                return self

            def first(self):
                return None

        db.query = lambda *a, **k: _Query()
        db.add = lambda obj: recorded.setdefault("obj", obj)

        result = search_query_repo.save_search_query(
            db, "task-1", "query text", "intent", None, None, 1,
        )
        assert result is recorded["obj"]
        assert db.flushes == 2, "the locked flush must have been retried"


class TestPhaseConvergence:
    """A failed round must leave no phase advertising itself as in-flight."""

    def test_mark_running_phases_failed_scopes_by_round(self):
        import os
        import tempfile

        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.db.models import PhaseRun, ResearchTask
        from app.db.repositories import phase_repo

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        command.upgrade(cfg, "head")
        # Bind a session to the TEMP db explicitly. SessionLocal points at the
        # project database, so using it here would write into (and collide
        # with) real task rows.
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        session = sessionmaker(bind=engine)()
        try:
            session.add(ResearchTask(id="t-converge", user_input="x", status="searching"))
            session.flush()
            for name, rnd in (("search_round_1", 1), ("search_round_2", 2),
                              ("summarize_round_2", 2), ("clarify", None)):
                session.add(PhaseRun(task_id="t-converge", phase_name=name,
                                     status="running", round_number=rnd))
            session.commit()

            n = phase_repo.mark_running_phases_failed(
                session, "t-converge", "boom", round_number=2)
            session.commit()

            assert n == 2, "only the two round-2 phases should be touched"
            statuses = {
                p.phase_name: p.status
                for p in session.query(PhaseRun).filter(
                    PhaseRun.task_id == "t-converge").all()
            }
            assert statuses["search_round_2"] == "failed"
            assert statuses["summarize_round_2"] == "failed"
            # A different round and a task-level phase (round_number is NULL)
            # must survive, otherwise a round rollback would clobber unrelated
            # in-flight work.
            assert statuses["search_round_1"] == "running"
            assert statuses["clarify"] == "running"
        finally:
            session.close()
            engine.dispose()
            os.unlink(db_path)

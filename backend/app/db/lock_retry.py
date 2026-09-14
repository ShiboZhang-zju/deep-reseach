"""SQLite write-lock retry contract.

Background
----------
The agent pipeline runs multiple tasks concurrently against one SQLite file in
WAL mode. Writes are short (single-row INSERT/UPDATE), but the write lock can
still be held past ``busy_timeout`` (10s) by:

* a sibling task's long "write-then-await" transaction (a COMMIT that is only
  reached after an ``await`` keeps the lock for the whole LLM round trip), and
* OS-level holders we do not control at all — Windows filesystem scanners /
  antivirus reading the DB file, and WAL checkpoint contention.

Both are *transient* conditions in the sense that the operation succeeds once
the lock is released. They are NOT logical failures.

History (why this module exists)
--------------------------------
Lock handling was previously implemented ad hoc, per repository function, and
that predictably failed to cover every write path:

* 2026-09-07  task 789bec4a died on a bare ``UPDATE research_tasks``
              -> ``task_repo._standalone_status_write`` added
* 2026-09-09/10 two batch topics died on the ``state_json`` UPDATE
              -> ``task_repo.save_state`` inline retry added
* 2026-09-10  task ea57b3c0 lost 5 papers to a locked evidence INSERT
              -> ``extract_evidence._extract_from_sections`` inline retry added
* 2026-09-14  tasks 60607a47 / 10c75f49 (batch resume) died on
              ``INSERT INTO search_query_records`` in ``save_search_query``
              -> the fourth distinct location; the pattern is now obvious.

Rather than adding a fifth inline retry, every write on the agent path goes
through the single contract below. ``db.flush``/``db.commit`` under
``retry_on_locked`` is idempotent-safe because the helper restores the session
to a usable state (``rollback()``) and the caller re-issues the same writes.
"""

import asyncio
import logging
import random
import time
from functools import wraps

from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Backoff schedule for synchronous callers. The first two delays are short
# because most contention is a <100ms collision with a sibling's per-query
# commit; the long tail covers the multi-second-to-minute storms (OS-level
# holders, sibling write-then-await transactions).
SYNC_BACKOFF = (0.15, 0.4, 1.0, 2.5, 5.0, 10.0, 15.0)
# Async callers can afford to wait longer without holding a thread.
ASYNC_BACKOFF = (0.2, 0.5, 1.2, 3.0, 6.0, 12.0, 20.0)


def is_lock_error(exc: BaseException) -> bool:
    """True when the exception is a SQLite/Postgres-style lock contention.

    Matches both ``sqlite3.OperationalError: database is locked`` (wrapped by
    SQLAlchemy as ``OperationalError``) and lock-timeout wording, but NOT
    unrelated operational errors such as 'no such table' or a disk I/O error —
    those must keep failing fast so real bugs stay visible.
    """
    if not isinstance(exc, OperationalError):
        return False
    text = str(exc).lower()
    return "database is locked" in text or "lock timeout" in text or "database table is locked" in text


def _sleep_for(backoff: tuple[float, ...], attempt: int) -> float:
    """Delay before retry ``attempt`` (0-based), with jitter.

    Jitter matters under N-way concurrency: without it the contending tasks
    wake in lockstep and re-collide, turning a short storm into a long one.
    Capped at +25% so the schedule stays bounded.
    """
    base = backoff[min(attempt, len(backoff) - 1)]
    return base * (1.0 + random.random() * 0.25)


def retry_on_locked(func):
    """Run a sync session-writing callable with lock retries.

    The wrapped callable receives ``db`` as its first positional argument and
    must be safe to re-invoke after a ``rollback()`` — i.e. it must rebuild its
    pending writes rather than rely on objects added before the call. Every
    helper in ``app/db/repositories`` is written that way.

    On a non-lock ``OperationalError`` the exception propagates unchanged.
    """

    @wraps(func)
    def wrapper(db, *args, **kwargs):
        for attempt in range(len(SYNC_BACKOFF) + 1):
            try:
                return func(db, *args, **kwargs)
            except OperationalError as exc:
                if not is_lock_error(exc) or attempt == len(SYNC_BACKOFF):
                    raise
                db.rollback()
                delay = _sleep_for(SYNC_BACKOFF, attempt)
                logger.warning(
                    "db lock contention in %s (attempt %d/%d), retrying in %.2fs",
                    func.__name__, attempt + 1, len(SYNC_BACKOFF) + 1, delay,
                )
                time.sleep(delay)

    return wrapper


def retry_on_locked_async(func):
    """Async variant for callables that may ``await`` between writes.

    ``asyncio.sleep`` releases the event loop so a lock wait does not stall
    every other request/agent coroutine (the reason ``busy_timeout`` is kept
    at 10s: the synchronous wait inside the SQLite driver blocks the loop).
    """

    @wraps(func)
    async def wrapper(db, *args, **kwargs):
        for attempt in range(len(ASYNC_BACKOFF) + 1):
            try:
                return await func(db, *args, **kwargs)
            except OperationalError as exc:
                if not is_lock_error(exc) or attempt == len(ASYNC_BACKOFF):
                    raise
                db.rollback()
                delay = _sleep_for(ASYNC_BACKOFF, attempt)
                logger.warning(
                    "db lock contention in %s (attempt %d/%d), retrying in %.2fs",
                    func.__name__, attempt + 1, len(ASYNC_BACKOFF) + 1, delay,
                )
                await asyncio.sleep(delay)

    return wrapper


def flush_with_retry(db, *, max_attempts: int | None = None) -> None:
    """Flush the caller's pending writes, retrying on lock contention.

    Rolls back and re-flushes, so the caller's already-``add()``ed objects are
    still pending afterwards (SQLAlchemy restores them to ``pending`` on
    rollback) and the retry re-issues the same INSERT/UPDATE batch. Use this
    for code paths that mutate ORM objects across a wide body and cannot be
    wrapped in a decorator.
    """
    attempts = max_attempts if max_attempts is not None else len(SYNC_BACKOFF) + 1
    for attempt in range(attempts):
        try:
            db.flush()
            return
        except OperationalError as exc:
            if not is_lock_error(exc) or attempt == attempts - 1:
                raise
            db.rollback()
            time.sleep(_sleep_for(SYNC_BACKOFF, attempt))


def commit_with_retry(db, *, max_attempts: int | None = None) -> None:
    """Commit the caller's transaction, retrying on lock contention.

    Same contract as :func:`flush_with_retry` for call sites that commit
    directly. ``rollback()`` before retrying guarantees a fresh transaction
    (and thus a fresh shot at the write lock) instead of re-committing a
    poisoned one.
    """
    attempts = max_attempts if max_attempts is not None else len(SYNC_BACKOFF) + 1
    for attempt in range(attempts):
        try:
            db.commit()
            return
        except OperationalError as exc:
            if not is_lock_error(exc) or attempt == attempts - 1:
                raise
            db.rollback()
            time.sleep(_sleep_for(SYNC_BACKOFF, attempt))

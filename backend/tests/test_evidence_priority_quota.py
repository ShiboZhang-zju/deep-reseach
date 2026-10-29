"""Extraction must cover the whole candidate corpus, and must stay cheap to re-enter.

Two allocation rules used to fight each other and both are now gone:

1. A per-round `evidence_max_papers` cap. With 146 candidates and a cap of 15, a
   five-round task could assess at most 75 papers — the corpus was unreachable by
   construction, and *which* papers got read depended on how many rounds happened
   to run.
2. A priority-tier quota layered on top. It correctly stopped mid-tier papers
   from crowding out topic-critical ones, but it did so by dropping papers rather
   than by making extraction cheaper. In a measured run the papers that never got
   extracted were exactly the core topic papers (Dynamic Speculation Lookahead,
   DySpec, OPT-Tree, EdgeSpec, CLaSp, Hierarchical Speculative Decoding), and 5
   of the 8 were downloadable on the spot — a pure allocation loss.

What makes full coverage affordable is the shared concurrency pool, plus the fact
that a paper already extracted from its full text is skipped. These tests pin all
three properties, because any one of them regressing silently reintroduces
truncation.
"""

import inspect
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    """Alembic-migrated temp DB, matching the project's other test modules."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield engine, sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _source():
    from app.agent.steps import extract_evidence

    return inspect.getsource(extract_evidence.extract_evidence_units)


def test_no_per_round_paper_cap():
    """The phase must not limit how many candidates it reads."""
    src = _source()

    assert "settings.evidence_max_papers" not in src, \
        "a per-round cap makes the corpus unreachable and drops high-priority papers"
    assert "budget" not in src, "quota allocation must be gone, not merely unused"


def test_priority_tiers_are_not_dropped_by_an_allocation_rule():
    """Both tiers must be queried without a LIMIT applied to the merged set."""
    src = _source()

    assert 'TaskPaper.priority.in_(["high", "medium"])' in src
    # A `.limit(` on the candidate query would reintroduce truncation.
    assert ".limit(" not in src, "candidate selection must not be truncated"


def test_a_single_shared_pool_replaces_per_batch_semaphores():
    """A per-batch semaphore with a barrier between batches serialises the round.

    The old shape created `Semaphore(batch_size)` inside the loop and awaited the
    batch before starting the next, so concurrency never exceeded batch_size and
    duration scaled as ceil(N/batch) x per-paper-time. One pool keeps exactly
    `evidence_extract_concurrency` papers in flight continuously, which is what
    makes full-corpus coverage affordable.
    """
    src = _source()

    assert "evidence_extract_concurrency" in src
    assert "Semaphore(concurrency)" in src
    # The barrier is what must not come back.
    assert "for batch_start in range(" not in src, \
        "a batch barrier reintroduces serialisation"


def test_reentry_skip_is_keyed_on_recorded_evidence_not_a_provenance_flag():
    """Full coverage must not mean full cost on every re-entry.

    The skip must consult *recorded evidence*, not `extraction_status`. Keying it
    on the flag breaks the moment the same PDF is reached through a second task:
    a PDF belongs to a Paper, but the flag lives on a per-task TaskPaper, so the
    second task sees a fresh NULL and re-extracts everything. Measured: 25% of
    papers with evidence had been re-extracted by another task, 3,408 (task,
    paper) pairs were queued for a repeat, and 2,694 of those PDFs were already
    on disk.
    """
    from app.agent.steps import extract_evidence

    src = inspect.getsource(extract_evidence._extract_from_paper)
    assert "_has_usable_evidence" in src, \
        "the skip must be content-keyed, not flag-keyed"
    assert 'extraction_status", None) == "full_text"' not in src, \
        "flag-keyed skip re-extracts a paper reached through a second task"


def test_only_span_backed_units_count_as_already_extracted():
    """A degraded result must not be frozen in place by the skip.

    `abstract_only` means an abstract-level result; a later round holding the PDF
    should still be free to upgrade it. `unverified` means the span could not be
    located — a failed extraction that must be retried.
    """
    from app.agent.steps import extract_evidence

    assert set(extract_evidence._USABLE_VERIFICATION_STATUSES) == {"verified", "upgraded"}
    assert "abstract_only" not in extract_evidence._USABLE_VERIFICATION_STATUSES
    assert "unverified" not in extract_evidence._USABLE_VERIFICATION_STATUSES


def test_skip_lookup_degrades_to_not_extracted():
    """A lookup failure may cost time, never silently drop a paper."""
    from app.agent.steps import extract_evidence

    class _Boom:
        def query(self, *a, **k):
            raise RuntimeError("db exploded")

    assert extract_evidence._has_usable_evidence(_Boom(), "t", "p") is False


def test_skip_lookup_returns_true_for_recorded_evidence(temp_db):
    """The lookup must actually find recorded evidence.

    This guards a real bug: an earlier version of this helper was correct against
    an ORM Session but silently returned False against a raw connection, because
    it relied on a query API the connection did not have. The failure mode was
    invisible — the skip simply never fired, so every paper was re-extracted at
    full cost while all tests still passed. Asserting the positive case is the
    only way to catch that class of bug.
    """
    from app.agent.steps import extract_evidence
    from app.db.models import EvidenceUnit, Paper, ResearchTask

    engine, Session = temp_db
    session = Session()
    task = ResearchTask(id="t-skip-pos", user_input="x", status="extracting_evidence")
    paper = Paper(id="p-skip-pos", title="t", normalized_title="t", title_hash="h")
    session.add_all([task, paper])
    session.flush()
    session.add(EvidenceUnit(
        id="eu-skip-pos", task_id=task.id, paper_id=paper.id,
        evidence_type="result", normalized_claim="c", original_span="s",
        verification_status="verified",
    ))
    session.commit()

    assert extract_evidence._has_usable_evidence(session, task.id, paper.id) is True

    session.close()


def test_skip_lookup_ignores_unverified_and_abstract_only(temp_db):
    """Degraded extractions must not be frozen in place by the skip."""
    from app.agent.steps import extract_evidence
    from app.db.models import EvidenceUnit, Paper, ResearchTask

    engine, Session = temp_db
    session = Session()
    task = ResearchTask(id="t-skip-neg", user_input="x", status="extracting_evidence")
    paper = Paper(id="p-skip-neg", title="t", normalized_title="t", title_hash="h")
    session.add_all([task, paper])
    session.flush()
    for idx, status in enumerate(("unverified", "abstract_only")):
        session.add(EvidenceUnit(
            id="eu-neg-{}".format(idx), task_id=task.id, paper_id=paper.id,
            evidence_type="result", normalized_claim="c", original_span="s",
            verification_status=status,
        ))
    session.commit()

    assert extract_evidence._has_usable_evidence(session, task.id, paper.id) is False

    session.close()


def test_skip_lookup_uses_the_worker_session_when_none_is_passed():
    """The per-paper worker runs on its own session, so db may be None."""
    import inspect as _inspect
    from app.agent.steps import extract_evidence

    src = _inspect.getsource(extract_evidence._has_usable_evidence)
    assert "SessionLocal()" in src


@pytest.mark.asyncio
async def test_already_extracted_paper_is_skipped_without_touching_the_llm(monkeypatch):
    """The skip must short-circuit before any PDF parse or LLM call."""
    from app.agent.steps import extract_evidence

    class _Paper:
        id = "p-skip"

    monkeypatch.setattr(extract_evidence, "_has_usable_evidence", lambda *a, **k: True)

    # No llm, no db, no pdf_path: if the skip did not fire first this would raise.
    result = await extract_evidence._extract_from_paper(
        None, None, "task-skip", _Paper(), _TaskPaper(), None, 1)

    assert result == 0


@pytest.mark.asyncio
async def test_paper_without_recorded_evidence_is_re_extracted(monkeypatch):
    """The complement of the skip: nothing recorded means the paper must be read."""
    from app.agent.steps import extract_evidence

    class _Paper:
        id = "p-fresh"
        abstract = ""

    monkeypatch.setattr(extract_evidence, "_has_usable_evidence", lambda *a, **k: False)

    tp = _TaskPaper(extraction_status="full_text", extraction_gap_reason="stale")
    extract_evidence._download_diagnoses.pop("task-fresh", None)

    await extract_evidence._extract_from_paper(
        _NoopDB(), None, "task-fresh", _Paper(), tp, None, 1)

    # Re-assessed means the provenance write ran and overwrote the stale reason.
    assert tp.extraction_gap_reason != "stale", \
        "a paper with no recorded evidence must be extracted, not skipped"


class _NoopDB:
    """Minimal stand-in so the provenance write has somewhere to land."""

    def add(self, _obj):
        return None

    def flush(self):
        return None

    def commit(self):
        return None


class _TaskPaper:
    """TaskPaper stand-in with per-instance state.

    Must be an instance rather than a class with class attributes: the code under
    test assigns to these fields, and a class attribute would be shadowed by the
    instance write while the assertion still read the untouched class value.
    """

    def __init__(self, **kwargs):
        self.id = "tp-test"
        self.extraction_status = kwargs.get("extraction_status")
        self.extraction_gap_reason = kwargs.get("extraction_gap_reason")


def test_concurrency_is_configurable():
    from app.config import settings

    assert hasattr(settings, "evidence_extract_concurrency")
    assert settings.evidence_extract_concurrency >= 1


def test_skip_is_scoped_to_the_task_that_owns_the_evidence():
    """Plan A: evidence stays owned by the task that produced it.

    Widening the skip to "another task already read this paper" looks like a free
    dedup win but is a correctness bug: coverage counts this task's own units, so
    skipping would leave the task with zero rows for that paper — indistinguishable
    from having found nothing. That is the absence-of-evidence error wearing a new
    hat. Cross-task reuse requires giving evidence shared ownership first.
    """
    from app.agent.steps import extract_evidence

    src = inspect.getsource(extract_evidence._has_usable_evidence)
    assert "EvidenceUnit.task_id == task_id" in src, \
        "the skip must stay scoped to the owning task"
    assert "EvidenceUnit.paper_id == paper_id" in src


def test_chunk_dedup_is_keyed_on_the_chunk_not_the_claim():
    """A chunk legitimately yields many claims; dedup must not collapse them.

    Measured: all 6,437 multi-row (task, paper, chunk) groups in a production DB
    held distinct claims, and a single chunk produced 29 of them (a related-work
    paragraph enumerating baselines). Keying dedup on `normalized_claim` would
    silently reduce an enumerating chunk to one row.
    """
    from app.agent.steps import extract_evidence

    src = inspect.getsource(extract_evidence._extract_from_sections)
    assert "EvidenceUnit.source_chunk_hash == chunk_hash" in src
    assert "normalized_claim ==" not in src, \
        "deduping on the claim collapses multi-claim chunks"

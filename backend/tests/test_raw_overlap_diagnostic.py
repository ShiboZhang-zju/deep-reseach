"""P1-1 raw recall diagnostic: persistence + waterfall + inconclusive finalization."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.paper_sources.base import RawPaper


class _FakeSearch:
    """Returns two papers per query from a single 'fake' source."""

    def __init__(self):
        self.n = 0

    async def search_multiple_queries(self, queries, limit=15):
        papers = []
        for q in queries:
            self.n += 1
            for i in range(2):
                papers.append(RawPaper(
                    title=f"Paper {self.n}_{i}: {q[:20]}",
                    abstract="memory token budget evaluation with GNN attention",
                    year=2024, venue="ICML",
                    doi=f"10.1000/fake{self.n}_{i}",
                    source="fake", raw_data={},
                ))
        return papers


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
    cfg = AlembicConfig()
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    alembic_command.upgrade(cfg, "head")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    yield engine, Session


def _make_task(db, Session):
    from app.db.models import ResearchTask
    from app.agent.state import ResearchState
    task = ResearchTask(user_input="test", status="pending")
    state = ResearchState(task_id=task.id, user_input="test", pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    return task.id


def test_raw_results_persisted_with_rank_and_canonical(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id = _make_task(db, Session)
    from app.agent.state import ResearchState
    state = ResearchState(task_id=task_id, user_input="test", pipeline_version=2)

    from app.agent.steps.generate_queries import SearchQueryExecution
    from app.agent.steps.search_papers import search_and_save_papers
    from app.db.models import SearchRawResult

    qe = [SearchQueryExecution(query_id="q1", query_text="query one",
                               intent="seminal", target_question_id="qq1",
                               expected_evidence_type="method")]

    async def _no_filter(papers, *a, **k):
        return papers

    with patch("app.agent.steps.search_papers.search_service", _FakeSearch()), \
         patch("app.agent.steps.search_papers._prefilter_by_similarity", new=_no_filter):
        import asyncio
        asyncio.run(search_and_save_papers(db, state, qe, task_id, round_num=1))

    rows = db.query(SearchRawResult).filter(SearchRawResult.query_id == "q1").order_by(SearchRawResult.raw_rank).all()
    assert len(rows) == 2
    assert [r.raw_rank for r in rows] == [0, 1]
    assert all(r.source == "fake" for r in rows)
    assert all(r.external_paper_id for r in rows)
    # Canonical paper id backfilled for the two surviving (pre-filtered) papers.
    assert all(r.canonical_paper_id is not None for r in rows)
    db.close()


def test_raw_results_idempotent_on_retry(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id = _make_task(db, Session)
    from app.agent.state import ResearchState
    state = ResearchState(task_id=task_id, user_input="test", pipeline_version=2)

    from app.agent.steps.generate_queries import SearchQueryExecution
    from app.agent.steps.search_papers import search_and_save_papers
    from app.db.models import SearchRawResult

    qe = [SearchQueryExecution(query_id="q1", query_text="query one",
                               intent="seminal", target_question_id="qq1",
                               expected_evidence_type="method")]

    async def _no_filter(papers, *a, **k):
        return papers

    with patch("app.agent.steps.search_papers.search_service", _FakeSearch()), \
         patch("app.agent.steps.search_papers._prefilter_by_similarity", new=_no_filter):
        import asyncio
        asyncio.run(search_and_save_papers(db, state, qe, task_id, round_num=1))
        # Retry the same query id (same SearchQueryRecord).
        asyncio.run(search_and_save_papers(db, state, qe, task_id, round_num=1))

    rows = db.query(SearchRawResult).filter(SearchRawResult.query_id == "q1").all()
    assert len(rows) == 2  # no duplicate raw rows across retries
    db.close()


def test_finalize_inconclusive_gaps(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id = _make_task(db, Session)

    from app.db.models import GapCandidate
    from app.db.repositories import gap_repo
    from app.agent.runner import _finalize_inconclusive_gaps

    gap = gap_repo.create_gap_candidate(
        db, task_id, "coverage_gap", "a candidate gap",
        claimed_delta="some delta", status="auditing",
    )
    # gap2: auditing but last audit was NOT more_search -> stays auditing.
    gap2 = gap_repo.create_gap_candidate(
        db, task_id, "coverage_gap", "another gap",
        claimed_delta="delta2", status="auditing",
    )
    gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=task_id, audit_result="uncertain",
        recommended_action="more_search", audit_round=1,
    )
    gap_repo.create_gap_audit(
        db, gap_id=gap2.id, task_id=task_id, audit_result="uncertain",
        recommended_action="narrow", audit_round=1,
    )
    db.commit()

    closed = _finalize_inconclusive_gaps(db, task_id)
    db.commit()

    assert closed == 1
    assert db.get(GapCandidate, gap.id).status == "inconclusive"
    assert db.get(GapCandidate, gap2.id).status == "auditing"
    db.close()


def test_overlap_waterfall_layers(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id = _make_task(db, Session)

    from app.db.models import (GapCandidate, SearchQueryRecord, SearchQueryPaper,
                               SearchRawResult)
    from app.db.repositories import gap_repo
    from app.agent.steps.audit_gaps import _compute_overlap_waterfall

    gap = gap_repo.create_gap_candidate(db, task_id, "coverage_gap", "gap")

    # Two variants in one family.
    q1 = SearchQueryRecord(id="q1", task_id=task_id, round_number=1, query_text="v1",
                           normalized_query_text="v1", intent="gap_exact_gap",
                           target_gap_id=gap.id, query_family="exact_gap")
    q2 = SearchQueryRecord(id="q2", task_id=task_id, round_number=1, query_text="v2",
                           normalized_query_text="v2", intent="gap_exact_gap",
                           target_gap_id=gap.id, query_family="exact_gap")
    db.add_all([q1, q2])

    # Raw: q1 = [A1(Shared Paper), B(Beta)], q2 = [A2(Shared Paper), C(Gamma)].
    # A1/A2 share a title so canonicalization merges them; raw external ids differ.
    raw = [
        SearchRawResult(query_id="q1", source="fake", raw_rank=0, external_paper_id="A1", title="Shared Paper"),
        SearchRawResult(query_id="q1", source="fake", raw_rank=1, external_paper_id="B", title="Beta"),
        SearchRawResult(query_id="q2", source="fake", raw_rank=0, external_paper_id="A2", title="Shared Paper"),
        SearchRawResult(query_id="q2", source="fake", raw_rank=1, external_paper_id="C", title="Gamma"),
    ]
    db.add_all(raw)

    # Post-filter: q1 -> {pShared, pBeta}, q2 -> {pShared, pGamma}.
    sqp = [
        SearchQueryPaper(query_id="q1", paper_id="pShared", rank=0, source="fake"),
        SearchQueryPaper(query_id="q1", paper_id="pBeta", rank=1, source="fake"),
        SearchQueryPaper(query_id="q2", paper_id="pShared", rank=0, source="fake"),
        SearchQueryPaper(query_id="q2", paper_id="pGamma", rank=1, source="fake"),
    ]
    db.add_all(sqp)
    db.commit()

    w = _compute_overlap_waterfall(db, gap, ["pShared", "pBeta"])

    fam = w["exact_gap"]
    # raw: {A1,B} vs {A2,C} -> empty intersection -> Jaccard 0.
    assert fam["raw"]["5"]["jaccard"] == 0.0
    assert fam["raw"]["5"]["overlap_coefficient"] == 0.0
    # canonical: {hash(Shared),hash(Beta)} vs {hash(Shared),hash(Gamma)} -> 1/3, 1/2.
    assert fam["canonical"]["5"]["jaccard"] == pytest.approx(1 / 3)
    assert fam["canonical"]["5"]["overlap_coefficient"] == pytest.approx(0.5)
    # post-filter: {pShared,pBeta} vs {pShared,pGamma} -> 1/3, 1/2.
    assert fam["post_filter"]["5"]["jaccard"] == pytest.approx(1 / 3)
    assert fam["post_filter"]["5"]["overlap_coefficient"] == pytest.approx(0.5)
    assert w["npa_candidate"]["neighbor_paper_ids"] == ["pShared", "pBeta"]
    db.close()

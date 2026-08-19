"""Cumulative NPA Top-K stability + final_score distribution diagnostics."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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


def _setup(db, Session):
    from app.db.models import ResearchTask, SearchQueryRecord, SearchQueryPaper, TaskPaper
    from app.db.repositories import gap_repo
    task = ResearchTask(user_input="test", status="pending")
    db.add(task)
    db.flush()
    gap = gap_repo.create_gap_candidate(db, task.id, "coverage_gap", "gap")

    # 3 queries: two share a family (variants), one in another family.
    qs = [
        SearchQueryRecord(id="q1", task_id=task.id, round_number=1, query_text="v1",
                          normalized_query_text="v1", intent="gap_exact_gap",
                          target_gap_id=gap.id, query_family="exact_gap"),
        SearchQueryRecord(id="q2", task_id=task.id, round_number=1, query_text="v2",
                          normalized_query_text="v2", intent="gap_exact_gap",
                          target_gap_id=gap.id, query_family="exact_gap"),
        SearchQueryRecord(id="q3", task_id=task.id, round_number=1, query_text="v3",
                          normalized_query_text="v3", intent="gap_synonym",
                          target_gap_id=gap.id, query_family="synonym"),
    ]
    db.add_all(qs)
    db.flush()

    # Paper A is recalled by q1+q2 (hits=2), B by q1, C by q3.
    mappings = [
        SearchQueryPaper(query_id="q1", paper_id="pA", rank=0, source="fake"),
        SearchQueryPaper(query_id="q2", paper_id="pA", rank=1, source="fake"),
        SearchQueryPaper(query_id="q1", paper_id="pB", rank=1, source="fake"),
        SearchQueryPaper(query_id="q3", paper_id="pC", rank=0, source="fake"),
    ]
    db.add_all(mappings)
    tps = [
        TaskPaper(task_id=task.id, paper_id="pA", discovered_round=1, final_score=0.9, priority="high"),
        TaskPaper(task_id=task.id, paper_id="pB", discovered_round=1, final_score=0.5, priority="medium"),
        TaskPaper(task_id=task.id, paper_id="pC", discovered_round=1, final_score=0.1, priority="low"),
    ]
    db.add_all(tps)
    db.commit()
    return task.id, gap.id


def test_rank_retrieved_papers_hits_dominate(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id, gap_id = _setup(db, Session)
    from app.db.models import GapCandidate
    from app.agent.steps.audit_gaps import _rank_retrieved_papers
    gap = db.get(GapCandidate, gap_id)
    ranked = _rank_retrieved_papers(db, gap, ["q1", "q2", "q3"])
    # pA (hits=2) must rank first; exact order of pB/pC depends on tiebreak.
    assert ranked[0] == "pA"
    assert set(ranked) == {"pA", "pB", "pC"}
    db.close()


def test_compute_cumulative_npa_stability(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id, gap_id = _setup(db, Session)
    from app.db.models import GapCandidate
    from app.agent.steps.audit_gaps import _compute_cumulative_npa_stability
    gap = db.get(GapCandidate, gap_id)
    out = _compute_cumulative_npa_stability(db, gap, ["q1", "q2", "q3"])
    assert out["n_queries"] == 3
    assert len(out["convergence_curve"]) == 3
    # step 1 -> 2 adds pA via q2 but pA already top via q1, so Top-K should be stable
    assert out["final_step_jaccard"] is not None
    assert 0.0 <= out["final_step_jaccard"] <= 1.0
    db.close()


def test_compute_final_score_distribution(temp_db):
    engine, Session = temp_db
    db = Session()
    task_id, gap_id = _setup(db, Session)
    from app.db.models import GapCandidate
    from app.agent.steps.audit_gaps import _compute_final_score_distribution
    gap = db.get(GapCandidate, gap_id)
    dist = _compute_final_score_distribution(db, gap, ["q1", "q2", "q3"])
    assert dist["count"] == 3
    assert dist["min"] == pytest.approx(0.1)
    assert dist["max"] == pytest.approx(0.9)
    assert dist["median"] == pytest.approx(0.5)
    db.close()

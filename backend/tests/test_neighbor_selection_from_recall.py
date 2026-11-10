"""Corpus-recall hits must be SELECTABLE as neighbours, not just poolable.

Regression for a defect found during the neighbour-budget experiment
(2026-11-03), on the first real audit trace:

    limit=8  selected=2  cap_bound=False  candidate_pool_size=10

The treatment arm was configured for 8 neighbours, its admission pool held 10
papers, and yet only 2 were selected — while the budget was not even binding.
Cause: `select_gap_specific_neighbors` ranked only papers that had
adversarial-search mappings, and its fallback walked
`collect_same_question_neighbors` (evidence links of the gap's OWN questions).
Corpus-recall hits appear in `candidate_paper_ids` (so admission counts them and
`candidate_pool_size` looks healthy) but were in neither source, so they could
never become neighbours.

That is the original 67.1% loss bucket relocated: recall moved killers from
"structurally unreachable" to "pooled but never compared", which changes the
trace without changing what the auditor reads. These tests fail on the old
behaviour and pin the fix.
"""

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
    cfg.set_main_option("script_location",
                        os.path.join(os.path.dirname(__file__), "..",
                                     "alembic_migrations"))
    alembic_command.upgrade(cfg, "head")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


def _seed(db, n_papers):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import (GapCandidate, Paper, ResearchContract, ResearchTask,
                              TaskPaper)

    task = ResearchTask(user_input="recall selectability", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="t",
                                input_hash="h-sel", version=1)
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="d", question_ids_json="[]", supporting_evidence_ids_json="[]",
        contradicting_evidence_ids_json="[]", mining_round=1, status="candidate",
        target_setting="x", observed_problem="y", existing_coverage="z",
        missing_capability="m", claimed_delta="c", testable_hypothesis="h",
        falsification_condition="f", provenance_status="complete",
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    papers = []
    for i in range(n_papers):
        p = Paper(title=f"corpus paper {i}", abstract="a",
                  normalized_title=f"corpus paper {i}")
        db.add(p)
        db.flush()
        db.add(TaskPaper(task_id=task.id, paper_id=p.id, discovered_round=1,
                         final_score=0.5))
        papers.append(p)
    db.commit()
    return task, gap, papers


def _add_relevance(db, gap, papers, scores):
    from app.agent.steps.audit_gaps import GapPaperRelevance
    from app.config import settings
    for paper, score in zip(papers, scores):
        db.add(GapPaperRelevance(
            gap_id=gap.id, paper_id=paper.id, task_id=gap.task_id,
            relevance_score=score, problem_overlap="yes", mechanism_overlap="no",
            evaluation_overlap="no", claim_overlap="no", rationale="r",
            scoring_version=settings.gap_relevance_scoring_version))
    db.commit()


def test_recall_hits_without_query_mappings_become_neighbors(temp_db, monkeypatch):
    """THE regression: a pooled, scored paper with no search mapping is selectable.

    All papers here have NO SearchQueryPaper rows, so the legacy ranking path is
    empty and the fallback must supply them. Before the fix this returned 0
    neighbours despite a full candidate pool.
    """
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    task, gap, papers = _seed(db, 6)
    _add_relevance(db, gap, papers, [0.9, 0.8, 0.7, 0.2, 0.1, 0.05])

    neighbors = select_gap_specific_neighbors(
        db, gap, [], limit=4, candidate_paper_ids=[p.id for p in papers])

    assert len(neighbors) == 4, (
        "recall-only candidates must fill the neighbour budget; returning "
        "fewer reproduces 'pool=10 selected=2 cap_bound=False'")
    db.close()


def test_recall_only_selection_is_ordered_by_gap_relevance(temp_db, monkeypatch):
    """Selection must follow gap relevance, not task-level score.

    Task score is identical (0.5) for every paper here, so only the gap
    relevance signal can produce this order — which is the point: the neighbour
    list the auditor reads must be the most gap-relevant papers available.
    """
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    task, gap, papers = _seed(db, 5)
    _add_relevance(db, gap, papers, [0.15, 0.95, 0.55, 0.05, 0.75])

    neighbors = select_gap_specific_neighbors(
        db, gap, [], limit=3, candidate_paper_ids=[p.id for p in papers])

    got = [p.id for p in neighbors]
    assert got == [papers[1].id, papers[4].id, papers[2].id]
    db.close()


def test_fallback_still_respects_the_candidate_pool_allowlist(temp_db, monkeypatch):
    """A high-relevance paper OUTSIDE the admitted pool must not be pulled in.

    The allowlist is an audit-identity guarantee: neighbours must come from the
    pool the audit was admitted on, otherwise the recorded admission no longer
    describes what was compared.
    """
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    task, gap, papers = _seed(db, 4)
    _add_relevance(db, gap, papers, [0.9, 0.8, 0.7, 0.6])

    allowed = [papers[2].id, papers[3].id]
    neighbors = select_gap_specific_neighbors(
        db, gap, [], limit=4, candidate_paper_ids=allowed)

    assert {p.id for p in neighbors} <= set(allowed)
    assert len(neighbors) == 2
    db.close()


def test_unscored_recall_hit_does_not_outrank_a_scored_one(temp_db, monkeypatch):
    """Missing relevance is unknown, not bad — but a scored paper still leads.

    `rel_map.get(pid, -1.0)` deliberately sorts unscored below scored so the
    screen's signal dominates; this pins that an unscored paper is still
    selectable when the budget exceeds the scored set.
    """
    from app.agent.steps.audit_gaps import select_gap_specific_neighbors

    db = temp_db()
    task, gap, papers = _seed(db, 3)
    _add_relevance(db, gap, papers[:2], [0.6, 0.4])

    neighbors = select_gap_specific_neighbors(
        db, gap, [], limit=3, candidate_paper_ids=[p.id for p in papers])

    assert len(neighbors) == 3
    assert [p.id for p in neighbors[:2]] == [papers[0].id, papers[1].id]
    db.close()

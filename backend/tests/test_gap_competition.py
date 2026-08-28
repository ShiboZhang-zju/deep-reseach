"""Top-2 gap competition (run7 review): mechanical cheap rank, audit-set
trimming, and winner selection among survivors."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.models import GapEvidenceLink, ResearchTask  # noqa: E402


@pytest.fixture()
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _make_gap(db, contract_id, task_id, sig, feas, nov, evidence_n=0, status="candidate"):
    from app.db.models import GapCandidate
    gap = GapCandidate(
        contract_id=contract_id, task_id=task_id,
        gap_type="boundary", description="observed",
        observed_problem="observed",
        missing_capability="capability", claimed_delta="delta x",
        testable_hypothesis="hypothesis", target_setting="setting",
        significance_score=sig, feasibility_score=feas, novelty_score=nov,
        status=status,
    )
    db.add(gap)
    db.commit()
    for i in range(evidence_n):
        db.add(GapEvidenceLink(
            gap_id=gap.id,
            evidence_id=f"ev-{gap.id[:8]}-{i}",
        ))
    db.commit()
    return gap


def _seed_task(db):
    task = ResearchTask(user_input="topic")
    db.add(task)
    db.commit()
    return task


def test_cheap_gap_rank_combines_weighted_dimensions(temp_db):
    """0.30*Importance + 0.25*Evidence + 0.25*Testability + 0.20*Differentiation.
    Evidence support is mechanical (verified-link count, capped at 8 -> 1.0)."""
    from app.agent.runner import _cheap_gap_rank_score

    db = temp_db()
    task = _seed_task(db)
    g_none = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5, evidence_n=0)
    g_evidence = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5, evidence_n=4)
    g_saturated = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5, evidence_n=20)

    base = _cheap_gap_rank_score(db, g_none)
    with_ev = _cheap_gap_rank_score(db, g_evidence)
    saturated = _cheap_gap_rank_score(db, g_saturated)
    # No evidence: 0.30*.5 + 0.25*0 + 0.25*.5 + 0.20*.5 = 0.375
    assert abs(base - 0.375) < 1e-6
    # 4 links -> ev=0.5 adds 0.125
    assert abs(with_ev - (base + 0.125)) < 1e-6
    # Evidence saturates at 8 links.
    assert abs(saturated - (base + 0.25)) < 1e-6
    db.close()


def test_competition_trims_audit_set_to_top2(temp_db):
    """Five candidates: only the top-2 by cheap rank enter the full audit;
    the rest stay candidates (deferred, not rejected) and the decision is
    traced."""
    from app.agent.runner import _apply_gap_competition_selection
    from app.db.models import AgentTrace

    db = temp_db()
    task = _seed_task(db)
    gaps = [
        _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9),   # rank 1
        _make_gap(db, task.id, task.id, sig=0.85, feas=0.8, nov=0.8),  # rank 2
        _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5),   # defer
        _make_gap(db, task.id, task.id, sig=0.4, feas=0.6, nov=0.3),   # defer
        _make_gap(db, task.id, task.id, sig=0.3, feas=0.3, nov=0.9),   # defer
    ]
    all_ids = [g.id for g in gaps]

    trimmed = _apply_gap_competition_selection(
        db, task.id, gaps, list(all_ids), pending_narrowed=False)

    assert len(trimmed) == 2
    assert trimmed == [gaps[0].id, gaps[1].id]
    for g in gaps[2:]:
        db.refresh(g)
        assert g.status == "candidate", "deferred candidates are kept, not rejected"
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_competition").one()
    payload = json.loads(trace.output_json)
    assert payload["stage"] == "select_audit_candidates"
    assert set(payload["deferred_gap_ids"]) == {g.id for g in gaps[2:]}
    db.close()


def test_competition_skips_trim_within_budget_and_narrowed(temp_db):
    """No reviewer spend when there is no contest: <= top-K candidates, or a
    narrowed re-audit round whose set is already the precise rewritten
    subset."""
    from app.agent.runner import _apply_gap_competition_selection

    db = temp_db()
    task = _seed_task(db)
    two = [_make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9),
           _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5)]
    assert _apply_gap_competition_selection(
        db, task.id, two, [g.id for g in two], pending_narrowed=False) == \
        [g.id for g in two]

    # Narrowed re-audit set is never trimmed.
    many = two + [
        _make_gap(db, task.id, task.id, sig=0.1, feas=0.1, nov=0.1)
        for _ in range(3)
    ]
    narrowed_ids = [many[0].id, many[1].id]
    assert _apply_gap_competition_selection(
        db, task.id, many, narrowed_ids, pending_narrowed=True) == narrowed_ids
    db.close()


def test_winner_selection_keeps_runner_up_verdict(temp_db):
    """Two survivors AFTER both completed their audits: the higher-ranked one
    is funded; the other keeps its confirmed verdict as confirmed_runner_up
    (visible for review, reviveable later) instead of being discarded."""
    from app.agent.runner import _resolve_gap_competition_winner
    from app.db.models import AgentTrace

    db = temp_db()
    task = _seed_task(db)
    winner = _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9,
                       status="surviving")
    runner_up = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5,
                          status="surviving")

    result = _resolve_gap_competition_winner(db, task.id, [runner_up.id, winner.id])

    assert result == [winner.id]
    db.refresh(runner_up)
    assert runner_up.status == "confirmed_runner_up"
    db.refresh(winner)
    assert winner.status == "surviving"
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_competition").one()
    payload = json.loads(trace.output_json)
    assert payload["stage"] == "winner_selection"
    assert payload["winner_gap_id"] == winner.id
    assert payload["runner_up_gap_ids"] == [runner_up.id]
    db.close()


def test_single_survivor_skips_competition(temp_db):
    from app.agent.runner import _resolve_gap_competition_winner

    db = temp_db()
    task = _seed_task(db)
    only = _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9,
                     status="surviving")
    assert _resolve_gap_competition_winner(db, task.id, [only.id]) == [only.id]
    db.refresh(only)
    assert only.status == "surviving"
    db.close()

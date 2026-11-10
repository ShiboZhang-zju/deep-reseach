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


# --- Starvation control (probe 0ae1a146, 2026-10-30) ------------------------

def _write_competition_trace(db, task_id, deferred_ids):
    from app.db.models import AgentTrace
    db.add(AgentTrace(
        task_id=task_id, step_name="gap_competition", step_type="decision",
        output_json=json.dumps({
            "stage": "select_audit_candidates",
            "top_k": 2,
            "deferred_gap_ids": deferred_ids,
        }),
    ))
    db.commit()


def test_deferred_bonus_accumulates_and_is_capped(temp_db):
    """Each previously-deferred round adds +0.03, capped at 3 rounds (+0.09).

    Probe 0ae1a146 left 16/20 gaps at `candidate`: the pool is semantically
    stable, so the same losers lost every round. The bonus gives a near-tied
    candidate its turn without letting it outrank a clearly stronger one.
    """
    from app.agent.runner import _cheap_gap_rank_score

    db = temp_db()
    task = _seed_task(db)
    gap = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5)

    base = _cheap_gap_rank_score(db, gap)
    assert _cheap_gap_rank_score(db, gap, 0) == base
    assert _cheap_gap_rank_score(db, gap, 1) == pytest.approx(base + 0.03)
    assert _cheap_gap_rank_score(db, gap, 3) == pytest.approx(base + 0.09)
    # Capped: 10 deferred rounds must not keep growing.
    assert _cheap_gap_rank_score(db, gap, 10) == pytest.approx(base + 0.09)
    # Negative input is clamped, never a penalty.
    assert _cheap_gap_rank_score(db, gap, -5) == base
    db.close()


def test_deferred_round_counts_reads_competition_traces(temp_db):
    from app.agent.runner import _deferred_round_counts
    from app.db.models import AgentTrace

    db = temp_db()
    task = _seed_task(db)
    _write_competition_trace(db, task.id, ["gap-a", "gap-b"])
    _write_competition_trace(db, task.id, ["gap-a"])
    # A winner_selection trace must not be mistaken for a deferral.
    db.add(AgentTrace(task_id=task.id, step_name="gap_competition",
                      step_type="decision",
                      output_json=json.dumps({"stage": "winner_selection",
                                              "winner_gap_id": "gap-c"})))
    db.commit()

    counts = _deferred_round_counts(db, task.id)
    assert counts["gap-a"] == 2
    assert counts["gap-b"] == 1
    assert "gap-c" not in counts
    db.close()


def test_starvation_retires_candidates_past_the_limit(temp_db, monkeypatch):
    """A candidate deferred `gap_starvation_deferral_limit` times is retired.

    Explicit terminal state, so "never audited" stops looking like "waiting
    for its turn" (probe 0ae1a146: 16 candidates accumulated silently).
    """
    from app.agent.runner import _apply_gap_competition_selection
    from app.config import settings
    from app.db.models import AgentTrace

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "max_gaps_to_deep_audit", 1)
    monkeypatch.setattr(settings, "gap_starvation_deferral_limit", 2)

    db = temp_db()
    task = _seed_task(db)
    strong = _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9)
    # Already deferred once, so this round makes it 2 -> retired.
    stale = _make_gap(db, task.id, task.id, sig=0.1, feas=0.1, nov=0.1)
    _write_competition_trace(db, task.id, [stale.id])

    selected = _apply_gap_competition_selection(
        db, task.id, [strong, stale], [strong.id, stale.id], pending_narrowed=False)

    assert selected == [strong.id]
    db.refresh(stale)
    assert stale.status == "rejected"

    payload = json.loads(db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_competition").order_by(
        AgentTrace.created_at.desc()).first().output_json)
    assert payload["starved_gap_ids"] == [stale.id]
    assert stale.id not in payload["deferred_gap_ids"]
    retire = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_starvation_retire").one()
    assert json.loads(retire.output_json)["gap_ids"] == [stale.id]
    db.close()


def test_starvation_limit_zero_disables_retirement(temp_db, monkeypatch):
    from app.agent.runner import _apply_gap_competition_selection
    from app.config import settings

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "max_gaps_to_deep_audit", 1)
    monkeypatch.setattr(settings, "gap_starvation_deferral_limit", 0)

    db = temp_db()
    task = _seed_task(db)
    strong = _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9)
    stale = _make_gap(db, task.id, task.id, sig=0.1, feas=0.1, nov=0.1)
    for _ in range(5):
        _write_competition_trace(db, task.id, [stale.id])

    _apply_gap_competition_selection(
        db, task.id, [strong, stale], [strong.id, stale.id], pending_narrowed=False)

    db.refresh(stale)
    assert stale.status == "candidate", "retirement must be off when limit is 0"
    db.close()


# --- Winner selection must respect the audited novelty (probe 1caa24c2) -----

def _audit_confirmed(db, gap, novelty_confidence):
    from app.db.repositories import gap_repo
    return gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=gap.task_id, adversarial_queries=[],
        audit_result="confirmed", recommended_action="continue",
        remaining_delta="residual delta", novelty_confidence=novelty_confidence,
        audit_round=1)


def test_winner_prefers_gap_whose_audit_clears_the_novelty_gate(temp_db):
    """A gap at novelty 0.90 must beat a slightly higher cheap-rank gap at 0.42.

    Probe 1caa24c2 (2026-10-30): a boundary gap audited at novelty 0.90 lost
    winner selection to a cross-domain gap at 0.42 (cheap rank 0.7335 vs
    0.7567, a 0.023 margin). Novelty <0.6 caps interventions at tier B, so the
    funded gap could only ever yield `conditional_review` — every Idea capped
    at tier B and the run ended `abstained/no_minimal_experiment_generated`
    while the gap that could have produced an executable candidate sat unused
    as a runner-up.
    """
    from app.agent.runner import _resolve_gap_competition_winner

    db = temp_db()
    task = _seed_task(db)
    # Higher cheap rank but audited novelty in the WARN band.
    warn_gap = _make_gap(db, task.id, task.id, sig=0.75, feas=0.7, nov=0.78,
                         evidence_n=4, status="surviving")
    # Lower cheap rank but audited novelty clears the gate.
    clean_gap = _make_gap(db, task.id, task.id, sig=0.78, feas=0.72, nov=0.66,
                          evidence_n=4, status="surviving")
    _audit_confirmed(db, warn_gap, 0.42)
    _audit_confirmed(db, clean_gap, 0.90)
    db.commit()

    result = _resolve_gap_competition_winner(
        db, task.id, [warn_gap.id, clean_gap.id])

    assert result == [clean_gap.id]
    db.refresh(warn_gap)
    assert warn_gap.status == "confirmed_runner_up"
    db.close()


def test_winner_falls_back_to_cheap_rank_when_both_clear_the_gate(temp_db):
    """The novelty criterion must not override rank among equally-fundable gaps."""
    from app.agent.runner import _resolve_gap_competition_winner

    db = temp_db()
    task = _seed_task(db)
    strong = _make_gap(db, task.id, task.id, sig=0.9, feas=0.9, nov=0.9,
                       status="surviving")
    weak = _make_gap(db, task.id, task.id, sig=0.5, feas=0.5, nov=0.5,
                     status="surviving")
    _audit_confirmed(db, strong, 0.8)
    _audit_confirmed(db, weak, 0.7)
    db.commit()

    assert _resolve_gap_competition_winner(
        db, task.id, [weak.id, strong.id]) == [strong.id]
    db.close()

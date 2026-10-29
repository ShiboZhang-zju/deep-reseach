"""Relevance floor on mechanical-novelty judgments (P2, 2026-10-29).

The bug these guard against, measured on the v7 replay pilot: the mechanical
novelty signal is `uncovered_ratio x decided_ratio` over the persisted
per-neighbour x per-claim coverage matrix. Both factors measure the JUDGMENT —
was every claim ruled on — and neither measures the JUDGED OBJECT. So when the
audit's neighbour pool is filled with off-domain papers, every one of them is
judged NONE (correctly: they cover nothing about the claim), decided_ratio hits
1.0, and `mechanical_novelty` reaches 1.0.

Observed on 11 v7-only gaps that survived via that override: all 33 neighbours
screened a median gap_relevance of 0.17, 17/33 below 0.3, and one gap's
neighbours were gravitational-wave / general-relativity papers at relevance
0.00. The relevance score already existed in `gap_paper_relevance`; only the
floor was missing.

The rules pinned here:
1. A judgment from a neighbour below `gap_relevance_min_neighbor` is evidence
   of NOTHING: it neither covers nor fails to cover, so it must not enter the
   numerator and must not raise decided_ratio.
2. When every judgment is demoted, the signal returns None (fail-open to the
   legacy gate) — "no usable prior art was retrieved" cannot certify novelty.
3. The floor defaults to 0 (disabled), so enabling it is a deliberate act and
   legacy behaviour is unchanged by default.
4. Unscored neighbours are admitted: a missing screen is not a relevance
   verdict, and treating it as one would silently change every old task.
5. Diagnostics are recorded with the gate off too, so the floor can be
   calibrated from a shadow run instead of guessed.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option(
        "script_location",
        os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _seed(db, judgments_by_claim, relevance=None):
    """Seed one gap with 3 atomic claims and a neighbour coverage matrix.

    judgments_by_claim: list[list[(neighbor_paper_id, coverage)]], one entry
    per claim index. relevance: dict[paper_id] -> gap_relevance score; omitted
    papers stay unscored.
    """
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import GapCandidate, ResearchContract, ResearchTask
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="lora interference", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="LoRA", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    gap = gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="mechanism_gap",
        description="residual interference", target_setting="lora",
        observed_problem="interference", existing_coverage="generic",
        missing_capability="quantify residual interference",
        claimed_delta="quantify residual interference magnitude",
        testable_hypothesis="h", falsification_condition="f",
        provenance_status="complete", question_ids=[], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    claims = [
        gap_repo.create_atomic_claim(db, task.id, gap.id, index, f"C{index}")
        for index in range(len(judgments_by_claim))
    ]
    for claim_index, judgments in enumerate(judgments_by_claim):
        for paper_id, coverage in judgments:
            gap_repo.create_neighbor_claim_coverage(
                db, task_id=task.id, gap_id=gap.id,
                neighbor_paper_id=paper_id, claim_id=claims[claim_index].id,
                coverage=coverage, rationale="",
            )
    for paper_id, score in (relevance or {}).items():
        gap_repo.upsert_gap_paper_relevance(
            db, gap.id, paper_id, task.id, score,
            scoring_version="gap-rel-v1")
    db.commit()
    return gap, claims


def _score(db, gap):
    from app.agent.steps.audit_gaps import _mechanical_novelty_score
    return _mechanical_novelty_score(db, gap)


# --- rule 3: the default is the calibrated floor --------------------------

def test_floor_defaults_to_the_calibrated_value():
    """0.3 is pinned, not incidental.

    Calibration on the v7 replay pilot: every floor in 0.2..0.3 gives the same
    outcome (8/11 overrides survive) because the relevance pool has no samples
    between 0.17 and 0.38; 0.4 starts discarding relevant neighbours and drops
    to 4/11. 0.0 is the documented legacy escape hatch and is exercised by the
    gate-off test below.
    """
    from app.config import settings

    assert settings.gap_relevance_min_neighbor == 0.3


def test_gate_off_ignores_relevance_entirely(temp_db, monkeypatch):
    """Three off-domain neighbours at relevance 0.00 still certify novelty
    while the gate is off — this is exactly the pre-P2 behaviour the floor
    exists to change, pinned so the change is observable."""
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.0)
    db = temp_db()
    gap, _ = _seed(db, [[("gw-1", "NONE")], [("gw-1", "NONE")], [("gw-1", "NONE")]],
                   relevance={"gw-1": 0.0})
    result = _score(db, gap)

    assert result["novelty"] == pytest.approx(1.0)
    assert result["judgments_demoted_by_relevance"] == 0
    db.close()


# --- rule 1: below-floor judgments are not evidence ----------------------

def test_all_off_domain_neighbours_demoted_returns_none(temp_db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    db = temp_db()
    gap, _ = _seed(
        db,
        [[("gw-1", "NONE")], [("gw-2", "NONE")], [("gw-1", "NONE")]],
        relevance={"gw-1": 0.0, "gw-2": 0.17})
    result = _score(db, gap)

    # Rule 2: no usable judgment -> fail-open to the legacy gate.
    assert result is None
    db.close()


def test_mixed_pool_only_relevant_judgments_count(temp_db, monkeypatch):
    from app.config import settings

    # Isolate the floor rule; the qualified-neighbour count is a separate gate.
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 1)
    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    db = temp_db()
    # claim 0: one relevant NONE; claim 1: only off-domain NONE;
    # claim 2: no judgment at all.
    gap, _ = _seed(
        db,
        [[("rel-1", "NONE"), ("gw-1", "NONE")],
         [("gw-1", "NONE")],
         []],
        relevance={"rel-1": 0.68, "gw-1": 0.0})
    result = _score(db, gap)

    assert result is not None
    # 1 of 3 claims decided, and it is uncovered -> 1/3 x 1/3.
    assert result["decided_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert result["uncovered_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert result["novelty"] == pytest.approx(1 / 9, abs=1e-3)
    assert result["judgments_demoted_by_relevance"] == 2
    db.close()


def test_relevant_neighbour_still_certifies_coverage(temp_db, monkeypatch):
    """The floor must not weaken real evidence: a relevant FULL still closes.

    FULL on an abstract-only neighbour is discounted to 0.6 by the existing
    evidence factor, so the residual is 0.4 per judged claim rather than 0 —
    the assertion is that relevance gating does not touch that credit, not that
    an unverified FULL is worth 1.0.
    """
    from app.config import settings

    # Isolate the floor rule; the qualified-neighbour count is a separate gate.
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 1)
    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    db = temp_db()
    gap, _ = _seed(
        db,
        [[("rel-1", "FULL")], [("rel-1", "FULL")], [("gw-1", "NONE")]],
        relevance={"rel-1": 0.68, "gw-1": 0.0})
    result = _score(db, gap)

    assert result["decided_ratio"] == pytest.approx(2 / 3, abs=1e-3)
    assert result["uncovered_ratio"] == pytest.approx(0.8 / 3, abs=1e-3)
    # Judged and largely covered, versus the 1.0 the same pool would score if
    # relevance were ignored and the off-domain NONE counted as decided.
    assert result["novelty"] < 0.2
    db.close()


# --- rule 4: an unscored neighbour is not a relevance verdict ------------

def test_unscored_neighbours_are_admitted(temp_db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    db = temp_db()
    # No relevance rows at all: legacy tasks that never ran the screen. Two
    # distinct neighbours, so the count gate is satisfied on admissions rather
    # than on scores.
    gap, _ = _seed(db, [[("unscored-a", "NONE")], [("unscored-b", "NONE")],
                        [("unscored-a", "NONE")]])
    result = _score(db, gap)

    assert result["novelty"] == pytest.approx(1.0)
    assert result["judgments_demoted_by_relevance"] == 0
    assert result["n_scored_neighbors"] == 0
    assert result["neighbor_relevance_min"] is None
    db.close()


# --- rule 5: diagnostics always available for calibration ----------------

def test_diagnostics_recorded_with_gate_off(temp_db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.0)
    db = temp_db()
    gap, _ = _seed(db, [[("a", "NONE")], [("b", "NONE")], [("c", "NONE")]],
                   relevance={"a": 0.0, "b": 0.17, "c": 0.68})
    result = _score(db, gap)

    assert result["neighbor_relevance_floor"] == 0.0
    assert result["n_judging_neighbors"] == 3
    assert result["n_scored_neighbors"] == 3
    assert result["neighbor_relevance_min"] == pytest.approx(0.0)
    assert result["neighbor_relevance_median"] == pytest.approx(0.17)
    # Nothing is demoted while the gate is off, but the count that WOULD be
    # demoted at a 0.3 floor is visible from the raw scores.
    assert result["judgments_demoted_by_relevance"] == 0
    db.close()


def test_below_floor_count_reflects_the_configured_floor(temp_db, monkeypatch):
    from app.config import settings

    # Isolate the floor rule; the qualified-neighbour count is a separate gate.
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 1)
    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    db = temp_db()
    gap, _ = _seed(db, [[("a", "NONE")], [("b", "NONE")], [("c", "NONE")]],
                   relevance={"a": 0.0, "b": 0.17, "c": 0.68})
    result = _score(db, gap)

    assert result["neighbors_below_floor"] == 2
    assert result["judgments_demoted_by_relevance"] == 2
    assert result["decided_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    db.close()


def test_a_single_qualified_neighbour_cannot_certify_novelty(temp_db, monkeypatch):
    """The second half of P2.

    With the 0.3 floor applied to the v7 pilot, 3 of the 8 surviving override
    gaps rested on exactly ONE qualified neighbour and two of those scored
    mechanical_novelty 1.00. One paper that did not mention a claim is not
    evidence that nobody has — so the signal refuses rather than guessing.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 2)
    db = temp_db()
    gap, _ = _seed(
        db,
        [[("rel-1", "NONE")], [("rel-1", "NONE")], [("rel-1", "NONE")]],
        relevance={"rel-1": 0.68})

    assert _score(db, gap) is None
    db.close()


def test_two_qualified_neighbours_are_enough(temp_db, monkeypatch):
    """The count is a floor, not a target: two qualified neighbours must still
    be able to certify, so the fix does not disable the override outright."""
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 2)
    db = temp_db()
    gap, _ = _seed(
        db,
        [[("rel-1", "NONE"), ("rel-2", "NONE")],
         [("rel-1", "NONE"), ("rel-2", "NONE")],
         [("rel-1", "NONE"), ("rel-2", "NONE")]],
        relevance={"rel-1": 0.68, "rel-2": 0.5})
    result = _score(db, gap)

    assert result is not None
    assert result["n_qualified_neighbors"] == 2
    assert result["novelty"] == pytest.approx(1.0)
    db.close()


def test_count_requirement_is_inert_while_the_floor_is_off(temp_db, monkeypatch):
    """Legacy escape hatch: with the floor off there is no notion of a
    'qualified' neighbour, so the count must not quietly start rejecting."""
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.0)
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 2)
    db = temp_db()
    gap, _ = _seed(db, [[("only", "NONE")], [("only", "NONE")], [("only", "NONE")]])
    result = _score(db, gap)

    assert result is not None
    assert result["novelty"] == pytest.approx(1.0)
    db.close()


def test_diagnostics_report_the_qualified_count(temp_db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.3)
    monkeypatch.setattr(settings, "mech_novelty_min_qualified_neighbors", 2)
    db = temp_db()
    gap, _ = _seed(db, [[("a", "NONE")], [("b", "NONE")], [("c", "NONE")]],
                   relevance={"a": 0.68, "b": 0.17, "c": 0.5})
    result = _score(db, gap)

    assert result["n_qualified_neighbors"] == 2
    assert result["min_qualified_neighbors"] == 2
    assert result["n_judging_neighbors"] == 3
    db.close()


def test_floor_above_every_score_certifies_nothing(temp_db, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gap_relevance_min_neighbor", 0.9)
    db = temp_db()
    gap, _ = _seed(db, [[("a", "NONE")], [("b", "NONE")], [("c", "NONE")]],
                   relevance={"a": 0.5, "b": 0.68, "c": 0.7})
    assert _score(db, gap) is None
    db.close()

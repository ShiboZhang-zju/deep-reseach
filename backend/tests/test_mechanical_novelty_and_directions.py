"""Tests for the 2026-09-08 option A+B batch:

Option A — undecided gaps survive an abstention as research_direction_only
records (runner._record_direction_candidates).
Option B — the low-novelty gate consults the mechanical claim-coverage matrix
(audit_gaps._mechanical_novelty_score) instead of trusting the swinging
self-reported novelty_confidence alone.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location", os.path.join(
        os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


def _seed_inconclusive_gap(db, claimed="PEFT-based editing cost-benefit beyond 70B models"):
    """A gap whose last audit said more_search (the abstention shape)."""
    from app.db.models import GapAudit, GapCandidate, Paper, ResearchContract, ResearchTask
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="knowledge editing", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Knowledge Editing",
                                status="active", version=1, input_hash="v1")
    paper = Paper(title="LocFT-BF", abstract="LocFT-BF evaluates at 7-8B scale.",
                  citation_count=9)
    db.add_all([contract, paper])
    db.flush()
    gap = GapCandidate(
        task_id=task.id,
        contract_id=contract.id,
        gap_type="boundary_gap",
        description="cost-benefit boundary unverified",
        claimed_delta=claimed,
        status="auditing",
    )
    db.add(gap)
    db.flush()
    gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=task.id, adversarial_queries=[],
        audit_result="uncertain", neighbor_paper_ids=[paper.id],
        recommended_action="more_search",
        nearest_neighbor_summary="LocFT-BF only evaluates 7-8B models.",
        differentiation_summary="The >70B cost-benefit boundary was not measured.",
        rejection_reason=("audit_search_unstable: retrieval union did not converge"),
        audit_round=1, search_policy_version="gap-search-budgeted-v16",
        search_admission_status="PASS", search_admission_reasons=[],
        search_query_ids=[], audited_claimed_delta=claimed,
    )
    db.commit()
    return db, task, gap, paper


# === Option A: undecided gaps become visible research_direction_only records ===

def test_finalize_records_direction_candidates(temp_db):
    from app.agent.runner import _finalize_inconclusive_gaps
    from app.db.models import ResearchIdea

    Session = temp_db
    db, task, gap, paper = _seed_inconclusive_gap(Session())

    closed = _finalize_inconclusive_gaps(db, task.id)
    db.commit()

    assert closed == 1
    assert gap.status == "inconclusive"
    ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 1
    idea = ideas[0]
    # Explicitly non-executable bookkeeping, visible but never counted as output.
    assert idea.decision == "research_direction_only"
    assert idea.gap_id == gap.id
    assert "unverified novelty" in idea.title
    assert "unproven, NOT disproven" in idea.motivation
    # The audit context survives: the user can see WHY it was undecided.
    assert "LocFT-BF" in idea.motivation
    assert "not measured" in idea.motivation
    assert json.loads(idea.quality_reason_codes_json) == [
        "UNDECIDED_NOVELTY_AT_BUDGET_EXHAUSTION"]
    db.close()


def test_direction_records_idempotent(temp_db):
    from app.agent.runner import _finalize_inconclusive_gaps
    from app.db.models import ResearchIdea

    Session = temp_db
    db, task, gap, paper = _seed_inconclusive_gap(Session())

    _finalize_inconclusive_gaps(db, task.id)
    db.commit()
    _finalize_inconclusive_gaps(db, task.id)
    db.commit()

    ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 1
    db.close()


def test_finalize_records_under_production_autoflush_off(temp_db):
    """Regression (task 25c8edf4): SessionLocal uses autoflush=False, so the
    status="inconclusive" flips inside _finalize_inconclusive_gaps used to be
    invisible to the very next query and no direction record was written on
    real runs (the first test passed only because this fixture's session
    autoflushes). Reproduce the production session config exactly."""
    import tempfile as _tf

    from alembic import command
    from alembic.config import Config as _Cfg
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.agent.runner import _finalize_inconclusive_gaps
    from app.db.models import ResearchIdea

    fd, db_path = _tf.mkstemp(suffix=".db")
    os.close(fd)
    config = _Cfg()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location", os.path.join(
        os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    ProdSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    try:
        db = ProdSession()
        db, task, gap, paper = _seed_inconclusive_gap(db)

        _finalize_inconclusive_gaps(db, task.id)
        db.commit()

        ideas = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task.id).all()
        assert len(ideas) == 1, (
            "direction record must be written even when the session does not "
            "autoflush the status flips")
        db.close()
    finally:
        engine.dispose()
        try:
            os.unlink(db_path)
        except PermissionError:
            pass


def test_ceiling_killed_gap_also_recorded(temp_db):
    """A gap the audit's verdict ceiling already marked inconclusive (not via
    _finalize_inconclusive_gaps' more_search path) must also be recorded."""
    from app.agent.runner import _record_direction_candidates
    from app.db.models import ResearchIdea

    Session = temp_db
    db, task, gap, paper = _seed_inconclusive_gap(Session())
    gap.status = "inconclusive"  # ceiling already closed it inside the audit
    db.commit()

    _record_direction_candidates(db, task.id)
    db.commit()

    ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 1
    assert ideas[0].gap_id == gap.id
    db.close()


# === Option B: mechanical claim-coverage matrix ===

def _seed_matrix_gap(db):
    """Gap + one full-text-verified neighbor + two atomic claims."""
    from app.db.models import (EvidenceUnit, GapCandidate, Paper,
                               ResearchContract, ResearchTask, TaskPaper)
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="llm watermarking", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Watermarking",
                                status="active", version=1, input_hash="v1")
    neighbor = Paper(title="SemStamp", abstract="Semantic watermarking.",
                     citation_count=30)
    db.add_all([contract, neighbor])
    db.flush()
    db.add(TaskPaper(task_id=task.id, paper_id=neighbor.id, discovered_round=1,
                     final_score=0.9, priority="high"))
    # Full-text verification: the neighbor was actually read (evidence factor 1.0).
    db.add(EvidenceUnit(task_id=task.id, paper_id=neighbor.id,
                        evidence_type="limitation",
                        normalized_claim="x", verification_status="verified"))
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="bigram perturbation boundary",
        claimed_delta="quantified robustness boundary under bigram perturbation",
        status="auditing",
    )
    db.add(gap)
    db.flush()
    c0 = gap_repo.create_atomic_claim(db, task.id, gap.id, 0,
                                      "quantified boundary under bigram perturbation")
    c1 = gap_repo.create_atomic_claim(db, task.id, gap.id, 1,
                                      "perplexity below 5% maintained")
    db.flush()
    return db, task, gap, neighbor, c0, c1


def test_mechanical_novelty_fulltext_uncovered_is_high(temp_db):
    """Claims explicitly judged NONE by a full-text-verified neighbor: the
    strongest novelty signal the system has — gated novelty must be high."""
    from app.agent.steps.audit_gaps import _mechanical_novelty_score
    from app.db.repositories import gap_repo

    Session = temp_db
    db, task, gap, neighbor, c0, c1 = _seed_matrix_gap(Session())
    for claim in (c0, c1):
        gap_repo.create_neighbor_claim_coverage(
            db, task_id=task.id, gap_id=gap.id,
            neighbor_paper_id=neighbor.id, claim_id=claim.id,
            coverage="NONE", rationale="not addressed in the paper")
    db.commit()

    result = _mechanical_novelty_score(db, gap)
    assert result is not None
    assert result["decided_ratio"] == 1.0
    assert result["uncovered_ratio"] == 1.0
    assert result["novelty"] == pytest.approx(1.0)
    db.close()


def test_mechanical_novelty_all_uncertain_is_not_novelty(temp_db):
    """All-UNCERTAIN = search was too weak to decide: decided_ratio 0 → the
    gated score must be ~0, never a false 'fully novel' 1.0. No decided claims
    at all → None (fail-open to the legacy gate)."""
    from app.agent.steps.audit_gaps import _mechanical_novelty_score
    from app.db.repositories import gap_repo

    Session = temp_db
    db, task, gap, neighbor, c0, c1 = _seed_matrix_gap(Session())
    for claim in (c0, c1):
        gap_repo.create_neighbor_claim_coverage(
            db, task_id=task.id, gap_id=gap.id,
            neighbor_paper_id=neighbor.id, claim_id=claim.id,
            coverage="UNCERTAIN", rationale="cannot tell from abstract")
    db.commit()

    assert _mechanical_novelty_score(db, gap) is None
    db.close()


def test_mechanical_novelty_mixed_dilutes(temp_db):
    """1 decided-uncovered + 1 undecided: the undecided claim dilutes —
    gated = (1/2 uncovered) × (1/2 decided) = 0.25, below the 0.5 floor."""
    from app.agent.steps.audit_gaps import _mechanical_novelty_score
    from app.db.repositories import gap_repo

    Session = temp_db
    db, task, gap, neighbor, c0, c1 = _seed_matrix_gap(Session())
    gap_repo.create_neighbor_claim_coverage(
        db, task_id=task.id, gap_id=gap.id, neighbor_paper_id=neighbor.id,
        claim_id=c0.id, coverage="NONE", rationale="absent")
    gap_repo.create_neighbor_claim_coverage(
        db, task_id=task.id, gap_id=gap.id, neighbor_paper_id=neighbor.id,
        claim_id=c1.id, coverage="UNCERTAIN", rationale="unclear")
    db.commit()

    result = _mechanical_novelty_score(db, gap)
    assert result["decided_ratio"] == pytest.approx(0.5)
    assert result["uncovered_ratio"] == pytest.approx(0.5)
    assert result["novelty"] == pytest.approx(0.25)
    db.close()


def test_mechanical_novelty_covered_is_zero(temp_db):
    """A FULL cover from the verified neighbor kills the score — the hollow-
    novelty guard (d6f64087) direction."""
    from app.agent.steps.audit_gaps import _mechanical_novelty_score
    from app.db.repositories import gap_repo

    Session = temp_db
    db, task, gap, neighbor, c0, c1 = _seed_matrix_gap(Session())
    for claim in (c0, c1):
        gap_repo.create_neighbor_claim_coverage(
            db, task_id=task.id, gap_id=gap.id,
            neighbor_paper_id=neighbor.id, claim_id=claim.id,
            coverage="FULL", rationale="directly implemented")
    db.commit()

    assert _mechanical_novelty_score(db, gap)["novelty"] == pytest.approx(0.0)
    db.close()


def _matrix_llm(claim_indices, novelty):
    """Audit LLM returning confirmed + a low self-report + explicit NONE
    coverage rows (which the pipeline persists into the matrix)."""
    from app.agent.steps.audit_gaps import (
        ClaimCoverageSchema, GapAuditDecisionSchema, NeighborAuditSchema)

    class _LLM:
        async def chat_json(self, messages, schema):
            text = messages[1]["content"]
            evidence_id = __import__("ast").literal_eval(next(
                line.split(": ", 1)[1] for line in text.splitlines()
                if line.startswith("Supporting evidence IDs:")))[0]
            paper_id = next(line.split(": ", 1)[1] for line in text.splitlines()
                            if "Paper ID:" in line)
            return GapAuditDecisionSchema(
                audit_result="confirmed",
                recommended_action="continue",
                remaining_delta="neighbors did not measure the bigram boundary",
                nearest_neighbor_summary="SemStamp covers semantic watermarks.",
                differentiation_summary="bigram boundary not quantified anywhere",
                evidence_for_gap_ids=[evidence_id],
                novelty_confidence=novelty,
                audit_confidence=0.8,
                comparisons=[NeighborAuditSchema(
                    paper_id=paper_id,
                    similarity_score=0.6,
                    overlap_ratio=0.2,
                    overlap_risk=0.2,
                    claim_coverage=[
                        ClaimCoverageSchema(claim_index=i, coverage="NONE",
                                            rationale="not in the paper")
                        for i in claim_indices
                    ],
                )],
            )
    return _LLM()


def _async_value(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner()


@pytest.mark.asyncio
async def test_low_novelty_mechanical_backing_survives_with_corrected_score(
        temp_db, monkeypatch):
    """confirmed + self-report 0.3 + verified-uncovered matrix → the gate must
    NOT downgrade: the matrix is the evidence, the self-report was noise. The
    score is corrected upward so the downstream tier gate sees the mechanical
    value (tier A reachable), and the correction is traced."""
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps import audit_gaps as module
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from test_audit_gaps import _pin_admission_and_neighbors, _seed_gap

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "audit_low_novelty_provisional_survive", True)
    monkeypatch.setattr(settings, "audit_mechanical_novelty_gate_enabled", True)

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap)
    # Two atomic claims so the matrix has something to aggregate.
    from app.db.repositories import gap_repo
    claims = [gap_repo.create_atomic_claim(db, task.id, gap.id, index,
                                           f"claim {index}")
              for index in range(2)]
    db.commit()
    monkeypatch.setattr(module, "_ensure_atomic_claims",
                        lambda *args: _async_value(claims))
    monkeypatch.setattr(module, "generate_english_adversarial_queries",
                        lambda *args: _async_value([
                            module.AdversarialQuerySpec(
                                "overlap", "bigram perturbation boundary")]))
    monkeypatch.setattr(module, "score_all_gap_candidates",
                        lambda *args: _async_value([]))
    # The NPA union-convergence gate downgrades a claim-derived confirmed when
    # the query union is UNMEASURED (a single fixture query). This test is
    # about the low-novelty gate, not convergence — supply a converged set.
    from types import SimpleNamespace
    monkeypatch.setattr(module, "_compute_npa_diagnostics",
                        lambda db_, g: SimpleNamespace(
                            cumulative_convergence=0.9,
                            instable_families=[],
                            median_family_stability=0.9,
                            stability_at_k={}, family_stabilities=[],
                            cross_round_stability=None,
                            search_confidence="HIGH", family_coverage=1.0))

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id,
                          current_round=2)
    results = await audit_gap_candidates(
        db, state, _matrix_llm([0, 1], novelty=0.3), task.id,
        perform_search=False)

    assert [r.recommended_action for r in results] == ["continue"]
    assert gap.status == "surviving"
    # The self-reported 0.3 was search noise; the mechanical value (full
    # verified uncoverage) is what the downstream tier gate should see.
    from app.db.models import GapAudit
    audit = db.query(GapAudit).filter(
        GapAudit.gap_id == gap.id).order_by(
        GapAudit.created_at.desc()).first()
    assert audit.novelty_confidence == pytest.approx(1.0)
    db.close()


@pytest.mark.asyncio
async def test_low_novelty_without_matrix_keeps_provisional_path(temp_db, monkeypatch):
    """No matrix rows (mechanical fails open) → the existing provisional
    survive applies: the gap survives but the self-reported low score stays,
    so the downstream gate still caps ideas at tier B."""
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps import audit_gaps as module
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from test_audit_gaps import _pin_admission_and_neighbors, _seed_gap

    monkeypatch.setattr(settings, "gap_audit_budgeted", True)
    monkeypatch.setattr(settings, "audit_low_novelty_provisional_survive", True)
    monkeypatch.setattr(settings, "audit_mechanical_novelty_gate_enabled", True)

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap)
    monkeypatch.setattr(module, "_ensure_atomic_claims",
                        lambda *args: _async_value([]))
    monkeypatch.setattr(module, "generate_english_adversarial_queries",
                        lambda *args: _async_value([
                            module.AdversarialQuerySpec(
                                "overlap", "bigram perturbation boundary")]))
    monkeypatch.setattr(module, "score_all_gap_candidates",
                        lambda *args: _async_value([]))

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id,
                          current_round=2)
    # No claims persisted → the matrix aggregator fails open → provisional
    # survive without a corrected score.
    results = await audit_gap_candidates(
        db, state, _matrix_llm([], novelty=0.3), task.id, perform_search=False)

    assert gap.status == "surviving"
    from app.db.models import GapAudit
    audit = db.query(GapAudit).filter(
        GapAudit.gap_id == gap.id).order_by(
        GapAudit.created_at.desc()).first()
    assert audit.novelty_confidence == pytest.approx(0.3)
    db.close()

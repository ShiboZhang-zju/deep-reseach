"""V2-no-audit ablation arm (2026-11-02).

The ablation exists to answer a question production_e2e_v1 never answered:
how much does the adversarial audit ITSELF contribute, given that the bundle
comparison only ever showed `Full V2 > simple baselines`?

The property these tests protect is INTERPRETABILITY, not behaviour: an
un-audited gap must be distinguishable from a gap the auditor looked at and
found nothing, because otherwise a null result cannot be attributed to the
audit. Concretely, the arm must

  * stamp a distinct search-policy version (no mixing with real audits),
  * record that no search and no judgement happened,
  * pass the intervention novelty gate EXPLICITLY instead of leaving it UNKNOWN
    (an UNKNOWN would cap every intervention at tier B for a bookkeeping
    reason and the arm would lose on executable count without saying anything
    about the audit).
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


def _seed_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import GapCandidate, ResearchContract, ResearchTask

    task = ResearchTask(user_input="ablation topic", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="t",
                                input_hash="h-ablation", version=1)
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="d", question_ids_json="[]", supporting_evidence_ids_json="[]",
        contradicting_evidence_ids_json="[]", mining_round=1, status="candidate",
        target_setting="long-context QA", observed_problem="degrades past 32k",
        existing_coverage="evaluated to 32k",
        missing_capability="no measurement beyond 32k",
        claimed_delta="Accuracy beyond 32k tokens has not been measured.",
        testable_hypothesis="accuracy drops >10pp beyond 32k",
        falsification_condition="a paper measures beyond 32k",
        provenance_status="complete",
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.commit()
    return task, contract, gap


def _state(task, contract):
    from app.agent.state import ResearchState
    return ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)


# === Policy stamp ==========================================================

def test_ablation_stamps_a_distinct_search_policy_version():
    """Ablation rows must never be mistaken for real audits.

    GAP_SEARCH_POLICY_VERSION is hashed into the audit phase input_version and
    filtered by `_latest_decided_audit`, so a shared value would let the two
    arms' records stand in for each other.
    """
    import importlib

    from app.config import settings

    original = settings.gap_audit_disabled_ablation
    try:
        import app.agent.steps.audit_gaps as mod
        settings.gap_audit_disabled_ablation = False
        importlib.reload(mod)
        normal = mod.GAP_SEARCH_POLICY_VERSION
        settings.gap_audit_disabled_ablation = True
        importlib.reload(mod)
        ablated = mod.GAP_SEARCH_POLICY_VERSION
        assert ablated == "gap-search-ablation-none-v1"
        assert ablated != normal
    finally:
        settings.gap_audit_disabled_ablation = original
        importlib.reload(importlib.import_module("app.agent.steps.audit_gaps"))


def test_audit_disabled_reads_the_switch_lazily(monkeypatch):
    from app.config import settings
    from app.agent.steps.audit_gaps import audit_disabled

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", False)
    assert audit_disabled() is False
    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    assert audit_disabled() is True


# === The ablated audit record ==============================================

async def test_ablated_audit_survives_without_searching_or_judging(temp_db, monkeypatch):
    from app.config import settings
    from app.agent.steps.audit_gaps import audit_gap_candidate
    from app.db.models import GapAudit

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    db = temp_db()
    task, contract, gap = _seed_gap(db)
    state = _state(task, contract)

    # No LLM is supplied on purpose: the ablated path must not need one. A
    # `None` that raises on attribute access is the strictest way to assert
    # "no model call happened".
    result = await audit_gap_candidate(db, state, None, task.id, gap)

    db.refresh(gap)
    assert gap.status == "surviving"
    assert result.audit_result == "confirmed"
    assert result.recommended_action == "continue"

    audits = db.query(GapAudit).filter(GapAudit.gap_id == gap.id).all()
    assert len(audits) == 1
    audit = audits[0]
    assert audit.search_policy_version == "gap-search-ablation-none-v1"
    assert audit.search_admission_status == "PASS"
    assert "ABLATION_NO_AUDIT" in (audit.search_admission_reasons_json or "")
    assert audit.adversarial_queries_json in ("[]", None)
    assert audit.neighbor_paper_ids_json in ("[]", None)
    # remaining_delta must carry the gap's own claim, so downstream prompts get
    # text rather than an empty field that reads as "auditor found nothing to
    # say".
    assert audit.remaining_delta == gap.claimed_delta
    db.close()


async def test_ablated_audit_keeps_neighbour_fields_empty_not_fabricated(temp_db,
                                                                        monkeypatch):
    """No neighbours must be reported, ever.

    The whole point of the arm is "no comparison material was gathered". A
    fabricated or corpus-topped-up neighbour list would make the arm look like
    a real audit that happened to find nothing.
    """
    from app.config import settings
    from app.agent.steps.audit_gaps import audit_gap_candidate
    from app.db.models import GapAudit

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    db = temp_db()
    task, contract, gap = _seed_gap(db)
    await audit_gap_candidate(db, _state(task, contract), None, task.id, gap)

    audit = db.query(GapAudit).filter(GapAudit.gap_id == gap.id).one()
    assert audit.neighbor_paper_ids_json == "[]"
    assert audit.novelty_confidence is None
    db.close()


# === Downstream interpretability ==========================================

def _audit_stub(**kwargs):
    from types import SimpleNamespace
    base = dict(audit_result="confirmed", remaining_delta=None,
                novelty_confidence=None, neighbor_paper_ids_json="[]")
    base.update(kwargs)
    return SimpleNamespace(**base)


def _gap_stub():
    from types import SimpleNamespace
    return SimpleNamespace(provenance_status="complete")


def _contract_stub():
    from types import SimpleNamespace
    return SimpleNamespace(gpu_available=True, allow_model_training=True,
                           allow_large_benchmark=True, max_gpu_hours=24,
                           max_api_budget=100, max_runtime_minutes=600)


def _candidate_stub():
    from types import SimpleNamespace
    return SimpleNamespace(proposed_intervention="add a probe",
                           implementation_cost="low",
                           dependency_paper_ids=[])


def test_novelty_gate_passes_explicitly_in_ablation_arm(monkeypatch):
    """UNKNOWN would cap the arm at tier B for a bookkeeping reason.

    Without this, the ablation arm can never produce an executable_candidate
    regardless of pipeline quality, so its executable count is guaranteed to be
    zero and carries no information about the audit.
    """
    from app.config import settings
    from app.agent.steps.generate_interventions import _evaluate_hard_gates

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    gates = _evaluate_hard_gates(_gap_stub(), None, ["e1", "e2"],
                                 _candidate_stub(), _contract_stub())
    assert gates["gate_statuses"]["novelty"] == "PASS"
    assert "ABLATION_NO_AUDIT" in gates["gate_rationale"]["novelty"]


def test_novelty_gate_still_unknown_without_audit_outside_ablation(monkeypatch):
    """The ablation must not leak: outside it, a missing audit stays UNKNOWN."""
    from app.config import settings
    from app.agent.steps.generate_interventions import _evaluate_hard_gates

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", False)
    gates = _evaluate_hard_gates(_gap_stub(), None, ["e1", "e2"],
                                 _candidate_stub(), _contract_stub())
    assert gates["gate_statuses"]["novelty"] == "UNKNOWN"
    assert "ABLATION_NO_AUDIT" not in gates["gate_rationale"]["novelty"]


def test_ablation_does_not_override_a_real_closed_verdict(monkeypatch):
    """Guard against the ablation switch silencing a real auditor.

    A resumed task (or a contaminated run) can carry real audits recorded
    before the switch was flipped. The switch is an arm marker, not a verdict
    eraser: turning `closed` into PASS would resurrect gaps the auditor killed.
    """
    from app.config import settings
    from app.agent.steps.generate_interventions import _evaluate_hard_gates

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    real_audit = _audit_stub(audit_result="closed",
                             search_policy_version="gap-search-budgeted-v16")
    gates = _evaluate_hard_gates(_gap_stub(), real_audit,
                                 ["e1", "e2"], _candidate_stub(),
                                 _contract_stub())
    assert gates["gate_statuses"]["novelty"] == "FAIL", (
        "a real `closed` verdict must survive the ablation switch")
    assert "ABLATION_NO_AUDIT" not in gates["gate_rationale"]["novelty"]


def test_ablation_pass_applies_to_ablation_arm_records(monkeypatch):
    """An ablation-arm record (its own policy stamp) gets PASS-by-ablation."""
    from app.config import settings
    from app.agent.steps.audit_gaps import ABLATION_SEARCH_POLICY_VERSION
    from app.agent.steps.generate_interventions import _evaluate_hard_gates

    monkeypatch.setattr(settings, "gap_audit_disabled_ablation", True)
    ablation_audit = _audit_stub(
        audit_result="confirmed",
        search_policy_version=ABLATION_SEARCH_POLICY_VERSION)
    gates = _evaluate_hard_gates(_gap_stub(), ablation_audit,
                                 ["e1", "e2"], _candidate_stub(),
                                 _contract_stub())
    assert gates["gate_statuses"]["novelty"] == "PASS"
    assert "ABLATION_NO_AUDIT" in gates["gate_rationale"]["novelty"]

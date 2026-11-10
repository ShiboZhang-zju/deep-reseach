"""v6 experimental control arm (fixed-budget experiment, 2026-10-31).

`GAP_MINING_LEGACY_V6_MODE` reproduces the v6 two-type mining policy on the
current stack so the experiment's arm B differs from arm A ONLY in mining
policy (see eval_results/fixed_budget_pilot/EXPERIMENT_DESIGN.md §3.1):

  - two lenses (boundary_gap, missing_evaluation) with v6-faithful prompts;
  - question-level blockers restored (NO_LIMITATION_SIGNAL /
    UNRESOLVED_VERIFIED_CONTRADICTION);
  - the v7.2 admission gates (PERFORMANCE_TARGET_DELTA; the scaling precedent
    check never fires because the scaling lens does not exist in this mode);
  - candidates stamped `evidence-admission-v6-exp`, separable from both the
    v7.2 rows and the 281 historical `evidence-admission-v6` rows.
"""

import json
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
                        os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
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


# === Policy version + contract set ==========================================

def test_v6_mode_resolves_policy_version(monkeypatch):
    from app.config import settings
    from app.agent.steps.mine_gaps import (
        GAP_MINING_POLICY_VERSION, V6_EXPERIMENTAL_POLICY_VERSION,
        current_mining_policy_version,
    )

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", False)
    assert current_mining_policy_version() == GAP_MINING_POLICY_VERSION
    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    assert current_mining_policy_version() == V6_EXPERIMENTAL_POLICY_VERSION
    # The experimental stamp must not collide with the historical one:
    # 281 rows already carry "evidence-admission-v6" and mixing them would
    # contaminate both the experiment sample and historical analyses.
    assert V6_EXPERIMENTAL_POLICY_VERSION == "evidence-admission-v6-exp"


def test_v6_contract_set_is_two_legacy_types_with_v6_rules():
    from app.agent.steps.mine_gaps import _V6_EXPERIMENT_CONTRACTS

    types = [c.gap_type for c in _V6_EXPERIMENT_CONTRACTS]
    assert types == ["boundary_gap", "missing_evaluation"]
    for c in _V6_EXPERIMENT_CONTRACTS:
        # v6's GLOBAL citation rule: limitation or negative_result for BOTH
        # types (v7.2 widened missing_evaluation to include future_work —
        # that widening is part of the hypothesis under test).
        assert c.required_evidence_types == frozenset({"limitation",
                                                       "negative_result"})
        # The control-arm prompts must NOT carry v7.2's outcome-target
        # wording: it is part of the v7.2 hypothesis and leaking it into arm
        # B would destroy the isolation the experiment depends on.
        assert "PERFORMANCE_TARGET_DELTA" not in c.discovery_directive
        assert "outcome target" not in c.discovery_directive


# === Question-level blockers (v6 semantics restored) ========================

def _admission_for(question, links, evidence_by_id):
    from app.agent.steps.mine_gaps import evaluate_gap_mining_admission
    return evaluate_gap_mining_admission(question, links, evidence_by_id)


def _mk_units(task_id, paper_id, spec):
    """spec: list of (evidence_type, verification_status)."""
    from types import SimpleNamespace
    return [SimpleNamespace(
        id=f"ev-{i}", paper_id=f"{paper_id}-{i}", task_id=task_id,
        evidence_type=t, verification_status=s,
        original_span="span", source_chunk_hash="h", page_number=1,
        span_start=0, span_end=5,
    ) for i, (t, s) in enumerate(spec)]


def _mk_links(units):
    from types import SimpleNamespace
    return [SimpleNamespace(evidence_id=u.id, relation_type="supports",
                            relevance_score=0.9) for u in units]


def test_v6_mode_blocks_question_without_limitation_signal(monkeypatch):
    from app.config import settings
    from types import SimpleNamespace

    question = SimpleNamespace(id="q1", status="open")
    # Two independent papers, but only comparison/result evidence — no
    # limitation signal anywhere.
    units = _mk_units("t1", "p", [("comparison", "verified"),
                                  ("result", "verified")])
    links = _mk_links(units)
    by_id = {u.id: u for u in units}

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", False)
    modern = _admission_for(question, links, by_id)
    assert modern.status == "PASS"  # v7: per-type relaxation

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    legacy = _admission_for(question, links, by_id)
    assert legacy.status == "UNKNOWN"
    assert "NO_LIMITATION_SIGNAL" in legacy.reason_codes


def test_v6_mode_blocks_question_with_unresolved_contradiction(monkeypatch):
    from app.config import settings
    from types import SimpleNamespace

    question = SimpleNamespace(id="q1", status="open")
    units = _mk_units("t1", "p", [("limitation", "verified"),
                                  ("limitation", "verified")])
    links = _mk_links(units) + [SimpleNamespace(
        evidence_id=units[0].id, relation_type="contradicts",
        relevance_score=0.9)]
    by_id = {u.id: u for u in units}

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", False)
    modern = _admission_for(question, links, by_id)
    assert modern.status == "PASS"  # v7: contradiction is raw material

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    legacy = _admission_for(question, links, by_id)
    assert legacy.status == "UNKNOWN"
    assert "UNRESOLVED_VERIFIED_CONTRADICTION" in legacy.reason_codes


# === Mining-chain integration ===============================================

def _seed_pool(Session):
    from app.db.models import (
        EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract,
        ResearchQuestion, ResearchTask,
    )

    db = Session()
    task = ResearchTask(user_input="retrieval augmentation", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(
        task_id=task.id, topic="Retrieval Augmentation", status="active",
        version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="How does model scale affect retrieval augmentation?",
        question_type="failure", importance=0.9, searchability=0.8,
        status="partially_covered",
    )
    db.add(question)
    db.flush()
    first_paper = Paper(title="Evidence Paper A")
    second_paper = Paper(title="Evidence Paper B")
    db.add_all([first_paper, second_paper])
    db.flush()
    first_evidence = EvidenceUnit(
        task_id=task.id, paper_id=first_paper.id, evidence_type="limitation",
        normalized_claim="Evaluation only covers models above 1B parameters.",
        original_span="only covers models above 1B", section="Limitations",
        page_number=8, span_start=10, span_end=42, source_chunk_hash="fulltext-a",
        verification_status="verified", extraction_confidence=0.8,
    )
    second_evidence = EvidenceUnit(
        task_id=task.id, paper_id=second_paper.id, evidence_type="negative_result",
        normalized_claim="Sub-1B behaviour degrades and was never reported.",
        verification_status="verified", extraction_confidence=0.8,
    )
    db.add_all([first_evidence, second_evidence])
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=first_evidence.id,
                             relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id,
                             relation_type="supports", relevance_score=0.9),
    ])
    db.commit()
    return db, task, contract, question, first_evidence, second_evidence


def _parse_prompt_ids(messages):
    user_content = messages[1]["content"]
    question_id = next(line.split(": ", 1)[1] for line in user_content.splitlines()
                       if "Question ID:" in line)
    evidence_ids = [line.split(": ", 1)[1] for line in user_content.splitlines()
                    if "Evidence ID:" in line]
    return question_id, evidence_ids


class _LensFakeLLM:
    """Returns one candidate of `lens_type` for that lens, nothing elsewhere."""

    def __init__(self, lens_type, claimed_delta):
        self.lens_type = lens_type
        self.claimed_delta = claimed_delta
        self.lens_calls: list[str] = []

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        if schema is not GapCandidateList:
            raise AssertionError(f"unexpected schema in lens call: {schema}")
        header = messages[0]["content"]
        for line in header.splitlines():
            if "Opportunity type for this call:" in line:
                self.lens_calls.append(line.split(":", 1)[1].strip())
        if f"Opportunity type for this call: {self.lens_type}" not in header:
            return GapCandidateList(gaps=[])
        question_id, evidence_ids = _parse_prompt_ids(messages)
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type=self.lens_type,
            description="Sub-1B retrieval augmentation behaviour is undocumented.",
            target_setting="Retrieval augmentation below 1B parameters",
            observed_problem="Evaluations only cover models above 1B parameters.",
            existing_coverage="Results reported for 1B-70B models.",
            missing_capability="Measurements in the sub-1B regime.",
            claimed_delta=self.claimed_delta,
            testable_hypothesis="Retrieval augmentation changes sign below 1B parameters.",
            falsification_condition="A paper reports sub-1B retrieval augmentation results.",
            question_ids=[question_id],
            supporting_evidence_ids=evidence_ids,
        )])


def _mining_trace(db, task_id):
    from app.db.models import AgentTrace

    return json.loads(db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id,
        AgentTrace.step_name == "gap_mining_candidates",
    ).order_by(AgentTrace.created_at.desc()).first().output_json)


@pytest.mark.asyncio
async def test_v6_mode_fans_out_two_lenses_and_stamps_v6exp(temp_db, monkeypatch):
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = _LensFakeLLM(
        "boundary_gap",
        "Fixed-budget memory fails to retain state-change evidence; no prior "
        "work evaluates this boundary below 1B parameters.",
    )

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    # Only the two v6 lenses ran at all.
    called_types = [c.split(" —")[0].strip() for c in llm.lens_calls]
    assert sorted(called_types) == ["boundary_gap", "missing_evaluation"]
    trace = _mining_trace(db, task.id)
    assert [lf["lens"] for lf in trace["lens_fanout"]] == [
        "boundary_gap", "missing_evaluation"]

    # The accepted candidate carries the experimental v6 stamp.
    assert len(gaps) == 1
    assert gaps[0].mining_policy_version == "evidence-admission-v6-exp"
    db.close()


@pytest.mark.asyncio
async def test_v6_mode_skips_performance_target_gate(temp_db, monkeypatch):
    """The v7.2 gate must not fire in the control arm.

    The exact delta shape v7.2 rejects (the pilot's human-true survivors)
    passes in v6 mode — PERFORMANCE_TARGET_DELTA is part of the v7.2
    hypothesis under test and must stay out of arm B.
    """
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = _LensFakeLLM(
        "boundary_gap",
        "Achieves >10% higher accuracy than fixed-budget baselines while "
        "maintaining <5% latency overhead.",
    )

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1, "v6 mode must accept the outcome-target shape"
    trace = _mining_trace(db, task.id)
    reasons = [r["reason"] for r in trace.get("rejected_candidates", [])]
    assert "PERFORMANCE_TARGET_DELTA" not in reasons
    db.close()


@pytest.mark.asyncio
async def test_v6_mode_no_scaling_precedent_trace(temp_db, monkeypatch):
    """The scaling precedent check must not execute in the control arm.

    The check is gated on the scaling_regime lens, which does not exist in
    v6 mode — assert no precedent-check trace is written even though the
    feature flag is on.
    """
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import AgentTrace

    assert settings.gap_mining_precedent_check_enabled, (
        "the flag is on by default; the lens absence is what must skip it")
    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = _LensFakeLLM(
        "missing_evaluation",
        "No prior work evaluates retrieval augmentation below 1B parameters.",
    )

    await mine_gap_candidates(db, state, llm, task.id)

    n = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "scaling_precedent_check").count()
    assert n == 0
    db.close()

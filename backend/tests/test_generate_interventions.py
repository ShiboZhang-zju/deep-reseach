"""Tests for gap-to-intervention synthesis and its citation handling."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.steps.generate_interventions import InterventionSchema, _evaluate_hard_gates


def _candidate(method, cost):
    return InterventionSchema(intervention_type="evaluation_protocol", failure_mechanism="state change failure", proposed_intervention=method, intermediate_effect="separates state changes", measurable_outcome="accuracy", implementation_cost=cost, mechanism_confidence=0.8)

def test_hard_gates_pass_for_evidence_backed_low_cost_intervention():
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, audit, ["e1", "e2"], _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"] == {"evidence": "PASS", "novelty": "PASS", "feasibility": "PASS"}

def test_hard_gates_reject_training_without_permission():
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, audit, ["e1", "e2"], _candidate("Fine-tune state model", "requires GPU training"), contract)
    assert result["gate_statuses"]["feasibility"] == "FAIL"


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
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


class _InterventionLLM:
    def __init__(self, dependency_paper_ids):
        self.dependency_paper_ids = dependency_paper_ids

    async def chat_json(self, messages, schema):
        from app.agent.steps.generate_interventions import InterventionList

        self.system_prompt = messages[0]["content"]
        return InterventionList(interventions=[InterventionSchema(
            intervention_type="evaluation_protocol",
            failure_mechanism="Generic QA hides state changes.",
            proposed_intervention="Add a fixed-budget state-change evaluation.",
            intermediate_effect="Separates stable and changing states.",
            measurable_outcome="State-change accuracy.",
            implementation_cost="Low cost evaluation.",
            dependency_paper_ids=list(self.dependency_paper_ids),
            mechanism_confidence=0.8,
        )])


def _seed_surviving_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import EvidenceUnit, Paper, ResearchContract, ResearchTask
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="agent memory", status="synthesizing_ideas")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1", gpu_available=False,
                                allow_model_training=False, allow_large_benchmark=False)
    neighbour = Paper(title="Neighbour", abstract="generic evaluation")
    db.add_all([contract, neighbour])
    db.flush()
    gap = gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="missing_evaluation",
        description="状态变化边界未被评估", target_setting="固定预算 Agent Memory",
        observed_problem="状态变化导致证据丢失", existing_coverage="一般问答准确率",
        claimed_delta="固定预算下的状态变化边界评测", missing_capability="状态变化边界评测",
        testable_hypothesis="状态变化性能较低", falsification_condition="已有同设置边界评测",
        provenance_status="complete", question_ids=[], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    gap.status = "surviving"
    for index in range(2):
        paper = Paper(title=f"Support {index}", abstract="a")
        db.add(paper)
        db.flush()
        unit = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation",
                            normalized_claim=f"claim {index}", verification_status="verified")
        db.add(unit)
        db.flush()
        gap_repo.create_gap_evidence_link(db, gap.id, unit.id, "suggests", 0.9)
    gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=task.id, adversarial_queries=[],
        audit_result="confirmed", recommended_action="continue",
        remaining_delta="固定预算下的状态变化边界仍未被任何近邻覆盖。",
        neighbor_paper_ids=[neighbour.id], audit_round=1)
    db.commit()
    return task, contract, gap, neighbour


@pytest.mark.asyncio
async def test_unoffered_dependency_paper_is_dropped_not_the_whole_intervention(temp_db):
    """One unusable citation must not discard an otherwise valid intervention.

    Production case (task 5e040ad5): every intervention for the only surviving
    gap was skipped as "unknown dependency paper", so the run produced no idea.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.generate_interventions import generate_interventions
    from app.db.models import AgentTrace, InterventionCandidate

    db = temp_db()
    task, contract, gap, neighbour = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    llm = _InterventionLLM([neighbour.id, "paper-that-was-never-offered"])

    result = await generate_interventions(db, state, llm, task.id)

    assert len(result.intervention_ids) == 1
    assert len(result.passed_intervention_ids) == 1
    item = db.get(InterventionCandidate, result.intervention_ids[0])
    assert json.loads(item.dependency_paper_ids_json) == [neighbour.id], (
        "only offered neighbour papers may be stored as dependencies"
    )
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "generate_interventions").one()
    dropped = json.loads(trace.output_json)["dropped_dependency_paper_ids"]
    assert dropped == [{"gap_id": gap.id,
                        "dropped_paper_ids": ["paper-that-was-never-offered"]}]
    db.close()


@pytest.mark.asyncio
async def test_intervention_prompt_separates_evidence_ids_from_paper_ids(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.generate_interventions import generate_interventions

    db = temp_db()
    task, contract, gap, neighbour = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    llm = _InterventionLLM([])

    await generate_interventions(db, state, llm, task.id)

    assert "dependency_paper_ids" in llm.system_prompt
    assert "Neighbor paper IDs" in llm.system_prompt, (
        "the prompt must name the only list dependency_paper_ids may come from"
    )
    db.close()


# --- P0-1a (task d6f64087): novelty_confidence must reach the gate decision ---

def _confirmed_audit(novelty_confidence):
    return SimpleNamespace(
        audit_result="confirmed", remaining_delta="untested state changes",
        novelty_confidence=novelty_confidence, neighbor_paper_ids_json="[]")


def test_novelty_gate_fails_confirmed_audit_with_very_low_confidence():
    """confirmed + very low novelty_confidence used to PASS — the number was
    printed in the rationale but never used in the decision, so three
    interventions reached tier A on an audit that itself reported 30%
    confidence in the gap's novelty. (Exactly-0.3 audits are caught earlier:
    the audit-side guard downgrades confirmed+<=0.4 before surviving, so the
    intervention-side FAIL band mainly protects legacy/resumed gaps.)"""
    gap = SimpleNamespace(provenance_status="complete")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, _confirmed_audit(0.2), ["e1", "e2"],
                                  _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"]["novelty"] == "FAIL"


def test_novelty_gate_warns_confirmed_audit_with_low_confidence():
    gap = SimpleNamespace(provenance_status="complete")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, _confirmed_audit(0.45), ["e1", "e2"],
                                  _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"]["novelty"] == "WARN"
    # WARN alone caps the tier at B (needs confirmation), never A.
    from app.agent.steps.generate_interventions import _compute_confidence_tier
    assert _compute_confidence_tier(gap, result["gate_statuses"]) == "B"


def test_novelty_gate_passes_confirmed_audit_with_solid_confidence():
    gap = SimpleNamespace(provenance_status="complete")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, _confirmed_audit(0.7), ["e1", "e2"],
                                  _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"]["novelty"] == "PASS"


def test_novelty_gate_keeps_legacy_audit_without_confidence_passing():
    """Legacy audits without a novelty_confidence value must not retro-fail."""
    gap = SimpleNamespace(provenance_status="complete")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(gap, SimpleNamespace(
        audit_result="confirmed", remaining_delta="untested state changes",
        novelty_confidence=None, neighbor_paper_ids_json="[]"), ["e1", "e2"],
        _candidate("Add evaluation protocol", "low cost"), contract)
    assert result["gate_statuses"]["novelty"] == "PASS"

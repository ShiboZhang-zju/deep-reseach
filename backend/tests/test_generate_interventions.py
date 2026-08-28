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
    """Lens-era mock: serves the first lens call and returns empty lists for
    the remaining lenses ("this lens does not fit") — keeping the single-gap
    single-intervention shape the older tests assert on."""

    def __init__(self, dependency_paper_ids):
        self.dependency_paper_ids = dependency_paper_ids
        self.calls = 0
        self.user_prompts = []

    async def chat_json(self, messages, schema):
        from app.agent.steps.generate_interventions import InterventionList

        self.calls += 1
        self.system_prompt = messages[0]["content"]
        self.user_prompts.append(messages[-1]["content"])
        if self.calls > 1:
            return InterventionList(interventions=[])
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


# --- Lens fan-out (2026-08-28, ARIS-inspired breadth stage) ---

class _PerLensLLM:
    """One intervention per lens call; records which lens each call served."""

    def __init__(self):
        self.user_prompts = []

    async def chat_json(self, messages, schema):
        from app.agent.steps.generate_interventions import InterventionList

        self.user_prompts.append(messages[-1]["content"])
        import re
        m = re.search(r"Lens: ([a-z_]+)", messages[-1]["content"])
        lens = m.group(1) if m else "unknown"
        return InterventionList(interventions=[InterventionSchema(
            intervention_type="method",
            failure_mechanism=f"{lens} targets a distinct failure surface.",
            proposed_intervention=f"Intervention generated from the {lens} lens.",
            intermediate_effect="separates state changes",
            measurable_outcome="state-change accuracy",
            implementation_cost="low cost",
            dependency_paper_ids=[],
            mechanism_confidence=0.8,
        )])


@pytest.mark.asyncio
async def test_lens_fanout_calls_every_lens_once(temp_db):
    """One LLM call per (gap, lens): six lenses, six independent candidate
    sources 鈥?the structural fix for single-call local convergence (task
    d6f64087: 2 of 3 single-call interventions tested the same hypothesis)."""
    import re
    from app.agent.state import ResearchState
    from app.agent.steps.generate_interventions import (
        _INTERVENTION_LENSES, generate_interventions)
    from app.db.models import AgentTrace, InterventionCandidate

    db = temp_db()
    task, contract, gap, neighbour = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    llm = _PerLensLLM()

    result = await generate_interventions(db, state, llm, task.id)

    lens_names = [re.search(r"Lens: ([a-z_]+)", p).group(1) for p in llm.user_prompts]
    assert lens_names == [name for name, _ in _INTERVENTION_LENSES], (
        "every lens must get exactly one independent call, in order")
    # Each lens contributed one distinct candidate.
    assert len(result.intervention_ids) == len(_INTERVENTION_LENSES)
    mechanisms = [db.get(InterventionCandidate, iid).failure_mechanism
                  for iid in result.intervention_ids]
    assert len(set(mechanisms)) == len(mechanisms), (
        "lens-sourced candidates must stay distinguishable")
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "generate_interventions").one()
    fanout = json.loads(trace.output_json)["lens_fanout"]
    assert [f["lens"] for f in fanout] == [name for name, _ in _INTERVENTION_LENSES]
    assert all(f["candidate_count"] == 1 for f in fanout)
    db.close()


@pytest.mark.asyncio
async def test_lens_that_does_not_fit_returns_empty(temp_db):
    """A lens with no sound intervention must return an empty list 鈥?padding
    would reintroduce the single-call convergence the fan-out exists to break."""
    from app.agent.state import ResearchState
    from app.agent.steps.generate_interventions import generate_interventions
    from app.db.models import AgentTrace

    class _EmptyLensLLM(_PerLensLLM):
        async def chat_json(self, messages, schema):
            from app.agent.steps.generate_interventions import InterventionList
            self.user_prompts.append(messages[-1]["content"])
            return InterventionList(interventions=[])

    db = temp_db()
    task, contract, gap, neighbour = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    llm = _EmptyLensLLM()

    result = await generate_interventions(db, state, llm, task.id)

    assert result.intervention_ids == []
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "generate_interventions").one()
    fanout = json.loads(trace.output_json)["lens_fanout"]
    assert len(fanout) == 6
    assert all(f["candidate_count"] == 0 for f in fanout)
    db.close()


@pytest.mark.asyncio
async def test_lens_candidates_are_capped_per_lens(temp_db):
    """A lens that ignores the 1-2 instruction cannot flood the pool: at most
    _MAX_CANDIDATES_PER_LENS candidates survive from any single lens."""
    from app.agent.state import ResearchState
    from app.agent.steps.generate_interventions import (
        _MAX_CANDIDATES_PER_LENS, generate_interventions)
    from app.db.models import AgentTrace

    class _GreedyLensLLM(_PerLensLLM):
        async def chat_json(self, messages, schema):
            from app.agent.steps.generate_interventions import InterventionList
            self.user_prompts.append(messages[-1]["content"])
            import re
            m = re.search(r"Lens: ([a-z_]+)", messages[-1]["content"])
            lens = m.group(1) if m else "unknown"
            return InterventionList(interventions=[
                InterventionSchema(
                    intervention_type="method",
                    failure_mechanism=f"{lens} surface {i}",
                    proposed_intervention=f"{lens} intervention {i}",
                    intermediate_effect="separates state changes",
                    measurable_outcome="state-change accuracy",
                    implementation_cost="low cost",
                    dependency_paper_ids=[],
                    mechanism_confidence=0.8,
                ) for i in range(3)])

    db = temp_db()
    task, contract, gap, neighbour = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    llm = _GreedyLensLLM()

    result = await generate_interventions(db, state, llm, task.id)

    assert len(result.intervention_ids) == 6 * _MAX_CANDIDATES_PER_LENS
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "generate_interventions").one()
    fanout = json.loads(trace.output_json)["lens_fanout"]
    assert all(f["candidate_count"] == _MAX_CANDIDATES_PER_LENS for f in fanout)
    db.close()

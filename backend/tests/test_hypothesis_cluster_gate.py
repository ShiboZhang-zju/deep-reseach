"""P2-A hypothesis-cluster gate tests (task 23ec8f20 follow-up).

The gate groups one gap's passed interventions by the experimental HYPOTHESIS
they test. Merged clusters yield ONE idea + ONE experiment with the variant
mechanisms folded into the baselines; degradation (LLM failure, ID mismatch,
low confidence) falls back to one-idea-per-intervention.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.steps.generate_minimal_experiments import (
    MinimalExperimentSchema,
    _cluster_interventions_by_hypothesis,
    _derive_scenario_atoms,
    generate_minimal_experiments,
)
from app.schemas.schemas import IdeaScore


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
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


class _ClusterMockLLM:
    """chat_json dispatch by schema name with a configurable cluster response."""

    def __init__(self, cluster_response=None, plan_texts=None):
        self.cluster_response = cluster_response
        self.plan_texts = plan_texts or []
        self.cluster_calls = 0
        self.plan_prompts = []

    async def chat_json(self, messages, schema):
        name = schema.__name__
        if name == "HypothesisClusterListSchema":
            self.cluster_calls += 1
            if isinstance(self.cluster_response, Exception):
                raise self.cluster_response
            return self.cluster_response
        if name == "MinimalExperimentSchema":
            self.plan_prompts.append(messages[-1]["content"])
            text = " ".join(self.plan_texts) if self.plan_texts else "filtering stylistic logical fixes"
            return MinimalExperimentSchema(
                title="Filtering stylistic fixes in self-correction",
                summary="A minimal experiment filtering stylistic-only corrections.",
                hypothesis="Filtering stylistic-only corrections reduces functional regressions.",
                dataset=f"HumanEval with injected stylistic noise covering: {text}",
                baselines="no-correction, apply-all, heuristic skip-stylistic",
                metrics="pass@1 delta, regression rate",
                model_spec=f"small model, logical fixes only: {text}",
                dataset_provenance="public dataset plus synthetic stylistic injection",
                oracle="execution engine verifier plus manual adjudication",
                statistical_analysis="paired bootstrap over per-problem outcomes",
                resource_budget="CPU only, under two hours",
                scenario_atoms=["stylistic", "logical"],
                controls=["same decoding budget"],
                steps=[f"inject stylistic noise: {text}", "run filtering arms and measure regressions"],
                success_condition="regression rate drops significantly",
                falsification_condition="no measurable regression difference",
                risks=f"verifier environment flakiness: {text}",
            )
        if name == "IdeaScore":
            return IdeaScore(novelty=0.7, feasibility=0.9, significance=0.7,
                             evidence_support=0.8, differentiation=0.7,
                             experimentability=0.9, potential_impact=0.7, risk=0.2,
                             reason="clustered hypothesis")
        raise AssertionError(f"unexpected schema: {name}")


def _seed(db, n_interventions=2):
    from app.db.models import (
        GapCandidate, GapPhenomenonPlan, InterventionCandidate,
        ResearchContract, ResearchTask,
    )

    task = ResearchTask(user_input="code self-correction regressions")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Self-Correction Filtering",
                                status="active", version=1, input_hash="v1",
                                gpu_available=False, allow_model_training=False)
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="missing_method",
        description="Stylistic edits cause regressions in code repair.",
        target_setting="small language models",
        observed_problem="Stylistic edits cause regressions in code repair.",
        missing_capability="No mechanism distinguishes stylistic from logical edits.",
        claimed_delta="A filter that applies only logical fixes reduces regressions.",
        testable_hypothesis="Filtering stylistic-only corrections reduces functional regressions.",
        status="surviving", mining_round=1, version=1,
        mining_policy_version="gap-mining-v5", provenance_status="complete",
    )
    db.add(gap)
    db.flush()
    db.add(GapPhenomenonPlan(
        task_id=task.id, contract_id=contract.id, gap_id=gap.id,
        phenomenon="Stylistic-only corrections raise regression rate.",
        mechanism_under_test="stylistic versus logical edit separation",
        supports_gap_claim="style-only edits regress functionality",
        critical_unknown="regression source separation",
        expected_observation="higher regression rate for stylistic edits",
        alternative_explanation="any regeneration degrades output",
        comparator="H0 no difference; H1 stylistic-only higher regression",
        oracle_experiment="run executable tests per correction type",
        kill_criterion="relative difference under 2 percent",
        kill_criterion_basis="noise floor",
        measurement="regression rate",
    ))
    db.flush()
    mechanisms = [
        ("diff risk classification", "Classify diffs by regression risk and gate execution."),
        ("two-phase feedback segregation", "Segregate feedback into logical-first, style-later phases."),
    ]
    interventions = []
    for i in range(n_interventions):
        mech, prop = mechanisms[i % len(mechanisms)]
        interventions.append(InterventionCandidate(
            task_id=task.id, gap_id=gap.id, contract_id=contract.id,
            intervention_type="method", failure_mechanism=mech,
            proposed_intervention=prop,
            intermediate_effect="only logical fixes reach the verifier",
            measurable_outcome="regression rate drop versus apply-all",
            implementation_cost="low", mechanism_confidence=0.8,
            evidence_gate="PASS", novelty_gate="PASS", feasibility_gate="PASS",
            gate_rationale_json="{}", confidence_tier="A", status="passed",
        ))
    db.add_all(interventions)
    db.commit()
    phenomenon = db.query(GapPhenomenonPlan).filter(GapPhenomenonPlan.gap_id == gap.id).first()
    return task, contract, gap, interventions, phenomenon


def _cluster_of(primary, variants, confidence=0.95):
    from app.agent.steps.generate_minimal_experiments import (
        HypothesisClusterListSchema, HypothesisClusterSchema,
    )
    return HypothesisClusterListSchema(clusters=[HypothesisClusterSchema(
        hypothesis="Filtering stylistic-only corrections reduces functional regressions.",
        primary_intervention_id=primary.id,
        variant_intervention_ids=[v.id for v in variants],
        differentiation_rationale="Both mechanisms test the same filtering hypothesis.",
        confidence=confidence,
    )])


@pytest.mark.asyncio
async def test_same_hypothesis_interventions_merge_into_one_idea(temp_db):
    """The 23ec8f20 shape: two tier-A interventions, same hypothesis -> ONE
    executable_candidate idea; variants ride along as ablation arms."""
    from app.db.models import AgentTrace, ResearchIdea
    from app.agent.state import ResearchState

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _ClusterMockLLM(cluster_response=_cluster_of(interventions[0], [interventions[1]]),
                          plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    result = await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 1, "merged cluster must yield exactly one idea"
    assert result.idea_ids == [ideas[0].id]
    assert "mechanism variant" in ideas[0].motivation
    assert ideas[0].decision == "executable_candidate"
    # The variant mechanism must be injected into the experiment prompt.
    assert llm.cluster_calls == 1
    assert any("two-phase feedback segregation" in p for p in llm.plan_prompts), (
        "variant mechanism must reach the experiment prompt as an ablation arm")
    # The cluster decision must leave a trace.
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "hypothesis_cluster",
    ).all()
    assert trace and json.loads(trace[0].output_json)["clustered"] is True
    db.close()


@pytest.mark.asyncio
async def test_cluster_llm_failure_degrades_to_per_intervention_ideas(temp_db):
    """Infrastructure failure must fail open: no merge, pre-v2 behaviour."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _ClusterMockLLM(cluster_response=RuntimeError("gateway 502"), plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    result = await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 2, "degraded gate keeps one idea per intervention"
    assert len(result.idea_ids) == 2
    db.close()


@pytest.mark.asyncio
async def test_low_confidence_merge_is_split_back(temp_db):
    """confidence < 0.7: no merge — false merges cost idea output, so the
    conservative direction on uncertainty is split."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _ClusterMockLLM(cluster_response=_cluster_of(interventions[0], [interventions[1]],
                                                       confidence=0.5), plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 2
    db.close()


@pytest.mark.asyncio
async def test_cluster_with_unknown_intervention_id_degrades(temp_db):
    """A cluster response that does not cover the supplied IDs exactly once
    is rejected wholesale -> degrade, never trust a partial partition."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea
    from app.agent.steps.generate_minimal_experiments import (
        HypothesisClusterListSchema, HypothesisClusterSchema,
    )

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    bad = HypothesisClusterListSchema(clusters=[HypothesisClusterSchema(
        hypothesis="Filtering stylistic-only corrections reduces functional regressions.",
        primary_intervention_id="not-a-real-id",
        variant_intervention_ids=[],
        differentiation_rationale="Coverage is incomplete.",
        confidence=0.95,
    )])
    llm = _ClusterMockLLM(cluster_response=bad, plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 2
    db.close()


@pytest.mark.asyncio
async def test_single_intervention_skips_cluster_llm(temp_db):
    """One intervention = one trivial cluster; no LLM call is spent."""
    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    llm = _ClusterMockLLM(cluster_response=None)
    clusters = await _cluster_interventions_by_hypothesis(llm, phenomenon, interventions)
    assert clusters is not None
    assert len(clusters) == 1
    assert clusters[0]["primary"].id == interventions[0].id
    assert clusters[0]["variants"] == []
    assert llm.cluster_calls == 0
    db.close()

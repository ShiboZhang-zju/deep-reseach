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
    EXPERIMENT_GENERATION_POLICY_VERSION,
    IdeaNoveltyQueriesSchema,
    IdeaNoveltyVerdictSchema,
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


class _NoveltyMockLLM(_ClusterMockLLM):
    """Extends the cluster mock with the P2-C novelty-check LLM branches."""

    def __init__(self, cluster_response, plan_texts,
                 novelty_queries=None, novelty_verdict=None):
        super().__init__(cluster_response=cluster_response, plan_texts=plan_texts)
        self.novelty_queries = novelty_queries
        self.novelty_verdict = novelty_verdict

    async def chat_json(self, messages, schema):
        name = schema.__name__
        if name == "IdeaNoveltyQueriesSchema":
            if isinstance(self.novelty_queries, Exception):
                raise self.novelty_queries
            if self.novelty_queries is None:
                return None
            return IdeaNoveltyQueriesSchema(queries=self.novelty_queries)
        if name == "IdeaNoveltyVerdictSchema":
            if isinstance(self.novelty_verdict, Exception):
                raise self.novelty_verdict
            if self.novelty_verdict is None:
                return None
            return self.novelty_verdict
        return await super().chat_json(messages, schema)


def _patch_retrieval(monkeypatch, prior_art_id):
    """Replace external retrieval with a fake that surfaces one prior-art paper."""
    from app.db.models import SearchQueryPaper

    async def fake_search(db, state, executions, task_id, round_num):
        for e in executions:
            db.add(SearchQueryPaper(query_id=e.query_id, paper_id=prior_art_id,
                                     rank=1, source="test", is_new_for_task=False))
        db.commit()
        return (1, 1, [])

    monkeypatch.setattr("app.agent.steps.search_papers.search_and_save_papers", fake_search)


def _seed_prior_art(db, title="Directly implements the diff-risk classifier"):
    from app.db.models import Paper
    paper = Paper(title=title, abstract="Prior art implementing the exact method.",
                  citation_count=1)
    db.add(paper)
    db.commit()
    return paper


@pytest.mark.asyncio
async def test_novelty_check_demotes_already_implemented(temp_db, monkeypatch):
    """P2-C: prior art directly implementing the method demotes the idea to
    conditional_review with METHOD_ALREADY_PUBLISHED and links the paper."""
    from app.agent.state import ResearchState
    from app.db.models import AgentTrace, Paper, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    prior_art = _seed_prior_art(db)
    _patch_retrieval(monkeypatch, prior_art.id)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _NoveltyMockLLM(
        cluster_response=None, plan_texts=atoms,
        novelty_queries=["diff risk classifier correction filtering",
                         "patch classification regression risk",
                         "program repair patch filtering"],
        novelty_verdict=IdeaNoveltyVerdictSchema(
            already_implemented=True, evidence_paper_id=prior_art.id,
            rationale="The paper builds the same classifier for the same purpose."),
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    assert idea.decision == "conditional_review"
    assert "METHOD_ALREADY_PUBLISHED" in json.loads(idea.quality_reason_codes_json or "[]")
    assert prior_art.id in json.loads(idea.related_paper_ids_json or "[]")
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id, AgentTrace.step_name == "idea_novelty_check").all()
    assert any(json.loads(t.output_json).get("verdict") == "already_implemented" for t in traces)
    db.close()


@pytest.mark.asyncio
async def test_novelty_check_passed_keeps_executable(temp_db, monkeypatch):
    """Related-but-different prior art must NOT demote."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    prior_art = _seed_prior_art(db, title="A survey of code correction (different technique)")
    _patch_retrieval(monkeypatch, prior_art.id)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _NoveltyMockLLM(
        cluster_response=None, plan_texts=atoms,
        novelty_queries=["diff risk classifier correction filtering"],
        novelty_verdict=IdeaNoveltyVerdictSchema(
            already_implemented=False, evidence_paper_id=None,
            rationale="Survey is related work, not the same mechanism."),
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    assert idea.decision == "executable_candidate"
    assert "METHOD_ALREADY_PUBLISHED" not in json.loads(idea.quality_reason_codes_json or "[]")
    db.close()


@pytest.mark.asyncio
async def test_novelty_check_degraded_never_demotes(temp_db, monkeypatch):
    """Infrastructure failure (query generation raises) degrades to a trace and
    keeps the executable decision — an outage is not evidence about the idea."""
    from app.agent.state import ResearchState
    from app.db.models import AgentTrace, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _NoveltyMockLLM(
        cluster_response=None, plan_texts=atoms,
        novelty_queries=RuntimeError("gateway 502"),
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    assert idea.decision == "executable_candidate"
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id, AgentTrace.step_name == "idea_novelty_check").all()
    assert any(json.loads(t.output_json).get("verdict") == "degraded" for t in traces)
    db.close()


def test_policy_version_bumped():
    """The experiment policy version encodes the novelty-check rules."""
    assert EXPERIMENT_GENERATION_POLICY_VERSION == "experiment-consistency-v6"


def _plan(**overrides):
    from app.agent.steps.generate_minimal_experiments import MinimalExperimentSchema
    base = dict(
        title="Diff-Risk Classification for Self-Correction Filtering",
        idea_method="We hypothesize that filtering stylistic corrections reduces regressions.",
        summary="Compare filtered vs unfiltered self-correction on HumanEval.",
        hypothesis="Filtering stylistic-only corrections reduces functional regression rate.",
        dataset="HumanEval with injected stylistic noise",
        baselines="apply-all corrections; heuristic filter",
        metrics="pass@1 delta and regression rate across 164 problems",
        model_spec="Qwen2.5-Coder-7B-Instruct",
        dataset_provenance="Inject stylistic edits verified function-preserving",
        oracle="Python execution engine plus 10% human adjudication",
        statistical_analysis="Paired t-test on per-problem pass/fail deltas (p<0.05).",
        resource_budget="CPU-only, <2 hours",
        scenario_atoms=["verifier"],
        controls=["no-correction", "apply-all"],
        steps=["Build dataset", "Run corrections", "Evaluate"],
        success_condition="Regression rate drops by >5% relative",
        falsification_condition="No significant difference",
        risks="Synthetic noise may not reflect real stylistic edits",
    )
    base.update(overrides)
    return MinimalExperimentSchema(**base)


def test_statistical_test_mismatch_on_ratio_metric_with_ttest():
    """Ratio-style paired metrics analyzed by a plain t-test without a
    non-parametric alternative must be flagged (E2E 2026-08-26 review)."""
    from app.agent.steps.generate_minimal_experiments import _validate_experiment_plan
    plan = _plan(statistical_analysis="Paired t-test on pass/fail outcomes (p<0.05).")
    failures = _validate_experiment_plan(plan, phenomenon=None, expected_atoms=[], gap_text="")
    assert "STATISTICAL_TEST_MISMATCH" in failures


def test_statistical_test_passes_with_nonparametric_or_nonratio():
    from app.agent.steps.generate_minimal_experiments import _validate_experiment_plan
    # Non-parametric alternative named -> clean.
    plan = _plan(statistical_analysis="McNemar's test on paired pass/fail outcomes.")
    failures = _validate_experiment_plan(plan, phenomenon=None, expected_atoms=[], gap_text="")
    assert "STATISTICAL_TEST_MISMATCH" not in failures
    # t-test but ratio wording absent from metrics -> clean (conservative).
    plan2 = _plan(statistical_analysis="Paired t-test (p<0.05).",
                  metrics="mean edit distance and latency milliseconds")
    failures2 = _validate_experiment_plan(plan2, phenomenon=None, expected_atoms=[], gap_text="")
    assert "STATISTICAL_TEST_MISMATCH" not in failures2


def test_model_scope_conflict_generalized():
    from app.agent.steps.generate_minimal_experiments import _validate_experiment_plan
    # Numeric cap from target_setting: <7B scope vs an 8B checkpoint.
    plan = _plan(model_spec="Llama-3-8B-Instruct")
    failures = _validate_experiment_plan(plan, phenomenon=None, expected_atoms=[],
                                         gap_text="study small models under 7B parameters in code generation")
    assert "MODEL_SCOPE_CONFLICT" in failures
    # SLM keyword scope: defaults to a 7B cap.
    plan2 = _plan(model_spec="Llama-3-13B-Instruct")
    failures2 = _validate_experiment_plan(plan2, phenomenon=None, expected_atoms=[],
                                          gap_text="applies to small language models generating code")
    assert "MODEL_SCOPE_CONFLICT" in failures2
    # Plural "SLMs" counts too (task 23ec8f20: gap target_setting said "SLMs or
    # static analyzers" while plans used Llama-3-8B).
    plan2b = _plan(model_spec="Llama-3-8B-Instruct")
    failures2b = _validate_experiment_plan(plan2b, phenomenon=None, expected_atoms=[],
                                           gap_text="Test-time self-correction using SLMs under compute budgets")
    assert "MODEL_SCOPE_CONFLICT" in failures2b
    # Within scope: 7B under a <7B cap is fine, and no scope wording means no check.
    plan3 = _plan()
    failures3 = _validate_experiment_plan(plan3, phenomenon=None, expected_atoms=[],
                                          gap_text="scope: models smaller than 7B")
    assert "MODEL_SCOPE_CONFLICT" not in failures3
    plan4 = _plan(model_spec="GPT-4o")
    failures4 = _validate_experiment_plan(plan4, phenomenon=None, expected_atoms=[], gap_text="")
    assert "MODEL_SCOPE_CONFLICT" not in failures4


@pytest.mark.asyncio
async def test_stale_ideas_superseded_on_regenerate(temp_db):
    """P2-B: a phase re-run soft-deletes the previous round's active ideas so the
    task surfaces one coherent generation (task 23ec8f20: v1/v2/v3 ideas
    coexisted side by side in the UI)."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    stale = ResearchIdea(task_id=task.id, title="Old round idea", description="old",
                         motivation="old", method_sketch="old",
                         expected_contribution="old", decision="executable_candidate",
                         idea_status="active", score_status="unscored",
                         related_paper_ids_json="[]")
    db.add(stale)
    db.commit()
    stale_id = stale.id

    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _ClusterMockLLM(cluster_response=None, plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    old = db.query(ResearchIdea).filter(ResearchIdea.id == stale_id).one()
    assert old.idea_status == "superseded"
    fresh = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task.id, ResearchIdea.idea_status == "active").all()
    assert len(fresh) == 1
    # P2-B: title prefix is stripped even if the model emits it.
    assert not fresh[0].title.lower().startswith("minimal experiment")


def test_normalize_idea_title_strips_prefix():
    from app.agent.steps.generate_minimal_experiments import _normalize_idea_title
    assert _normalize_idea_title("Minimal Experiment: Foo Bar") == "Foo Bar"
    assert _normalize_idea_title("  Diff-Risk Classification  ") == "Diff-Risk Classification"
    assert _normalize_idea_title("") == "Untitled experiment"

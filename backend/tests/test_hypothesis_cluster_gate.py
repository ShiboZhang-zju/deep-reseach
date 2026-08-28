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
    InterventionTriageItemSchema,
    InterventionTriageListSchema,
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


@pytest.fixture(autouse=True)
def _deterministic_embeddings(monkeypatch):
    """Deterministic embeddings for the whole file: identical texts map to the
    same one-hot vector (similarity 1.0), different texts to orthogonal ones
    (0.0). Keeps the sibling-merge guard exact and offline."""
    import importlib

    gme = importlib.import_module("app.agent.steps.generate_minimal_experiments")

    def fake_embed(texts):
        seen = {}

        def vec_of(t):
            if t not in seen:
                seen[t] = len(seen)
            v = [0.0] * 64
            v[seen[t] % 64] = 1.0
            return v

        return [vec_of(t) for t in texts]

    monkeypatch.setattr(gme.embedding_service, "embed_texts", fake_embed)


class _ClusterMockLLM:
    """chat_json dispatch by schema name with a configurable cluster response."""

    def __init__(self, cluster_response=None, plan_texts=None, idea_contribution=""):
        self.cluster_response = cluster_response
        self.plan_texts = plan_texts or []
        self.idea_contribution = idea_contribution
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
                idea_method="Classify corrections by type and gate execution on logical-only fixes.",
                idea_contribution=self.idea_contribution,
                core_factor=f"correction-type filter (arm set {len(self.plan_prompts)})",
                core_operation="toggle_filter",
                core_contrast="logical_only_vs_apply_all",
                expected_signature=(
                    "Regression rate drops in the logical-only arm, unchanged "
                    f"pass@1 (observation {len(self.plan_prompts)})."),
                mechanism_being_tested=(
                    "Stylistic corrections carry functional regression risk "
                    "the filter removes."),
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
        ("execution trace verification", "Verify executed edits against runtime traces before commit."),
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
async def test_idea_contribution_differentiated_from_measurable_outcome(temp_db):
    """P2-a (2026-08-28): expected_contribution states what the experiment
    would establish (knowledge gain); it used to be the intervention's
    measurable_outcome copied verbatim, which only names the metric."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    contribution = ("Establishes that separating stylistic from logical edits "
                    "causally reduces functional regressions — a boundary "
                    "condition for any correction-filtering method.")
    llm = _ClusterMockLLM(cluster_response=None, plan_texts=atoms,
                          idea_contribution=contribution)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    assert idea.expected_contribution == contribution
    assert idea.expected_contribution != interventions[0].measurable_outcome
    db.close()


@pytest.mark.asyncio
async def test_idea_contribution_falls_back_when_plan_omits_it(temp_db):
    """An empty idea_contribution field degrades to the intervention's
    measurable_outcome — differentiation is preferred, never mandatory."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _ClusterMockLLM(cluster_response=None, plan_texts=atoms,
                          idea_contribution="")
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    assert idea.expected_contribution == interventions[0].measurable_outcome
    db.close()


@pytest.mark.asyncio
async def test_related_papers_reselected_by_mechanism_relevance(temp_db, monkeypatch):
    """P2-b (2026-08-28): related_paper_ids come from the novelty check's
    mechanism-relevant selection instead of the gap audit's neighbour set —
    the neighbours are gap-relevant, not necessarily mechanism-relevant."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=1)
    prior_art = _seed_prior_art(db, title="Prior art touching the same mechanism")
    _patch_retrieval(monkeypatch, prior_art.id)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _NoveltyMockLLM(
        cluster_response=None, plan_texts=atoms,
        novelty_queries=["diff risk classifier correction filtering"],
        novelty_verdict=IdeaNoveltyVerdictSchema(
            already_implemented=False, evidence_paper_id=None,
            mechanism_relevant_paper_ids=[prior_art.id],
            rationale="Same mechanism family; not a direct implementation.",
        ),
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).one()
    related = json.loads(idea.related_paper_ids_json or "[]")
    assert related == [prior_art.id], (
        "the mechanism-relevant paper replaces the audit-neighbour inheritance"
    )
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
    assert EXPERIMENT_GENERATION_POLICY_VERSION == "experiment-consistency-v11"


def _plan(**overrides):
    from app.agent.steps.generate_minimal_experiments import MinimalExperimentSchema
    base = dict(
        title="Diff-Risk Classification for Self-Correction Filtering",
        idea_method="We hypothesize that filtering stylistic corrections reduces regressions.",
        summary="Compare filtered vs unfiltered self-correction on HumanEval.",
        hypothesis="Filtering stylistic-only corrections reduces functional regression rate.",
        core_factor="correction-type filter",
        core_operation="toggle_filter",
        core_contrast="logical_only_vs_apply_all",
        expected_signature="Regression rate drops in the logical-only arm, unchanged pass@1.",
        mechanism_being_tested="Stylistic corrections carry functional regression risk.",
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
    assert any(f.startswith("MODEL_SCOPE_CONFLICT") for f in failures)
    # SLM keyword scope: defaults to a 7B cap.
    plan2 = _plan(model_spec="Llama-3-13B-Instruct")
    failures2 = _validate_experiment_plan(plan2, phenomenon=None, expected_atoms=[],
                                          gap_text="applies to small language models generating code")
    assert any(f.startswith("MODEL_SCOPE_CONFLICT") for f in failures2)
    # Plural "SLMs" counts too (task 23ec8f20: gap target_setting said "SLMs or
    # static analyzers" while plans used Llama-3-8B).
    plan2b = _plan(model_spec="Llama-3-8B-Instruct")
    failures2b = _validate_experiment_plan(plan2b, phenomenon=None, expected_atoms=[],
                                           gap_text="Test-time self-correction using SLMs under compute budgets")
    assert any(f.startswith("MODEL_SCOPE_CONFLICT") for f in failures2b)
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


class _TriageMockLLM(_NoveltyMockLLM):
    """Extends the novelty mock with the independent-triage LLM branch."""

    def __init__(self, cluster_response, plan_texts, triage_items=None, **kwargs):
        super().__init__(cluster_response=cluster_response, plan_texts=plan_texts, **kwargs)
        self.triage_items = triage_items
        self.triage_calls = 0

    async def chat_json(self, messages, schema):
        name = schema.__name__
        if name == "InterventionTriageListSchema":
            self.triage_calls += 1
            if isinstance(self.triage_items, Exception):
                raise self.triage_items
            if self.triage_items is None:
                return None
            return InterventionTriageListSchema(triage=self.triage_items)
        return await super().chat_json(messages, schema)


def _triage_item(cluster_primary_intervention_id, priority):
    return InterventionTriageItemSchema(
        cluster_primary_intervention_id=cluster_primary_intervention_id,
        best_case="A clean causal answer on which mechanism explains the regression.",
        strongest_objection="The synthetic noise may not reflect real stylistic drift.",
        alternative_explanation="Verifier flakiness alone could produce the delta.",
        simplicity=0.7, information_value=0.8, priority=priority,
    )


@pytest.mark.asyncio
async def test_triage_funds_top_k_and_parks_rest(temp_db):
    """Independent scientific triage (ARIS-inspired): with 3 clusters and
    _TRIAGE_TOP_K=2, the top-2 by priority get experiments; the third persists
    as research_direction_only with the reviewer's annotations 鈥?never a
    silent drop."""
    from app.agent.state import ResearchState
    from app.db.models import AgentTrace, ExperimentPlan, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=3)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _TriageMockLLM(
        cluster_response=None, plan_texts=atoms,
        triage_items=[
            _triage_item(interventions[0].id, 2),
            _triage_item(interventions[1].id, 3),   # parked
            _triage_item(interventions[2].id, 1),
        ],
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    result = await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    executable = [i for i in ideas if i.decision == "executable_candidate"]
    parked = [i for i in ideas if i.decision == "research_direction_only"]
    assert len(executable) == 2
    assert len(parked) == 1
    assert parked[0].intervention_id == interventions[1].id
    codes = json.loads(parked[0].quality_reason_codes_json or "[]")
    assert "DEFERRED_BY_SCIENTIFIC_TRIAGE" in codes
    assert "TRIAGE_PRIORITY_3" in codes
    assert "Deferred by independent scientific triage" in parked[0].motivation
    assert "Strongest objection" in parked[0].motivation
    # Funded ideas carry the reviewer's note.
    assert all("[Triage note" in i.motivation for i in executable)
    # Only the funded clusters generated experiments.
    experiments = db.query(ExperimentPlan).filter(
        ExperimentPlan.task_id == task.id).all()
    assert len(experiments) == 2
    assert result.direction_only_idea_ids == [parked[0].id]
    assert llm.triage_calls == 1
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id, AgentTrace.step_name == "intervention_triage").all()
    assert any(json.loads(t.output_json).get("verdict") == "ranked" for t in traces)
    db.close()


@pytest.mark.asyncio
async def test_triage_failure_funds_all_clusters(temp_db):
    """Reviewer outage is not evidence against a cluster: triage failure keeps
    the pre-triage behaviour (every cluster proceeds)."""
    from app.agent.state import ResearchState
    from app.db.models import AgentTrace, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=3)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _TriageMockLLM(
        cluster_response=None, plan_texts=atoms,
        triage_items=RuntimeError("reviewer gateway down"),
    )
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 3
    assert all(i.decision == "executable_candidate" for i in ideas)
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id, AgentTrace.step_name == "intervention_triage").all()
    assert any(json.loads(t.output_json).get("verdict") == "degraded" for t in traces)
    db.close()


@pytest.mark.asyncio
async def test_triage_skipped_when_clusters_within_budget(temp_db):
    """<= _TRIAGE_TOP_K clusters: no reviewer call is spent 鈥?triage only
    arbitrates when there is an actual budget contest."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    atoms = _derive_scenario_atoms(gap, None, phenomenon)
    llm = _TriageMockLLM(cluster_response=None, plan_texts=atoms)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    assert llm.triage_calls == 0
    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 2
    db.close()


# --- Sibling collapse guard (run7 regression, 2026-08-28) ---

def _identity_plan(title, factor, operation, signature=None, contrast="self_vs_external"):
    return MinimalExperimentSchema(
        title=title,
        summary=f"Minimal experiment for {title}.",
        hypothesis="Filtering stylistic-only corrections reduces functional regressions.",
        idea_method="Classify corrections by type and gate execution on logical-only fixes.",
        idea_contribution="Establishes the causal share of stylistic edits in regressions.",
        core_factor=factor,
        core_operation=operation,
        core_contrast=contrast,
        expected_signature=signature or "Self-labelled hallucinations receive higher ratings.",
        mechanism_being_tested="Displayed source attribution overrides fact-checking.",
        dataset="HumanEval with injected stylistic noise",
        dataset_provenance="public dataset plus synthetic stylistic injection",
        baselines="no-correction, apply-all, heuristic skip-stylistic",
        metrics="pass@1 delta and regression rate",
        model_spec="small model",
        oracle="execution engine verifier",
        statistical_analysis="McNemar paired test on regression rate deltas",
        resource_budget="CPU only",
        success_condition="regression rate drops significantly",
        falsification_condition="no measurable regression difference",
        scenario_atoms=["stylistic", "logical"],
        controls=["same decoding budget"],
        steps=["inject stylistic noise", "run filtering arms"],
        risks="verifier flakiness",
    )


class _SequentialPlanLLM(_ClusterMockLLM):
    """Serves a queue of plans (one per funded cluster call)."""

    def __init__(self, plans, cluster_response=None):
        super().__init__(cluster_response=cluster_response,
                         idea_contribution="Knowledge gain statement.")
        self.plans = list(plans)

    async def chat_json(self, messages, schema):
        name = schema.__name__
        if name == "MinimalExperimentSchema" and self.plans:
            self.plan_prompts.append(messages[-1]["content"])
            return self.plans.pop(0)
        return await super().chat_json(messages, schema)


@pytest.mark.asyncio
async def test_sibling_manipulation_collapse_merges_into_one_idea(temp_db, monkeypatch):
    """run7 regression case: P1 (label swap) and P2 (nominally anonymous
    evaluation) both converged onto near-identical 'Self vs External'
    experiments — two lookalike executable ideas. Same factor + same
    operation + similar signature/mechanism = DUPLICATE: merge the second
    plan into the first idea as a condition variant — one idea, two
    experiments, collapse recorded in the trace."""
    from app.agent.state import ResearchState
    from app.db.models import AgentTrace, ExperimentPlan, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    llm = _SequentialPlanLLM([
        _identity_plan("Attribution swap testing",
                       "displayed source attribution", "swap_label"),
        _identity_plan("Blind provenance evaluation",
                       "displayed source attribution", "swap_label"),  # re-worded duplicate
    ], cluster_response=None)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    experiments = db.query(ExperimentPlan).filter(
        ExperimentPlan.task_id == task.id).all()
    assert len(ideas) == 1, "a collapsed sibling must not spawn a lookalike idea"
    assert len(experiments) == 2, "both mechanism experiments survive, on ONE idea"
    assert all(e.idea_id == ideas[0].id for e in experiments)
    variant = json.loads(
        [e for e in experiments if json.loads(e.steps_json).get("condition_variant")][0].steps_json)
    assert variant["condition_variant"] is True
    assert variant["sibling_relation"] == "DUPLICATE"
    assert "Condition variant merged" in ideas[0].motivation
    traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "idea_quality_gate").all()
    merge_traces = [json.loads(t.output_json) for t in traces
                    if json.loads(t.output_json).get("status") == "merged_condition_variant"]
    assert merge_traces and "SIBLING_MANIPULATION_COLLAPSE" in merge_traces[0]["reason_codes"]
    # The second cluster's prompt carried the relaxed sibling constraint.
    assert any("does NOT need to" in p for p in llm.plan_prompts), llm.plan_prompts
    db.close()


@pytest.mark.asyncio
async def test_same_factor_different_operation_is_complementary(temp_db, monkeypatch):
    """run7 review round 2: swap_label and remove_label on the SAME factor are
    complementary causal questions (does the label cause the bias / does
    removing it eliminate the bias), NOT duplicates — the v10 concatenated
    similarity would have merged them. They must land as two separate
    experiments under ONE idea."""
    from app.agent.state import ResearchState
    from app.db.models import ExperimentPlan, ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    llm = _SequentialPlanLLM([
        _identity_plan("Attribution swap causal test",
                       "displayed source attribution", "swap_label"),
        _identity_plan("Anonymous attribution ablation",
                       "displayed source attribution", "remove_label"),
    ], cluster_response=None)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    experiments = db.query(ExperimentPlan).filter(
        ExperimentPlan.task_id == task.id).all()
    assert len(ideas) == 1, "complementary designs share one idea"
    assert len(experiments) == 2
    assert all(e.idea_id == ideas[0].id for e in experiments)
    ops = {json.loads(e.steps_json).get("core_operation") for e in experiments}
    assert ops == {"swap_label", "remove_label"}, "both operations preserved"
    assert "[Complementary experiment]" in ideas[0].motivation
    assert "[Condition variant merged]" not in ideas[0].motivation
    db.close()


@pytest.mark.asyncio
async def test_distinct_factors_stay_separate_ideas(temp_db, monkeypatch):
    """Different factors (displayed attribution vs actual provenance) are
    genuinely separate studies — the guard merges collapse and folds
    complementary pairs, not independent factors."""
    from app.agent.state import ResearchState
    from app.db.models import ResearchIdea

    db = temp_db()
    task, contract, gap, interventions, phenomenon = _seed(db, n_interventions=2)
    llm = _SequentialPlanLLM([
        _identity_plan("Attribution swap testing",
                       "displayed source attribution", "swap_label"),
        _identity_plan("Actual provenance blind test",
                       "actual generation provenance", "blind_judge"),
    ], cluster_response=None)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=1)

    await generate_minimal_experiments(db, state, llm, task.id)

    ideas = db.query(ResearchIdea).filter(ResearchIdea.task_id == task.id).all()
    assert len(ideas) == 2
    assert not any("[Condition variant merged]" in (i.motivation or "")
                   or "[Complementary experiment]" in (i.motivation or "")
                   for i in ideas)
    db.close()


def test_sibling_relation_threshold_boundaries(monkeypatch):
    """The relation thresholds themselves, not just the code branches (run7
    review round 2): factor 0.85 boundary flips INDEPENDENT<->same-factor,
    and with same factor+operation the signature 0.85 boundary flips
    COMPLEMENTARY<->DUPLICATE."""
    import importlib

    gme = importlib.import_module("app.agent.steps.generate_minimal_experiments")

    # Vectors with an exact cosine: [1,0] vs [cos, sin].
    import math

    def vec_for(cos_target: float):
        return [cos_target, math.sqrt(max(0.0, 1.0 - cos_target * cos_target))]

    def patch_with(sim_map):
        """sim_map: pair index (0=factor,1=operation,2=signature,3=mechanism)
        -> cosine value for (plan vs sibling)."""
        plan_texts = ["p_factor", "s_factor", "p_op", "s_op",
                      "p_sig", "s_sig", "p_mech", "s_mech"]

        def fake_embed(texts):
            out = []
            for t in texts:
                if t == "p_factor":
                    out.append([1.0, 0.0])
                elif t == "s_factor":
                    out.append(vec_for(sim_map.get(0, 1.0)))
                elif t == "p_op":
                    out.append([1.0, 0.0])
                elif t == "s_op":
                    out.append(vec_for(sim_map.get(1, 1.0)))
                elif t == "p_sig":
                    out.append([1.0, 0.0])
                elif t == "s_sig":
                    out.append(vec_for(sim_map.get(2, 1.0)))
                elif t == "p_mech":
                    out.append([1.0, 0.0])
                elif t == "s_mech":
                    out.append(vec_for(sim_map.get(3, 1.0)))
                else:  # one-hot fallback for unrelated texts
                    out.append([0.0] * 64)
            return out

        monkeypatch.setattr(gme.embedding_service, "embed_texts", fake_embed)

    from types import SimpleNamespace

    plan = SimpleNamespace(core_factor="p_factor", core_operation="p_op",
                           expected_signature="p_sig",
                           mechanism_being_tested="p_mech")
    sibling = {
        "core_factor": "s_factor", "core_operation": "s_op",
        "core_contrast": "c", "expected_signature": "s_sig",
        "mechanism": "s_mech", "hypothesis": "h", "idea": None,
    }

    # Factor 0.86 + operation 1.0 + signature 0.86 + mechanism 0.81 -> DUPLICATE
    patch_with({0: 0.86, 1: 1.0, 2: 0.86, 3: 0.81})
    relation, detail = gme._classify_sibling_relation(plan, sibling)
    assert relation == "DUPLICATE", detail

    # Same as above but signature 0.84 (< 0.85) -> COMPLEMENTARY (not a duplicate)
    patch_with({0: 0.86, 1: 1.0, 2: 0.84, 3: 0.81})
    relation, _ = gme._classify_sibling_relation(plan, sibling)
    assert relation == "COMPLEMENTARY"

    # Same factor 0.86 but operation dissimilar -> COMPLEMENTARY
    patch_with({0: 0.86, 1: 0.30, 2: 0.5, 3: 0.5})
    relation, _ = gme._classify_sibling_relation(plan, sibling)
    assert relation == "COMPLEMENTARY"

    # Factor 0.84 (< 0.85) -> INDEPENDENT even with everything else identical
    patch_with({0: 0.84, 1: 1.0, 2: 0.99, 3: 0.99})
    relation, _ = gme._classify_sibling_relation(plan, sibling)
    assert relation == "INDEPENDENT"

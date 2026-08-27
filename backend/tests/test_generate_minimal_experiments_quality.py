from types import SimpleNamespace

from app.agent.steps.generate_minimal_experiments import (
    MinimalExperimentSchema,
    _validate_experiment_plan,
)


def _phenomenon():
    return SimpleNamespace(
        mechanism_under_test="Logic error detection",
        comparator="H0 no difference; H1 false-negative difference",
        oracle_experiment="Run executable hidden tests",
        kill_criterion="No meaningful difference",
        measurement="false-negative rate",
    )


def _plan(**overrides):
    values = dict(
        title="Verifier boundary experiment",
        summary="A bounded experiment for verifier errors.",
        hypothesis="Verifier false negatives differ across scenarios.",
        dataset="A constructed logic-error dataset",
        baselines="A deterministic verifier baseline",
        metrics="False-negative rate",
        model_spec="3B inference-only model",
        dataset_provenance="New synthetic dataset; construction and audit steps specified",
        oracle="Executable tests plus manual adjudication",
        statistical_analysis="Paired bootstrap confidence interval",
        resource_budget="One GPU-hour",
        scenario_atoms=["logic error", "false negative"],
        controls=["same prompt"],
        steps=["Construct cases", "Run verifier and oracle"],
        success_condition="The effect exceeds baseline variance.",
        falsification_condition="The effect is absent.",
        risks="The labels require audit.",
    )
    values.update(overrides)
    return MinimalExperimentSchema(**values)


def test_quality_gate_rejects_missing_experiment_contract_fields():
    plan = _plan(model_spec="", oracle="", statistical_analysis="")
    failures = _validate_experiment_plan(plan, phenomenon=_phenomenon(), expected_atoms=[])
    assert "MISSING_MODEL_SPEC" in failures
    assert "MISSING_ORACLE" in failures
    assert "MISSING_STATISTICAL_ANALYSIS" in failures


def test_quality_gate_rejects_model_scope_conflict_and_llm_only_oracle():
    plan = _plan(model_spec="Llama-3-8B", oracle="LLM judge only")
    failures = _validate_experiment_plan(
        plan,
        phenomenon=_phenomenon(),
        expected_atoms=["logic error"],
        gap_text="Study verifier behavior under a strict <7B model budget.",
    )
    assert "MODEL_SCOPE_CONFLICT" in failures
    assert "LLM_ONLY_ORACLE" in failures


def test_quality_gate_requires_scenario_atom_coverage():
    plan = _plan(dataset="A generic benchmark", steps=["Load data", "Run baseline"])
    failures = _validate_experiment_plan(
        plan, phenomenon=_phenomenon(), expected_atoms=["preference reversal"]
    )
    assert "SCENARIO_MISMATCH:preference reversal" in failures

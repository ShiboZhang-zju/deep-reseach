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
    assert any(f.startswith("MODEL_SCOPE_CONFLICT") for f in failures)
    assert any(f.startswith("LLM_ONLY_ORACLE") for f in failures)
    # P0-3 (task d6f64087): the failure strings must be actionable — they name
    # the violating value and the bound so the rejection-aware retry can fix
    # the exact problem instead of rewriting blind.
    scope_failure = next(f for f in failures if f.startswith("MODEL_SCOPE_CONFLICT"))
    assert "8B" in scope_failure and "7B" in scope_failure
    oracle_failure = next(f for f in failures if f.startswith("LLM_ONLY_ORACLE"))
    assert "LLM judge only" in oracle_failure


def test_quality_gate_requires_scenario_atom_coverage():
    plan = _plan(dataset="A generic benchmark", steps=["Load data", "Run baseline"])
    failures = _validate_experiment_plan(
        plan, phenomenon=_phenomenon(), expected_atoms=["preference reversal"]
    )
    assert "SCENARIO_MISMATCH:preference reversal" in failures


def test_model_scope_check_is_role_aware():
    """P1-2 (task d6f64087): the gap bounds the REWARD MODEL ("<3B reward
    models"), not the RLHF policy LLM — an oversized policy checkpoint is not
    a scope violation for the RM cap."""
    from app.agent.steps.generate_minimal_experiments import _check_model_scope

    # 8B policy model named next to a <=3B RM: the number binds to the policy
    # role, which the cap does not cover -> no failure.
    spec = ("Reward models must be <=3B parameters (e.g. Qwen2.5-3B-Instruct "
            "for the reward model); policy model Llama-3-8B-Instruct.")
    assert _check_model_scope(
        "the gap targets reward models (<3B parameters) in low-resource RLHF", spec) == []

    # The same 8B number bound to the reward model role -> violation, and the
    # message names both the offending size and the cap.
    spec_rm = "Reward model Llama-3-8B distilled or Qwen2.5-3B."
    failures = _check_model_scope(
        "the gap targets reward models (<3B parameters) in low-resource RLHF", spec_rm)
    assert len(failures) == 1
    assert failures[0].startswith("MODEL_SCOPE_CONFLICT")
    assert "8B" in failures[0] and "3B" in failures[0]

    # No role anywhere: global cap applies to every number (legacy behaviour).
    assert any(f.startswith("MODEL_SCOPE_CONFLICT") for f in _check_model_scope(
        "strict <7B model budget", "Llama-3-8B inference model"))

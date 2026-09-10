"""Regression for the LLM_ONLY_ORACLE false positive (2026-09-10, tasks
77610c15 + aaa34d46): the old detector fired on the word "LLM" naming the
EXPERIMENT SUBJECT ("Run two PPO-aligned LLMs ...") and missed the
mechanical-computation signals ("calculated analytically from the softmax
probabilities", "held-out prompts"). Two RLHF tasks each had a purely
analytic entropy oracle rejected twice, burning two tier-A ideas."""

import pytest

from app.agent.steps.generate_minimal_experiments import (
    MinimalExperimentSchema,
    _validate_experiment_plan,
)
from app.db.models import GapPhenomenonPlan

# The two real oracle texts that were falsely rejected (RLHF reward hacking).
_MECHANICAL_ENTROPY = (
    "Computed Shannon entropy of the model's output logits for each "
    "generated response. No LLM judging for entropy; entropy is calculated "
    "analytically from the softmax probabilities of the model.")
_MECHANICAL_TOKENPROB = (
    "Computational calculation of token probabilities from model logits on "
    "held-out prompts. No LLM oracle for correctness; entropy is computed "
    "directly from the model's output distribution.")
_TRUE_LLM_JUDGE = (
    "An LLM judge scores each response pair for helpfulness on a 1-9 scale; "
    "the preference labels are the ground truth.")
_NEUTRAL_NO_LLM = (
    "Unit-test pass rate on HumanEval hidden split, graded by the standard "
    "test harness.")


def _plan(oracle: str) -> MinimalExperimentSchema:
    """A minimal plan whose only variable is the oracle text."""
    return MinimalExperimentSchema(
        title="Reward hacking boundary test",
        summary="Does the model show attribution bias in preference rates",
        hypothesis="Exposure to self-generated text shifts preference rates",
        core_factor="reward signal exposure",
        core_operation="swap_label",
        core_contrast="self_vs_external",
        expected_signature="attribution bias appears in preference rates",
        mechanism_being_tested="attribution bias mechanism being measured",
        construct_identification={
            "construct": "attribution bias",
            "observed_variable": "preference rate",
            "claimed_construct": "self-preference bias",
            "operationalization": "label-swap contrast isolates the bias",
            "identification_assumptions": ["labels are independent"],
        },
        dataset="d",
        dataset_provenance="dp",
        model_spec="model m",
        baselines="fixed baseline",
        oracle=oracle,
        metrics="accuracy",
        statistical_analysis="bootstrap confidence interval",
        resource_budget="1 GPU day",
        success_condition="significance",
        falsification_condition="no difference",
        risks="r",
        steps=["step one", "step two"],
        controls=["baseline"],
        scenario_atoms=["a"],
    )


def _phenomenon() -> GapPhenomenonPlan:
    return GapPhenomenonPlan(
        mechanism_under_test="m", comparator="c",
        oracle_experiment="o", kill_criterion="k", measurement="ms",
    )


def test_mechanical_entropy_oracle_not_flagged():
    """The real RLHF oracle texts (pure logit math) must NOT be flagged."""
    failures = _validate_experiment_plan(
        _plan(_MECHANICAL_ENTROPY), phenomenon=_phenomenon())
    assert not any("LLM_ONLY_ORACLE" in f for f in failures)
    failures = _validate_experiment_plan(
        _plan(_MECHANICAL_TOKENPROB), phenomenon=_phenomenon())
    assert not any("LLM_ONLY_ORACLE" in f for f in failures)


def test_true_llm_judge_oracle_still_flagged():
    """A genuine LLM-as-judge oracle must keep being rejected."""
    failures = _validate_experiment_plan(
        _plan(_TRUE_LLM_JUDGE), phenomenon=_phenomenon())
    assert any("LLM_ONLY_ORACLE" in f for f in failures)


def test_neutral_oracle_without_llm_not_flagged():
    failures = _validate_experiment_plan(
        _plan(_NEUTRAL_NO_LLM), phenomenon=_phenomenon())
    assert not any("LLM_ONLY_ORACLE" in f for f in failures)

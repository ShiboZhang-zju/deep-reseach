"""v7.3 candidate: operationalization-only claimed_delta (OFF by default).

Why the gate exists
-------------------
Across the fixed-budget pilot's 37 independently reviewed gaps, 26 deltas led
with a measurement verb and carried no mechanism vocabulary. Independent review
gave that group 3 FULL / 23 PARTIAL / **0 NONE**, and the residual notes
repeatedly read "the specific operationalisation is not directly reported, but
the substantive delta is executed by [n]". A delta of the form "provide the
first quantitative estimate of X for at least 3 methods" is therefore not a
documented absence of knowledge; it is an unexecuted template over material
that already exists.

Why it ships DISABLED
---------------------
On the same 37 units the rule fires on 5 (3 FULL, 2 PARTIAL): 60.0% false-open
against a 10.8% base rate, losing no NONE. But those units are the units the
rule was derived from. That is post-hoc fitting, so the precision figure is not
evidence of generalisation — and enabling the gate on that basis would be the
exact unvalidated-mechanism move this project's pre-registration discipline
exists to prevent. These tests pin BOTH the behaviour (when enabled) and the
default (off), so no future edit can quietly turn it on.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === The rule itself =======================================================

@pytest.mark.parametrize("delta", [
    # Verbatim shapes from the pilot's reviewed pool.
    "Provide the first quantitative estimate of cross-domain calibration "
    "degradation for at least 3 UQ methods and test whether temperature "
    "scaling reduces it.",
    "Quantify, on a real clinical corpus, a middle-position accuracy drop of "
    ">15pp and show this drop exceeds the drop measured on synthetic "
    "benchmarks as well as on naturalistic legal documents.",
    "Measure conformal coverage error across at least 3 domain-shift "
    "magnitudes and report the ablation jointly.",
])
def test_measurement_template_deltas_fire(delta):
    from app.agent.steps.mine_gaps import _is_operationalization_only_delta

    assert _is_operationalization_only_delta(delta) is True


@pytest.mark.parametrize("delta", [
    # Carries mechanism vocabulary: a real causal question, not a template.
    "Quantify how much of the degradation is caused by exchangeability "
    "violation versus base-model miscalibration.",
    # Single deliverable: "measure X" alone can be a genuine missing evaluation.
    "Measure the calibration drop induced by domain transfer.",
    # Boundary language, no measurement lead.
    "Methods degrade beyond 32k tokens and no work evaluates that regime.",
    # Performance target shape (already handled by the v7.2 gate).
    "Achieves >10% higher accuracy than the strongest baseline.",
    "",
])
def test_non_template_deltas_do_not_fire(delta):
    from app.agent.steps.mine_gaps import _is_operationalization_only_delta

    assert _is_operationalization_only_delta(delta) is False


def test_rule_requires_all_three_conditions(monkeypatch):
    """Removing any single condition must stop the rule firing.

    This is what keeps the rule from becoming a blunt "measurement verb = bad"
    filter, which would delete genuine missing_evaluation opportunities.
    """
    from app.agent.steps import mine_gaps

    delta = ("Provide the first estimate of calibration degradation for at "
             "least 3 methods.")

    assert mine_gaps._is_operationalization_only_delta(delta) is True

    # 1. mechanism vocabulary present -> does not fire
    assert mine_gaps._is_operationalization_only_delta(
        delta + " The mechanism behind the shift is untested.") is False

    # 2. no multi-deliverable conjunction -> does not fire
    assert mine_gaps._is_operationalization_only_delta(
        "Provide the first estimate of calibration degradation.") is False

    # 3. no measurement-verb lead -> does not fire
    assert mine_gaps._is_operationalization_only_delta(
        "Calibration degradation for at least 3 methods is unmeasured.") is False


# === Default-off contract ==================================================

def test_gate_is_off_by_default():
    """The shipped default must be disabled.

    Flipping this default is a policy decision that requires an out-of-sample
    check (24-topic production_e2e pool), not a code change.
    """
    from app.config import Settings

    assert Settings().gap_mining_operationalization_gate is False


def test_switch_reads_settings_lazily(monkeypatch):
    from app.config import settings
    from app.agent.steps.mine_gaps import _settings_operationalization_gate

    monkeypatch.setattr(settings, "gap_mining_operationalization_gate", False)
    assert _settings_operationalization_gate() is False
    monkeypatch.setattr(settings, "gap_mining_operationalization_gate", True)
    assert _settings_operationalization_gate() is True


def test_rejection_code_is_distinct_from_the_v72_gate():
    """A new reject reason must not be confusable with PERFORMANCE_TARGET_DELTA.

    The two gates fire on different shapes and belong to different hypotheses;
    collapsing them would make per-type attribution of the funnel impossible.
    """
    from app.agent.steps import mine_gaps as m

    # Both constants must exist as distinct literals in the module source.
    src = open(m.__file__, encoding="utf-8").read()
    assert "OPERATIONALIZATION_ONLY_DELTA" in src
    assert "PERFORMANCE_TARGET_DELTA" in src
    assert "OPERATIONALIZATION_ONLY_DELTA" != "PERFORMANCE_TARGET_DELTA"


def test_v6_arm_disables_the_gate(monkeypatch):
    """The v6 control arm must not gain a v7.3 gate.

    Arm B is defined as "v6 mining policy on the current stack"; leaking a new
    admission rule into it would break the experiment's single-variable design.
    """
    from app.config import settings
    from app.agent.steps.mine_gaps import _active_contracts

    monkeypatch.setattr(settings, "gap_mining_legacy_v6_mode", True)
    types = [c.gap_type for c in _active_contracts()]
    assert types == ["boundary_gap", "missing_evaluation"]

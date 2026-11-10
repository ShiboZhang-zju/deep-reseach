"""Gap-supply fix: the SHAPE of `claimed_delta` caps what review can conclude.

Diagnosis this pins (neighbour-budget experiment, 2026-11-06)
-------------------------------------------------------------
Independent blind review of 28 audited gaps returned:

    NONE = 0    PARTIAL = 26    FULL = 2

so the primary metric T = NONE/(NONE+PARTIAL+FULL) was 0.0% in BOTH arms. The
3-vs-8 contrast therefore measured nothing (`RESULTS.md` §2.1) — this is the
same defect as the v7.2 TIE: an instrument with no headroom.

Why more neighbours cannot fix it
---------------------------------
Reading the 26 PARTIAL verdicts, the dominant reason was NOT "retrieved too
little" but the shape of the claim:
"substantially covers the individual ingredients of the claimed delta but not
the full combination" — 12/26 = 46%. And 49% of the 116 mined deltas enumerate
>=3 conjoined elements.

Structurally: every ingredient is published somewhere, so the honest answer to
"has anyone done this?" is always "the parts, yes; all together, no" =
PARTIAL. A conjunction cannot be NONE by construction, and no retrieval budget
changes that.

What shipped
------------
1. A structural constraint in `_GAP_MINING_SYSTEM` that forbids two shapes
   (configuration-only, conjunction-only) and requires ONE mechanism
   assertion. This is the actual fix: it changes what the model is asked to
   produce instead of filtering afterwards.
2. `_is_conjunction_only_delta` + `gap_mining_conjunction_gate`, OFF by
   default. The detector is precise (6/6 fired units are genuinely
   conjunction-shaped) but LOW RECALL (5.2% of 116), and it was derived from
   the same 28 units it would be scored on — enabling it would repeat the
   v7.3 mistake documented above the operationalization gate.

These tests assert the diagnosis, the prompt contract, and the gate's default.
They do NOT assert that the new prompt improves NONE rates: that requires an
out-of-sample run, and claiming otherwise is exactly what this project's
pre-registration discipline forbids.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === The diagnosis must stay visible in the code ==========================

def test_documented_diagnosis_matches_the_measurement():
    """Pin the numbers so a later "improvement" cannot quietly rewrite history."""
    n_none, n_partial, n_full = 0, 26, 2
    total = n_none + n_partial + n_full
    assert total == 28
    # The primary metric that had no headroom.
    assert n_none / total == 0.0
    # Share of mined deltas enumerating >=3 conjoined elements.
    assert round(57 / 116 * 100) == 49


def test_the_46_percent_claim_is_still_marked_as_retracted():
    """The 46% figure was wrong; the correction must not be silently dropped.

    An earlier revision used a loose regex (any "both"/"together"/"combines")
    and non-exclusive categories to claim the ingredients-not-combination
    argument drove 46% of PARTIALs. Strict matching gives 8%. The real top
    reasons are "partially covers the claim" and "adjacent work, different
    setting" (the rubric's catch-all), each ~50%.
    """
    strict, loose_claim = 2 / 26, 12 / 26
    assert round(strict * 100) == 8
    assert round(loose_claim * 100) == 46
    assert strict < loose_claim          # the retraction is not cosmetic

    # The retraction must be visible where the number was first stated.
    here = os.path.dirname(__file__)
    cfg = os.path.join(here, "..", "app", "config.py")
    with open(cfg, encoding="utf-8") as fh:
        src = fh.read()
    assert "CORRECTION (2026-11-09)" in src


def test_out_of_sample_result_is_recorded_as_null():
    """The prompt fix changed the SHAPE but not the NONE rate.

    Measured on 24 deltas from two out-of-sample topics: numbers 74%->0%,
    configuration phrasing 59%->0%, yet blind review returned 6/6 PARTIAL.
    Keeping this assertion prevents anyone citing the prompt change as a fix.
    """
    old_numbers, new_numbers = 86 / 116, 0 / 24
    old_geared, new_geared = 69 / 116, 0 / 24
    assert round(old_numbers * 100) == 74 and new_numbers == 0.0
    assert round(old_geared * 100) == 59 and new_geared == 0.0

    validation = {"NONE": 0, "PARTIAL": 6, "FULL": 0}
    n = sum(validation.values())
    assert validation["NONE"] / n == 0.0


def test_conjunction_cannot_be_none_by_construction():
    """A pure conjunction is answerable only as PARTIAL — that is the defect.

    This is a reasoning assertion, not a code assertion: if every ingredient is
    published, then for ANY retrieval budget the reviewer's honest answer is
    "parts covered, combination not", i.e. PARTIAL. It is recorded as a test so
    nobody re-runs the 3-vs-8 contrast expecting a different result.
    """
    ingredients_published = True
    combination_published = False
    assert ingredients_published and not combination_published
    verdict = "PARTIAL"          # by the rubric's own definition
    assert verdict != "NONE"


# === Prompt contract ======================================================

def test_mining_prompt_forbids_both_bad_shapes():
    from app.agent.steps.mine_gaps import _GAP_MINING_SYSTEM

    assert "CONFIGURATION-ONLY" in _GAP_MINING_SYSTEM
    assert "CONJUNCTION-ONLY" in _GAP_MINING_SYSTEM
    assert "CAPABILITY, NOT AN" in _GAP_MINING_SYSTEM


def test_mining_prompt_requires_the_single_assertion_shape():
    from app.agent.steps.mine_gaps import _GAP_MINING_SYSTEM

    assert "ONE assertion about how something works" in _GAP_MINING_SYSTEM
    assert "mechanism" in _GAP_MINING_SYSTEM.lower()


def test_mining_prompt_keeps_the_self_check_that_drops_bad_candidates():
    from app.agent.steps.mine_gaps import _GAP_MINING_SYSTEM

    assert "SELF-CHECK" in _GAP_MINING_SYSTEM
    # The conjunction test: drop any one clause and the claim must die.
    assert "delete any one of its clauses" in _GAP_MINING_SYSTEM
    assert "drop the candidate" in _GAP_MINING_SYSTEM


def test_mining_prompt_still_carries_the_older_constraints():
    """The new text must not have replaced the earlier, still-valid rules."""
    from app.agent.steps.mine_gaps import _GAP_MINING_SYSTEM

    assert "cite only supplied question IDs" in _GAP_MINING_SYSTEM
    assert "NEVER emit placeholder operands" in _GAP_MINING_SYSTEM
    assert "write every field in English" in _GAP_MINING_SYSTEM


def test_mining_prompt_formats_without_stray_braces():
    """`_GAP_MINING_SYSTEM` goes through str.format(): literal braces crash mining.

    This is not hypothetical — the first version of the conjunction constraint
    used "(i) and (ii)" style placeholders written as "{i}" / "{ii}", which made
    `.format()` raise KeyError('i') and broke every lens call. A prompt edit is
    a code edit; it must survive the same call the runtime makes.
    """
    from app.agent.steps.mine_gaps import _GAP_MINING_SYSTEM

    rendered = _GAP_MINING_SYSTEM.format(
        gap_type="boundary_gap",
        definition="d",
        discovery_directive="x",
        citation_rule="c",
    )
    assert "boundary_gap" in rendered
    # Nothing from the constraint text may vanish as an unknown field.
    assert "CONJUNCTION-ONLY" in rendered


# === Detector behaviour ===================================================

def test_detector_fires_on_combination_vocabulary():
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    assert _is_conjunction_only_delta(
        "The first work to combine spectral pruning with token-level "
        "compression and evaluate them jointly.")


def test_detector_fires_on_enumerated_deliverables():
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    assert _is_conjunction_only_delta(
        "Requires (i) holding ratio fixed, (ii) varying density and "
        "(iii) measuring faithfulness.")


def test_detector_fires_on_three_clause_conjunction():
    """Three conjoined deliverables in one sentence."""
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    assert _is_conjunction_only_delta(
        "Report ECE and NLL on 2 architecture families and on 2 tasks and "
        "under 3 graded OOD-shift levels and across 2 domains.")


def test_detector_does_not_fire_on_two_clause_delta():
    """Two clauses is normal prose, not an arrangement; the detector must not fire.

    This pins the LOW-RECALL reality documented in config.py: the filter is
    precise (everything it fires on really is conjunction-shaped) but it misses
    most conjunctions, which is exactly why the real fix is the mining prompt
    and the gate ships disabled.
    """
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    assert not _is_conjunction_only_delta(
        "Report ECE on 2 architecture families and on 2 tasks under 3 graded "
        "OOD-shift levels.")


def test_detector_spares_a_single_mechanism_claim():
    """The shape we WANT must survive: one causal assertion, no enumeration."""
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    assert not _is_conjunction_only_delta(
        "Load-balancing pressure, not selection frequency, is what makes "
        "expert-specialization verdicts unreliable, because the balance "
        "gradient acts on router parameters rather than expert representations.")


def test_detector_is_documented_as_low_recall():
    """Guard against someone enabling it while believing it is a strong filter."""
    from app.agent.steps.mine_gaps import _is_conjunction_only_delta

    samples = [
        "Load-balancing pressure, not selection frequency, is what makes "
        "specialization verdicts unreliable.",
        "Compressed tokens stop being traceable to source spans beyond a "
        "boundary that current evaluations never measure.",
    ]
    fired = sum(1 for s in samples if _is_conjunction_only_delta(s))
    assert fired == 0, (
        "these are mechanism claims; a detector that fires on them is not the "
        "precise-but-low-recall filter documented in config.py")


# === Gate default =========================================================

def test_gate_is_off_by_default():
    """Ships disabled: the rule was fitted on the same 28 units."""
    from app.config import settings

    assert settings.gap_mining_conjunction_gate is False


def test_gate_reader_reflects_the_setting(monkeypatch):
    from app.config import settings
    from app.agent.steps.mine_gaps import _settings_conjunction_gate

    monkeypatch.setattr(settings, "gap_mining_conjunction_gate", False)
    assert _settings_conjunction_gate() is False
    monkeypatch.setattr(settings, "gap_mining_conjunction_gate", True)
    assert _settings_conjunction_gate() is True


def test_gate_reason_code_is_distinct_from_the_older_gates():
    """A rejection must say WHICH shape it rejected."""
    from app.agent.steps import mine_gaps
    import inspect

    src = inspect.getsource(mine_gaps)
    assert "CONJUNCTION_ONLY_DELTA" in src
    assert "OPERATIONALIZATION_ONLY_DELTA" in src
    assert "PERFORMANCE_TARGET_DELTA" in src

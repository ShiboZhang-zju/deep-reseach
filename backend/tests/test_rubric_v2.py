"""Rubric v2: the instrument defect, the fix, and the gates that guard it.

The problem being pinned
------------------------
v1's PARTIAL was:

    "prior work covers PART of the delta, covers it in a different setting, or
     the delta is a re-parameterisation of known work"

The middle clause is satisfied by any paper on the same topic in any setting, and
every real topic has hundreds of those. So NONE required a literature vacuum and
PARTIAL absorbed everything. Measured: 34 blind-adjudicated gaps gave
NONE=0 / PARTIAL=32 / FULL=2, making T = NONE/(...) structurally 0.

A calibration probe settled it: "does LLM inference depend on lunar gravity"
(no adjacent work at all) came back NONE; a realistic claim came back PARTIAL.

These tests pin (a) the diagnosis, (b) v2's structure, (c) the gates that stop
the fix from degenerating into NONE-everywhere, and (d) the uncomfortable fact
that my own earlier human spot-check was anchored by v1 and is therefore not
independent evidence.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "..", "eval_results", "neighbor_budget")


def _v2mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rubric_v2", os.path.join(_HERE, "..", "scripts", "adjudicate_rubric_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(name):
    p = os.path.join(_RESULTS, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# === diagnosis ============================================================

def test_v1_partial_clause_is_the_defect():
    """The 'different setting' clause is what made PARTIAL a catch-all."""
    sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
    from adjudicate_pilot import RUBRIC
    flat = " ".join(RUBRIC.split())
    assert "covers it in a different setting" in flat


def test_v1_distribution_had_no_headroom():
    v1 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    v1.update({json.loads(l)["anon_id"]: json.loads(l)
               for l in open(os.path.join(_RESULTS, "validation_verdicts.jsonl"),
                             encoding="utf-8") if l.strip()})
    gaps = [r for r in v1.values() if r["unit_type"] == "gap"]
    none = sum(1 for r in gaps if r["verdict"] == "NONE")
    assert none == 0, "v1 produced zero NONE across every adjudicated gap"
    assert len(gaps) == 34


# === v2 structure =========================================================

def test_v2_rubric_separates_adjacency_from_coverage():
    r = _v2mod().RUBRIC_V2
    assert "ADJACENCY" in r
    assert "(A) ADJACENCY" in r and "(B) PARTIAL COVERAGE" in r
    # the exact clause that caused the v1 defect must be gone
    assert "covers it in a different setting" not in r


def test_v2_rubric_forbids_both_degenerate_defaults():
    r = _v2mod().RUBRIC_V2
    assert "Do NOT default to NONE" in r
    assert "Do NOT default to PARTIAL" in r


def test_v2_requires_the_delta_to_be_restated_first():
    m = _v2mod()
    fields = m.GapVerdictV2.model_fields
    assert "delta_stated" in fields
    assert "delta_established" in fields
    assert "adjacent_work_exists" in fields
    assert "before judging" in m.RUBRIC_V2


def test_v2_schema_is_rebuildable_under_importlib():
    """Pydantic forward refs break when a module is exec'd, not imported.

    This is how the evaluator loads it; without model_rebuild the first call
    raises 'GapVerdictV2 is not fully defined'.
    """
    from typing import Literal
    m = _v2mod()
    for model in (m.GapVerdictV2, m.IdeaVerdictV2):
        model.model_rebuild(_types_namespace={"Literal": Literal})
    assert "verdict" in m.GapVerdictV2.model_fields


# === gates ================================================================

def _probe_results():
    key = json.load(open(os.path.join(_RESULTS, "rubric_v2_probes_key.json"),
                         encoding="utf-8"))["probes"]
    got = {json.loads(l)["anon_id"]: json.loads(l)
           for l in open(os.path.join(_RESULTS, "probe_verdicts_v2.jsonl"),
                         encoding="utf-8") if l.strip()}
    return key, got


def test_gate_g1_forward_probe_reaches_none():
    """Without this, v2 is just v1 with extra fields."""
    key, got = _probe_results()
    fwd = [p for p in key if p.startswith("p2v-N")]
    reached = sum(1 for p in fwd if got.get(p, {}).get("verdict") == "NONE")
    assert reached >= 1, "v2 still cannot express adjacency-without-coverage"


def test_gate_g2_reverse_probes_still_reach_full():
    """Without this, v2 has degenerated into NONE-everywhere."""
    key, got = _probe_results()
    rev = [p for p in key if p.startswith("p2v-R")]
    full = sum(1 for p in rev if got.get(p, {}).get("verdict") == "FULL")
    none = sum(1 for p in rev if got.get(p, {}).get("verdict") == "NONE")
    assert full >= 3 and none == 0


def test_gate_g3_verdict_matches_delta_established():
    v2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    m = {"NONE": "no", "PARTIAL": "partly", "FULL": "yes"}
    ok = sum(1 for r in v2.values()
             if m.get(r["verdict"]) == r.get("delta_established"))
    assert ok == len(v2), "rubric internally inconsistent"


# === the result ===========================================================

def test_v2_gives_the_metric_headroom():
    v2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    gaps = [r for r in v2.values() if r["unit_type"] == "gap"]
    n = len(gaps)
    none = sum(1 for r in gaps if r["verdict"] == "NONE")
    assert n == 34
    assert none == 7
    assert abs(none / n * 100 - 20.6) < 0.1


def test_v2_only_narrowed_never_loosened():
    """No NONE->PARTIAL: the change tightened the catch-all, not the reverse."""
    v1 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    v1.update({json.loads(l)["anon_id"]: json.loads(l)
               for l in open(os.path.join(_RESULTS, "validation_verdicts.jsonl"),
                             encoding="utf-8") if l.strip()})
    v2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    common = set(v1) & set(v2)
    loosened = [a for a in common
                if v1[a]["verdict"] == "NONE" and v2[a]["verdict"] != "NONE"]
    assert not loosened


def test_adjacency_present_with_delta_unestablished_is_now_expressible():
    v2 = [json.loads(l) for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                                      encoding="utf-8") if l.strip()]
    both = [r for r in v2
            if r["verdict"] == "NONE" and r.get("adjacent_work_exists")]
    assert len(both) == 7, "this cell is exactly what v1 could not output"


# === the uncomfortable one ================================================

def test_v21_forbids_credit_for_weakening_an_absence_claim():
    """v2 still graded PARTIAL on adjacency + a weakened absence claim.

    On 7 of 11 disagreements with my blind judgement, v2 cited NO direct test of
    the delta yet returned PARTIAL, e.g. nb-065 whose own notes say "Search
    results do NOT contain a study that measures this correlation coefficient"
    and then answer PARTIAL anyway. v2.1 closes that route.
    """
    r = _v2mod().RUBRIC_V2
    flat = " ".join(r.split())
    assert "NO CREDIT FOR WEAKENING AN ABSENCE CLAIM" in flat
    assert "still adjacency" in flat
    assert "that is NONE, not PARTIAL" in flat


def test_v21_still_keeps_both_anti_default_warnings():
    r = _v2mod().RUBRIC_V2
    assert "Do NOT default to NONE" in r
    assert "Do NOT default to PARTIAL" in r


def test_v21_probe_gates_improved_or_held():
    """G1 must improve, G2 must not regress."""
    key, got = _probe_results()
    fwd = [p for p in key if p.startswith("p2v-N")]
    rev = [p for p in key if p.startswith("p2v-R")]
    none_fwd = sum(1 for p in fwd if got.get(p, {}).get("verdict") == "NONE")
    full_rev = sum(1 for p in rev if got.get(p, {}).get("verdict") == "FULL")
    none_rev = sum(1 for p in rev if got.get(p, {}).get("verdict") == "NONE")
    assert none_fwd == 2, "v2.1 should reach NONE on both forward probes"
    assert full_rev == 3 and none_rev == 0


def test_v21_only_tightened_never_loosened():
    v2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}
    v21 = {json.loads(l)["anon_id"]: json.loads(l)
           for l in open(os.path.join(_RESULTS, "rev21_verdicts.jsonl"),
                         encoding="utf-8") if l.strip()}
    common = set(v2) & set(v21)
    loosened = [a for a in common
                if v2[a]["verdict"] == "NONE" and v21[a]["verdict"] != "NONE"]
    assert not loosened
    tightened = [a for a in common
                 if v2[a]["verdict"] == "PARTIAL" and v21[a]["verdict"] == "NONE"]
    assert len(tightened) == 12


def test_v21_headroom_is_recorded_but_NOT_as_a_quality_conclusion():
    """T rose to 55.9%. That proves headroom, not that half the gaps are real."""
    v21 = [json.loads(l) for l in open(os.path.join(_RESULTS, "rev21_verdicts.jsonl"),
                                       encoding="utf-8") if l.strip()]
    n = len(v21)
    none = sum(1 for r in v21 if r["verdict"] == "NONE")
    assert n == 34 and none == 19
    assert abs(none / n * 100 - 55.9) < 0.1

    with open(os.path.join(_RESULTS, "RUBRIC_V21_FINAL.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "不能作为质量结论" in text


def test_decision_to_stop_tuning_is_recorded():
    """Stopping is a deliberate choice, not an oversight.

    My own blind judgement is wrong on 4 of 11 disagreements (v2's retrieval was
    better). Tuning the rubric toward that baseline would move the anchor from
    v1 to me.
    """
    with open(os.path.join(_RESULTS, "RUBRIC_V21_FINAL.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "我为什么停在这里" in text
    assert "过拟合" in text
    assert "val-005" in text, "the known over-correction case must stay visible"


def test_my_human_spotcheck_was_anchored_by_v1_and_is_not_independent():
    """Recorded so nobody cites the old 21/21 as proof of reliability.

    Against v1 my hand judgements agreed 21/21. Against v2 (same units, better
    instrument) only 3/5. The earlier agreement was not evidence that v1 was
    right — both my judgement and v1 shared the rule "adjacent work exists ->
    PARTIAL". This test keeps that correction visible.
    """
    human = json.load(open(os.path.join(_RESULTS, "spotcheck_judgements.json"),
                           encoding="utf-8"))["judgements"]
    v2 = {json.loads(l)["anon_id"]: json.loads(l)
          for l in open(os.path.join(_RESULTS, "rev2_verdicts.jsonl"),
                        encoding="utf-8") if l.strip()}

    overlapping = [a for a in human if a in v2]
    assert len(overlapping) == 5
    agree = sum(1 for a in overlapping if human[a]["verdict"] == v2[a]["verdict"])
    assert agree == 3, (
        "v2 disagrees with two of my hand judgements; those disagreements are "
        "the evidence that the old 21/21 was anchored, not correct")

    # the two disagreements are specifically the PARTIAL->NONE flips
    disagreed = [a for a in overlapping if human[a]["verdict"] != v2[a]["verdict"]]
    assert set(disagreed) == {"nb-015", "nb-099"}
    for a in disagreed:
        assert human[a]["verdict"] == "PARTIAL"
        assert v2[a]["verdict"] == "NONE"

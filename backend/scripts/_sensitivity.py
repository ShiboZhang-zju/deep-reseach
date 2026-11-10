"""Sensitivity analysis: is the -4pp arm difference robust or a single-unit artefact?

The primary metric T = NONE/(NONE+PARTIAL+FULL) is near-degenerate here:
35 of 37 gaps are PARTIAL, so T is 0% vs 4% and the whole arm difference rests
on ONE unit (arm B's single NONE). This script quantifies that fragility and
tests the alternative reading in which a PARTIAL carries partial credit.

Also tests the "conjunction bias" hypothesis: if the adjudicator calls a claim
PARTIAL merely because prior work covers the components but not their exact
conjunction, then PARTIAL should be read closer to FULL, not to NONE.
"""
import json
from collections import Counter

V = r"eval_results\fixed_budget_pilot\verdicts.jsonl"
M = r"eval_results\fixed_budget_pilot\review_mapping.json"
rows = [json.loads(l) for l in open(V, encoding="utf-8") if l.strip()]
mapping = json.load(open(M, encoding="utf-8"))["units"]
gaps = [{**r, **{"arm": mapping[r["anon_id"]]["arm"],
                 "topic": mapping[r["anon_id"]]["topic_id"]}}
        for r in rows if r["unit_type"] == "gap"]

print("=== 1. leave-one-out robustness of the arm difference ===")
def rate(sub, credit):
    s = sum(credit.get(r["verdict"], 0.0) for r in sub)
    return s / len(sub) if sub else None

schemes = {
    "T (preregistered, NONE only)": {"NONE": 1.0, "PARTIAL": 0.0, "FULL": 0.0},
    "T' (partial credit 0.5)": {"NONE": 1.0, "PARTIAL": 0.5, "FULL": 0.0},
    "T'' (partial credit 0.75)": {"NONE": 1.0, "PARTIAL": 0.75, "FULL": 0.0},
}
for name, credit in schemes.items():
    a = [r for r in gaps if r["arm"] == "A"]
    b = [r for r in gaps if r["arm"] == "B"]
    ra, rb = rate(a, credit), rate(b, credit)
    print(f"  {name:<32} A={ra:.1%} B={rb:.1%} delta={(ra-rb)*100:+.1f}pp")

print("\n  leave-one-out on arm B (drop one unit, recompute preregistered delta):")
b = [r for r in gaps if r["arm"] == "B"]
a = [r for r in gaps if r["arm"] == "A"]
base_a = rate(a, schemes["T (preregistered, NONE only)"])
deltas = []
for i in range(len(b)):
    sub = b[:i] + b[i + 1:]
    deltas.append((rate(sub, schemes["T (preregistered, NONE only)"]) - base_a) * 100)
deltas.sort()
print(f"    base delta = {(rate(b, schemes['T (preregistered, NONE only)']) - base_a)*100:+.1f}pp")
print(f"    after dropping one B unit: min={deltas[0]:+.1f}pp  max={deltas[-1]:+.1f}pp")
n_one_none = sum(1 for r in b if r["verdict"] == "NONE")
print(f"    arm B NONE count = {n_one_none}: the entire positive B rate is "
      f"{'that one unit' if n_one_none == 1 else 'distributed'}")

print("\n=== 2. does dropping the single NONE erase the NO-GO? ===")
print("  (i.e. how much of the decision is one reviewer call)")
b_wo = [r for r in b if r["verdict"] != "NONE"]
r_b_wo = rate(b_wo, schemes["T (preregistered, NONE only)"])
print(f"    arm B without its only NONE: T_B={r_b_wo:.1%} -> delta="
      f"{(base_a - r_b_wo)*100:+.1f}pp (A tie/no-gap)")

print("\n=== 3. both arms are dominated by PARTIAL: variance check ===")
for arm in ("A", "B"):
    sub = [r for r in gaps if r["arm"] == arm]
    c = Counter(r["verdict"] for r in sub)
    ent = 0.0
    for k, v in c.items():
        p = v / len(sub)
        ent -= p * __import__("math").log2(p)
    print(f"  arm {arm}: n={len(sub)} dist={dict(c)} entropy={ent:.2f} bits "
          f"({'degenerate' if ent < 0.5 else 'has variance'})")

print("\n=== 4. cause hypothesis: conjunction-shaped claims ===")
# A claim whose delta lists several required elements invites PARTIAL whenever
# prior work covers some but not all of them.
import re
conj = re.compile(r"\b(and|while|jointly|combination of|all three|both)\b", re.I)
for arm in ("A", "B"):
    sub = [r for r in gaps if r["arm"] == arm]
    with_conj = sum(1 for r in sub
                    if conj.search((r.get("residual_novelty") or "") + " "
                                   + (r.get("notes") or "")))
    print(f"  arm {arm}: residual/notes mention conjunction language in "
          f"{with_conj}/{len(sub)}")

print("\n=== 5. FULL / NONE cases in full ===")
for r in gaps:
    if r["verdict"] in {"FULL", "NONE"}:
        print(f"\n  {r['anon_id']} arm={r['arm']} verdict={r['verdict']} "
              f"conf={r['confidence']} keys={r.get('key_prior_work')}")
        print(f"    notes: {(r.get('notes') or '')[:400]}")

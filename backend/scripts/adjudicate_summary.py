"""Apply the FROZEN §7 decision rule to the pilot's blind verdicts.

Reads verdicts.jsonl + the sealed review_mapping.json (this is the unsealing
step — run it only after every verdict is written) and computes:

  - per-arm gap true-rate  T = NONE / (NONE + PARTIAL + FULL)
  - paired difference by topic (the design is paired, so topic-level pairs are
    reported alongside the pooled rate)
  - the §7 branch: GO / NO-GO / TIE, using the thresholds frozen in
    EXPERIMENT_DESIGN.md BEFORE any run started (>= +10pp and executable not
    worse; T_A < T_B -> NO-GO; |diff| < 10pp -> tie-break by secondary
    metrics, defaulting to NO-GO when v7.2 wins on nothing).
  - the §8.2 human spot-check scaffold: a stratified sample proposal whose
    agreement with the LLM adjudicator decides whether the LLM verdicts stand.

IMPORTANT: the §7 rule is applied verbatim. This script must not be edited to
"make the result nicer" — if the rule gives NO-GO, that is the result.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

OUT_DIR = r"eval_results\fixed_budget_pilot"
VERDICTS = os.path.join(OUT_DIR, "verdicts.jsonl")
MAPPING = os.path.join(OUT_DIR, "review_mapping.json")
SECONDARY = os.path.join(OUT_DIR, "RESULTS_SECONDARY.md")

GO_DELTA_PP = 10.0        # §7.1
TIE_BAND_PP = 10.0        # §7.3
SPOTCHECK_FRACTION = 0.20  # §8.2
SPOTCHECK_SEED = 20261102


def _true_rate(verdicts):
    counted = [v for v in verdicts if v in {"NONE", "PARTIAL", "FULL"}]
    if not counted:
        return None, Counter()
    c = Counter(counted)
    return c["NONE"] / len(counted), c


def main() -> None:
    if not os.path.exists(VERDICTS):
        sys.exit(f"no verdicts at {VERDICTS}")

    mapping = json.load(open(MAPPING, encoding="utf-8"))["units"]
    rows = [json.loads(l) for l in open(VERDICTS, encoding="utf-8") if l.strip()]

    units = []
    for r in rows:
        meta = mapping.get(r["anon_id"])
        if not meta:
            print(f"  !! {r['anon_id']} not in mapping — skipped")
            continue
        units.append({**r, "arm": meta["arm"], "topic": meta["topic_id"]})

    gaps = [u for u in units if u["unit_type"] == "gap"]
    ideas = [u for u in units if u["unit_type"] == "idea"]

    print(f"unsealed: {len(units)} verdicts "
          f"({len(gaps)} gap, {len(ideas)} idea)\n")

    # ---------------- primary metric ----------------
    print("=== PRIMARY METRIC: gap true-rate  T = NONE / (NONE+PARTIAL+FULL) ===")
    per_arm = {}
    for arm in ("A", "B"):
        sub = [u for u in gaps if u["arm"] == arm]
        rate, dist = _true_rate([u["verdict"] for u in sub])
        per_arm[arm] = rate
        print(f"\n  arm {arm} (n={len(sub)}): T = "
              + ("n/a" if rate is None else f"{rate:.1%}"))
        for k in ("NONE", "PARTIAL", "FULL"):
            print(f"    {k:<8} {dist.get(k, 0)}")

    print("\n  paired by topic (arm A vs arm B):")
    by_topic = defaultdict(lambda: {"A": [], "B": []})
    for u in gaps:
        by_topic[u["topic"]][u["arm"]].append(u["verdict"])
    print(f"    {'topic':<10}{'A n':>5}{'A T':>8}{'B n':>5}{'B T':>8}{'delta':>9}")
    deltas = []
    for topic in sorted(by_topic):
        ra, _ = _true_rate(by_topic[topic]["A"])
        rb, _ = _true_rate(by_topic[topic]["B"])
        d = "" if (ra is None or rb is None) else f"{(ra - rb) * 100:+.1f}pp"
        if ra is not None and rb is not None:
            deltas.append(ra - rb)
        print(f"    {topic:<10}{len(by_topic[topic]['A']):>5}"
              f"{'n/a' if ra is None else f'{ra:.0%}':>8}"
              f"{len(by_topic[topic]['B']):>5}"
              f"{'n/a' if rb is None else f'{rb:.0%}':>8}{d:>9}")
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        print(f"\n    mean paired delta: {mean_delta * 100:+.1f}pp "
              f"(A better in {wins}/{len(deltas)} topics)")

    # ---------------- §7 branch ----------------
    print("\n=== §7 FROZEN DECISION RULE ===")
    ta, tb = per_arm["A"], per_arm["B"]
    if ta is None or tb is None:
        print("  UNDECIDED: one arm has no gap verdicts.")
        return
    diff_pp = (ta - tb) * 100
    exec_a = sum(1 for u in ideas if u["arm"] == "A"
                 and u["verdict"] == "fund")
    exec_b = sum(1 for u in ideas if u["arm"] == "B"
                 and u["verdict"] == "fund")
    print(f"  T_A = {ta:.1%}   T_B = {tb:.1%}   delta = {diff_pp:+.1f}pp")
    print(f"  adjudicator-funded ideas: A={exec_a}  B={exec_b}")

    if diff_pp >= GO_DELTA_PP and exec_a >= exec_b:
        print("  -> GO (§7.1): widen to the full experiment incl. the "
              "V2-no-audit ablation.")
    elif diff_pp < 0:
        print("  -> NO-GO (§7.2): T_A < T_B. Next step is per-gap-type "
              "attribution, not a wholesale rollback.")
    elif abs(diff_pp) < TIE_BAND_PP:
        print("  -> TIE (§7.3): within the ±10pp band. Tie-break on secondary "
              "metrics; if v7.2 wins on nothing, treat as NO-GO.")
    else:
        print("  -> BETWEEN BANDS: delta is >= 0 but < +10pp — §7 defines no "
              "GO there, so this is a NO-GO with a positive trend.")

    # ---------------- idea layer ----------------
    print("\n=== IDEA LAYER (secondary) ===")
    for arm in ("A", "B"):
        c = Counter(u["verdict"] for u in ideas if u["arm"] == arm)
        n = sum(c.values())
        if not n:
            continue
        print(f"  arm {arm} (n={n}): fund={c.get('fund',0)} "
              f"maybe={c.get('maybe',0)} no={c.get('no',0)}")

    print("\n=== pipeline verdict vs adjudicator (gap layer) ===")
    # The pipeline's audit verdict is on the input record; joining it here is
    # what makes "the pipeline said confirmed but the reviewer said FULL"
    # countable — the calibration question the pilot itself could not answer.
    inputs = {json.loads(l)["anon_id"]: json.loads(l)
              for l in open(os.path.join(OUT_DIR, "adjudication_input.jsonl"),
                            encoding="utf-8") if l.strip()}
    cross = Counter()
    for u in gaps:
        audit = (inputs.get(u["anon_id"]) or {}).get("audit") or {}
        cross[(audit.get("audit_result"), u["verdict"])] += 1
    for (audit_result, verdict), n in sorted(cross.items(),
                                             key=lambda kv: -kv[1]):
        print(f"  pipeline={str(audit_result):<10} reviewer={str(verdict):<8} {n}")

    # ---------------- human spot-check scaffold ----------------
    print("\n=== §8.2 HUMAN SPOT-CHECK SAMPLE ===")
    need = int(round(len(units) * SPOTCHECK_FRACTION))
    strat = defaultdict(list)
    for u in units:
        strat[(u["unit_type"], u["arm"])].append(u["anon_id"])
    rng = random.Random(SPOTCHECK_SEED)
    sample = []
    keys = sorted(strat)
    i = 0
    while len(sample) < need and any(strat[k] for k in keys):
        k = keys[i % len(keys)]
        if strat[k]:
            sample.append(strat[k].pop(rng.randrange(len(strat[k]))))
        i += 1
    print(f"  need {need} of {len(units)} units ({SPOTCHECK_FRACTION:.0%}), "
          f"stratified by unit_type x arm:")
    print(f"  {sorted(sample)}")
    print("\n  Agreement rule (§8.2): >=70% -> LLM verdicts stand; "
          "<70% -> LLM verdicts void, full human adjudication.")


if __name__ == "__main__":
    main()

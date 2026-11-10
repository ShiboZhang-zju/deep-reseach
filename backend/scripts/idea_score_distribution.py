"""Analyse the (gate decision, LLM final_score) joint distribution.

Why this exists
---------------
`decision` comes from the hard gates (code) and `final_score` from the LLM
rubric — two independent axes. Probe 457dac9a produced an `executable_candidate`
scored 0.4681 against a 0.70 go line and the trace logged
`GATE_PASS_SCORE_BELOW_LINE`: the two axes disagreed on the same Idea, and
there was no data to say which one was right.

This script answers, from the whole database rather than one anecdote:
  1. how often do the axes disagree, and in which direction;
  2. is `final_score` distributed around 0.70 at all (is 0.70 calibrated?);
  3. does a gate-passed Idea with a low score differ systematically from a
     gate-passed Idea with a high score (gap type, novelty gate, tier);
  4. what would a score cut-off cost / save.

Read-only. Usage:
    python scripts/idea_score_distribution.py
    python scripts/idea_score_distribution.py --json out.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

GO_LINE = 0.70


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def collect(db) -> dict:
    from sqlalchemy import text as sql

    rows = db.execute(sql("""
        SELECT ri.decision, ri.final_score, ri.idea_status, ri.confidence_tier,
               gc.gap_type, ic.novelty_gate, ic.feasibility_gate, ic.evidence_gate
        FROM research_ideas ri
        LEFT JOIN intervention_candidates ic ON ic.id = ri.intervention_id
        LEFT JOIN gap_candidates gc ON gc.id = ic.gap_id
        WHERE ri.final_score IS NOT NULL
    """)).fetchall()

    by_decision: dict[str, list[float]] = defaultdict(list)
    cross: Counter = Counter()
    by_gap_type: dict[str, list[float]] = defaultdict(list)
    novelty_pair: Counter = Counter()
    scores_all: list[float] = []
    tiers: Counter = Counter()

    for decision, score, status, tier, gap_type, nov_gate, feas_gate, ev_gate in rows:
        score = float(score)
        decision = (decision or "unknown").strip()
        by_decision[decision].append(score)
        scores_all.append(score)
        tiers[tier or "-"] += 1
        if gap_type:
            by_gap_type[gap_type].append(score)
        gate_ok = decision == "executable_candidate"
        score_ok = score >= GO_LINE
        cross[(gate_ok, score_ok)] += 1
        novelty_pair[(nov_gate or "-", "gate_ok" if gate_ok else "gate_block")] += 1

    scores_all.sort()

    def _summary(vals):
        if not vals:
            return None
        s = sorted(vals)
        return {
            "n": len(s),
            "min": round(s[0], 3),
            "p25": round(_percentile(s, 0.25), 3),
            "median": round(_percentile(s, 0.50), 3),
            "p75": round(_percentile(s, 0.75), 3),
            "max": round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3),
            "above_go_line": sum(1 for x in s if x >= GO_LINE),
        }

    # What each candidate cut-off would keep / drop, among gate-passed ideas.
    passed = sorted(by_decision.get("executable_candidate", []))
    cutoffs = {}
    for cut in (0.50, 0.55, 0.60, 0.65, 0.70):
        kept = sum(1 for x in passed if x >= cut)
        cutoffs[f"{cut:.2f}"] = {
            "kept": kept,
            "dropped": len(passed) - kept,
            "keep_rate": round(kept / len(passed), 3) if passed else None,
        }

    return {
        "total_scored": len(scores_all),
        "overall": _summary(scores_all),
        "by_decision": {k: _summary(v) for k, v in sorted(by_decision.items())},
        "by_gap_type": {k: _summary(v) for k, v in sorted(by_gap_type.items())},
        "axes_cross": {
            "gate_pass_score_ok": cross[(True, True)],
            "gate_pass_score_low": cross[(True, False)],
            "gate_block_score_ok": cross[(False, True)],
            "gate_block_score_low": cross[(False, False)],
        },
        "novelty_gate_vs_decision": {
            f"{k[0]}|{k[1]}": v for k, v in novelty_pair.items()
        },
        "confidence_tiers": dict(tiers),
        "cutoff_simulation_on_gate_passed": cutoffs,
    }


def _fmt(s):
    if not s:
        return "n/a"
    return (f"n={s['n']:<4} min={s['min']:.2f} p25={s['p25']:.2f} "
            f"med={s['median']:.2f} p75={s['p75']:.2f} max={s['max']:.2f} "
            f"mean={s['mean']:.2f} >=0.70:{s['above_go_line']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write full report as JSON")
    args = ap.parse_args()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import settings

    engine = create_engine(settings.database_url)
    db = sessionmaker(bind=engine)()
    try:
        rep = collect(db)
    finally:
        db.close()
        engine.dispose()

    print(f"Idea (decision, final_score) distribution — go line {GO_LINE}")
    print(f"  total scored ideas: {rep['total_scored']}\n")

    print("OVERALL")
    print("  " + _fmt(rep["overall"]) + "\n")

    print("BY DECISION")
    for k, v in rep["by_decision"].items():
        print(f"  {k:<24} {_fmt(v)}")
    print()

    print("AXES CROSS (decision axis x score axis)")
    c = rep["axes_cross"]
    print(f"  gate PASS + score >= {GO_LINE:.2f} (agree)   : {c['gate_pass_score_ok']}")
    print(f"  gate PASS + score <  {GO_LINE:.2f} (disagree) : {c['gate_pass_score_low']}")
    print(f"  gate BLOCK + score >= {GO_LINE:.2f} (disagree): {c['gate_block_score_ok']}")
    print(f"  gate BLOCK + score <  {GO_LINE:.2f} (agree)   : {c['gate_block_score_low']}\n")

    print("BY GAP TYPE")
    for k, v in rep["by_gap_type"].items():
        print(f"  {k:<24} {_fmt(v)}")
    print()

    print("CONFIDENCE TIERS")
    for k, v in sorted(rep["confidence_tiers"].items()):
        print(f"  {k}: {v}")
    print()

    print("CUT-OFF SIMULATION (applied to gate-passed ideas only)")
    print(f"  {'cut':<6} {'kept':>5} {'dropped':>8} {'keep_rate':>10}")
    for cut, d in rep["cutoff_simulation_on_gate_passed"].items():
        print(f"  {cut:<6} {d['kept']:>5} {d['dropped']:>8} {str(d['keep_rate']):>10}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()

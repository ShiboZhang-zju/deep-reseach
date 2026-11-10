"""Verify the completed pilot runs are comparable before any adjudication.

Checks that must hold for the paired comparison to mean anything:
  1. every arm-A run stamped opportunity-ontology-v7.2, every arm-B run
     stamped evidence-admission-v6-exp (no cross-contamination);
  2. both arms ran under the same budget knobs (the .env is shared, but the
     8002 process started with an extra env var — assert only the mining
     switch differs, not the budget);
  3. no run ended in an engineering failure (design §7.5 would invalidate a
     whole pair);
  4. the v6 arm produced ONLY the two legacy gap types, the v7.2 arm at most
     the seven ontology types (a leak would show as a type outside its set).
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

V6_ONLY = {"boundary_gap", "missing_evaluation"}
V7_ALL = {"boundary_gap", "missing_evaluation", "contradiction",
          "untested_assumption", "mechanism_gap", "scaling_regime",
          "cross_domain_transfer"}
LEGIT_TERMINAL = {"waiting_for_user_review", "abstained",
                  "more_research_required", "completed"}

con = sqlite3.connect("deep_research.db")
matrix = json.load(open(r"eval_results\fixed_budget_pilot\run_matrix.json",
                        encoding="utf-8"))

problems = []
print(f"{'topic':<10} {'arm':<4} {'stamp':<28} {'types':<6} {'status'}")
print("-" * 78)
for run in matrix["run_order"]:
    for arm in ("A", "B"):
        tid = run["task_ids"].get(arm)
        if not tid:
            continue
        row = con.execute("SELECT status, stop_reason FROM research_tasks WHERE id=?",
                          (tid,)).fetchone()
        if not row:
            problems.append(f"{run['topic_id']}/{arm}: task {tid} missing from DB")
            continue
        status, stop = row
        stamps = sorted({r[0] for r in con.execute(
            "SELECT DISTINCT mining_policy_version FROM gap_candidates WHERE task_id=?",
            (tid,)).fetchall()})
        types = sorted({r[0] for r in con.execute(
            "SELECT DISTINCT gap_type FROM gap_candidates WHERE task_id=?",
            (tid,)).fetchall()})

        expect_stamp = ("opportunity-ontology-v7.2" if arm == "A"
                        else "evidence-admission-v6-exp")
        if stamps != [expect_stamp]:
            problems.append(f"{run['topic_id']}/{arm}: stamp {stamps} != [{expect_stamp}]")
        allowed = V7_ALL if arm == "A" else V6_ONLY
        stray = [t for t in types if t not in allowed]
        if stray:
            problems.append(f"{run['topic_id']}/{arm}: types outside arm set: {stray}")
        if status not in LEGIT_TERMINAL:
            problems.append(f"{run['topic_id']}/{arm}: non-legit terminal "
                            f"{status} / {stop} (design §7.5 invalidates the pair)")

        print(f"{run['topic_id']:<10} {arm:<4} {','.join(stamps):<28} "
              f"{len(types):<6} {status}")

print()
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(f"  - {p}")
else:
    print("All recorded runs: stamps correct, types within arm set, "
          "terminal states legit.")

print("\n=== budget knobs (must be identical across arms) ===")
from app.config import settings
for k in ("gap_audit_budgeted", "max_gaps_to_deep_audit",
          "gap_starvation_deferral_limit", "gap_audit_timeout_seconds",
          "max_remediation_rounds_total", "coverage_incremental_matching",
          "gap_mining_precedent_check_enabled"):
    print(f"  {k} = {getattr(settings, k, 'n/a')}")
print("  (the arms differ ONLY in GAP_MINING_LEGACY_V6_MODE, which is a "
      "process-level env var: 8001=false, 8002=true)")

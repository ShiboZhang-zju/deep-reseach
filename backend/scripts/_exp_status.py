"""Mid-experiment status table from run_matrix.json + DB (read-only)."""
import json
import sqlite3

con = sqlite3.connect("deep_research.db")
matrix = json.load(open(r"eval_results\fixed_budget_pilot\run_matrix.json",
                        encoding="utf-8"))

print(f"{'topic':<10} {'stratum':<17} {'arm':<4} {'status':<24} {'ideas(exec/dir/cond)'}")
print("-" * 80)
for run in matrix["run_order"]:
    for arm in ("A", "B"):
        tid = run["task_ids"].get(arm)
        if not tid:
            continue
        out = run.get("outcomes", {}).get(arm, {})
        ideas = out.get("ideas", {})
        n_exec = ideas.get("executable_candidate", 0)
        n_dir = ideas.get("research_direction_only", 0)
        n_cond = ideas.get("conditional_review", 0)
        print(f"{run['topic_id']:<10} {run['stratum']:<17} {arm:<4} "
              f"{out.get('status', '?'):<24} "
              f"{n_exec} / {n_dir} / {n_cond}")

print()
print("gap type diversity per arm (distinct types among candidates):")
for arm in ("A", "B"):
    types = set()
    for run in matrix["run_order"]:
        tid = run["task_ids"].get(arm)
        if not tid:
            continue
        for (t,) in con.execute(
                "SELECT DISTINCT gap_type FROM gap_candidates WHERE task_id=?",
                (tid,)).fetchall():
            types.add(t)
    print(f"  arm {arm}: {len(types)} types — {sorted(types)}")

"""Smoke check: is the 8002 task actually running in v6 mode?

Waits for the first gap_mining_candidates trace, then asserts:
  - lens_fanout has exactly the two v6 lenses;
  - accepted candidates carry the evidence-admission-v6-exp stamp.
Also records the task into run_matrix.json (run log of record).
"""
import json
import sqlite3
import sys

TASK = "3d954c12-d799-4c07-b5a8-a905089220c5"
MATRIX = r"eval_results\fixed_budget_pilot\run_matrix.json"

con = sqlite3.connect("deep_research.db")

status = con.execute("SELECT status FROM research_tasks WHERE id=?",
                     (TASK,)).fetchone()
print(f"task status: {status[0] if status else 'NOT FOUND'}")

trace = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? "
    "AND step_name='gap_mining_candidates' ORDER BY created_at DESC LIMIT 1",
    (TASK,)).fetchone()

if not trace:
    print("no mining trace yet (still in search/extract/coverage) — rerun later")
    sys.exit(0)

payload = json.loads(trace[0])
fanout = [lf["lens"] for lf in payload.get("lens_fanout", [])]
print(f"lens_fanout: {fanout}")
v6_ok = fanout == ["boundary_gap", "missing_evaluation"]

stamps = [r[0] for r in con.execute(
    "SELECT DISTINCT mining_policy_version FROM gap_candidates WHERE task_id=?",
    (TASK,)).fetchall()]
print(f"candidate stamps: {stamps}")
stamp_ok = stamps and all(s == "evidence-admission-v6-exp" for s in stamps)

rejections = [r["reason"] for r in payload.get("rejected_candidates", [])]
print(f"rejections: {rejections[:8]}")

if v6_ok and stamp_ok:
    print("\nV6 MODE CONFIRMED on 8002 — arm B is running the control policy.")
else:
    print("\n!!! MODE MISMATCH — kill 8002, restart with the env var, "
          "re-run this topic per interruption rule 7.5")

# Record into run_matrix.json regardless (audit trail).
with open(MATRIX, encoding="utf-8") as fh:
    matrix = json.load(fh)
for run in matrix["run_order"]:
    if run["topic_id"] == "pe2e-001" and "B" not in run["task_ids"]:
        run["task_ids"]["B"] = TASK
        break
with open(MATRIX, "w", encoding="utf-8") as fh:
    json.dump(matrix, fh, ensure_ascii=False, indent=2)
print("\nrun_matrix.json updated (pe2e-001 arm B).")

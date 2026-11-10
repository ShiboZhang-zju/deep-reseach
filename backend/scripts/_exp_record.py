"""Record a completed run's outcome into run_matrix.json (run log of record)."""
import json
import sqlite3
import sys

topic_id, arm, task_id = sys.argv[1], sys.argv[2], sys.argv[3]

con = sqlite3.connect("deep_research.db")
row = con.execute(
    "SELECT status, stop_reason FROM research_tasks WHERE id=?", (task_id,)).fetchone()
gaps = con.execute(
    "SELECT gap_type, status, COUNT(*) FROM gap_candidates WHERE task_id=? "
    "GROUP BY 1, 2 ORDER BY 1", (task_id,)).fetchall()
ideas = con.execute(
    "SELECT decision, COUNT(*) FROM research_ideas WHERE task_id=? GROUP BY 1",
    (task_id,)).fetchall()
papers = con.execute(
    "SELECT COUNT(*) FROM task_papers WHERE task_id=?", (task_id,)).fetchone()[0]

outcome = {
    "status": row[0], "stop_reason": row[1],
    "papers": papers,
    "gaps": {f"{t}/{s}": n for t, s, n in gaps},
    "ideas": {d: n for d, n in ideas},
}

path = r"eval_results\fixed_budget_pilot\run_matrix.json"
with open(path, encoding="utf-8") as fh:
    matrix = json.load(fh)
for run in matrix["run_order"]:
    if run["topic_id"] == topic_id:
        run["task_ids"][arm] = task_id
        run.setdefault("outcomes", {})[arm] = outcome
        break
with open(path, "w", encoding="utf-8") as fh:
    json.dump(matrix, fh, ensure_ascii=False, indent=2)

print(f"{topic_id} arm {arm} recorded: {outcome['status']} / {outcome['stop_reason']}")
print(f"  gaps: {outcome['gaps']}")
print(f"  ideas: {outcome['ideas']}")

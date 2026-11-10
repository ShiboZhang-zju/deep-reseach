"""Record a task id into run_matrix.json BEFORE the run finishes.

The run matrix is the experiment's log of record; losing an in-flight task id
to a host restart would leave an unlabelled run in the DB (already a real risk:
the host died once mid-experiment).

Usage: python scripts/_exp_assign.py <topic_id> <arm> <task_id>
"""
import json
import sys

topic_id, arm, task_id = sys.argv[1], sys.argv[2], sys.argv[3]
path = r"eval_results\fixed_budget_pilot\run_matrix.json"

with open(path, encoding="utf-8") as fh:
    matrix = json.load(fh)

for run in matrix["run_order"]:
    if run["topic_id"] == topic_id:
        run["task_ids"][arm] = task_id
        break
else:
    raise SystemExit(f"unknown topic {topic_id}")

with open(path, "w", encoding="utf-8") as fh:
    json.dump(matrix, fh, ensure_ascii=False, indent=2)

print(f"assigned {topic_id} arm {arm} -> {task_id}")

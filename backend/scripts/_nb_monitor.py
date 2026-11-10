"""Monitor the neighbour-budget experiment (read-only).

Reports, per arm: task status, phase progress, and — once audits run — the
`gap_audit_neighbor_budget` trace that proves the arm used its configured
neighbour count. That trace is the whole point of the instrumentation: an arm
that silently ran with 3 neighbours would otherwise be indistinguishable from
one that ran with 8.

Usage:
    python scripts/_nb_monitor.py            # one snapshot
    python scripts/_nb_monitor.py --watch 60 # poll every 60s
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

DB = "deep_research.db"
MATRIX = os.path.join("eval_results", "neighbor_budget", "run_matrix.json")


def _arms() -> list[tuple[str, str, str]]:
    """(arm, topic_id, task_id) for every recorded run."""
    with open(MATRIX, encoding="utf-8") as fh:
        matrix = json.load(fh)
    rows = []
    for run in matrix["run_order"]:
        for arm, tid in (run.get("task_ids") or {}).items():
            rows.append((arm, run["topic_id"], tid))
    return rows


def snapshot() -> int:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    arms = _arms()
    if not arms:
        print("no task ids recorded in run_matrix.json yet")
        return 1
    active = 0
    for arm, topic, tid in arms:
        task = con.execute(
            "select status, current_round from research_tasks where id=?",
            (tid,)).fetchone()
        if task is None:
            print(f"{topic} arm {arm}: TASK MISSING")
            continue
        status = task["status"]
        if status not in ("waiting_for_user_review", "failed", "abstained",
                          "more_research_required", "stopped", "completed"):
            active += 1
        papers = con.execute(
            "select count(*) from task_papers where task_id=?",
            (tid,)).fetchone()[0]
        gaps = con.execute(
            "select count(*) from gap_candidates where task_id=?",
            (tid,)).fetchone()[0]
        print(f"{topic} arm {arm}: {status} r{task['current_round']} "
              f"papers={papers} gaps={gaps}  ({tid[:8]})")

        traces = con.execute(
            "select output_json from agent_traces where task_id=? "
            "and step_name='gap_audit_neighbor_budget' "
            "order by created_at desc limit 5", (tid,)).fetchall()
        for tr in traces:
            payload = json.loads(tr[0])
            print(f"    audit: limit={payload['effective_limit']} "
                  f"selected={payload['selected_count']} "
                  f"cap_bound={payload['cap_bound']} "
                  f"pool={payload['candidate_pool_size']} "
                  f"below_floor={payload['selected_below_floor']}")
    con.close()
    return active


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0,
                        help="poll interval seconds (0 = single snapshot)")
    args = parser.parse_args(argv)
    if not args.watch:
        snapshot()
        return 0
    while True:
        os.system("cls")
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        active = snapshot()
        print(f"\nactive tasks: {active}")
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())

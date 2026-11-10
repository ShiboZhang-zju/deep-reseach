"""Read-only audit of killer-directed search effectiveness.

Background (probe df744606 / 0ae1a146): the uncertain-verdict killer search
(P1, 2026-10-30) only pays off when the audit's own killer description is
retrievable in community vocabulary. In probe B it executed once and found
nothing, so the question "is this guard buying anything?" is still open — and
answering it needs the hit RATE, not one anecdote.

This tool aggregates the traces the guard writes:
  - `gap_audit_uncertain_killer_search` : ran / skip_reason / hit_count
  - `gap_audit_killer_found`            : confirmed-path hit (downgrade)
  - `gap_audit_uncertain_killer_found`  : uncertain-path hit (close)

Usage:
    python scripts/killer_search_audit.py                 # all tasks
    python scripts/killer_search_audit.py --task <id>     # one task
    python scripts/killer_search_audit.py --json out.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _connect():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import settings

    engine = create_engine(settings.database_url)
    return sessionmaker(bind=engine)(), engine


def collect(db, task_id=None) -> dict:
    from sqlalchemy import text as sql_text

    where = "AND task_id = :tid" if task_id else ""
    params = {"tid": task_id} if task_id else {}

    rows = db.execute(sql_text(f"""
        SELECT task_id, step_name, output_json FROM agent_traces
        WHERE step_name IN (
            'gap_audit_uncertain_killer_search',
            'gap_audit_uncertain_killer_found',
            'gap_audit_killer_found'
        ) {where}
    """), params).fetchall()

    executed = hits = degraded = 0
    skip_reasons: Counter = Counter()
    confirmed_path_hits = Uncertain_path_hits = 0
    per_task: dict[str, dict] = {}

    for task, step, payload in rows:
        bucket = per_task.setdefault(
            str(task)[:8], {"runs": 0, "hits": 0, "skipped": 0, "degraded": 0})
        try:
            data = json.loads(payload) if payload else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
        if step == "gap_audit_uncertain_killer_search":
            if data.get("ran"):
                executed += 1
                bucket["runs"] += 1
                if data.get("search_degraded"):
                    degraded += 1
                    bucket["degraded"] += 1
                if (data.get("hit_count") or 0) > 0:
                    hits += 1
                    bucket["hits"] += 1
            else:
                bucket["skipped"] += 1
                skip_reasons[data.get("skip_reason") or "unknown"] += 1
        elif step == "gap_audit_uncertain_killer_found":
            Uncertain_path_hits += 1
        elif step == "gap_audit_killer_found":
            confirmed_path_hits += 1

    return {
        "uncertain_path": {
            "executed": executed,
            "hits": hits,
            "degraded": degraded,
            "hit_rate": round(hits / executed, 3) if executed else None,
            "skip_reasons": dict(skip_reasons),
        },
        "hit_events": {
            "uncertain_path_closed": Uncertain_path_hits,
            "confirmed_path_downgraded": confirmed_path_hits,
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="restrict to one task id")
    parser.add_argument("--json", help="write the report as JSON here")
    args = parser.parse_args()

    db, engine = _connect()
    try:
        report = collect(db, args.task)
    finally:
        db.close()
        engine.dispose()

    up = report["uncertain_path"]
    print("Killer-directed search effectiveness")
    print(f"  scope: {'task ' + args.task if args.task else 'all tasks'}")
    print()
    print(f"  uncertain path executed : {up['executed']}")
    print(f"  hits                    : {up['hits']}")
    rate = "n/a (no executions)" if up["hit_rate"] is None else f"{up['hit_rate']:.1%}"
    print(f"  hit rate                : {rate}")
    print(f"  degraded (infra)        : {up['degraded']}")
    if up["skip_reasons"]:
        print(f"  skipped                 : {up['skip_reasons']}")
    print()
    print(f"  closes (uncertain hit)  : {report['hit_events']['uncertain_path_closed']}")
    print(f"  downgrades (confirmed)  : {report['hit_events']['confirmed_path_downgraded']}")
    if report["per_task"]:
        print("\n  per task (runs / hits / skipped / degraded):")
        for tid, b in sorted(report["per_task"].items()):
            print(f"    {tid}  {b['runs']:3d} / {b['hits']:3d} / "
                  f"{b['skipped']:3d} / {b['degraded']:3d}")

    if up["executed"] == 0:
        print("\n  NOTE: no executions recorded. The trace only exists after "
              "2026-10-30; older runs are invisible by design.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n  JSON written to {args.json}")


if __name__ == "__main__":
    main()

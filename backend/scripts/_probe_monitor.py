"""Probe task monitor — one-shot snapshot of the live pipeline run."""
import sys

import sqlite3

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "df744606-5788-4c8a-9153-f1b9a389fce4"

con = sqlite3.connect("deep_research.db")
con.row_factory = sqlite3.Row

t = con.execute("SELECT status, current_round, max_rounds, stop_reason FROM research_tasks WHERE id=?", (TASK_ID,)).fetchone()
print(f"task: status={t['status']} round={t['current_round']}/{t['max_rounds']} stop={t['stop_reason']}")

print("\nrounds:")
for r in con.execute(
    "SELECT round_number, papers_found, new_papers, duplicate_rate FROM research_rounds "
    "WHERE task_id=? ORDER BY round_number", (TASK_ID,)).fetchall():
    print(f"  r{r['round_number']}: found={r['papers_found']} new={r['new_papers']} dup={r['duplicate_rate']:.2f}")

print("\ntask_papers by priority:")
for r in con.execute(
    "SELECT priority, COUNT(*) n FROM task_papers WHERE task_id=? GROUP BY priority ORDER BY n DESC", (TASK_ID,)).fetchall():
    print(f"  {r['priority']}: {r['n']}")

print("extraction_status:")
for r in con.execute(
    "SELECT COALESCE(extraction_status,'NULL') s, COUNT(*) n FROM task_papers WHERE task_id=? GROUP BY 1", (TASK_ID,)).fetchall():
    print(f"  {r['s']}: {r['n']}")

print(f"\nevidence_units: {con.execute('SELECT COUNT(*) FROM evidence_units WHERE task_id=?', (TASK_ID,)).fetchone()[0]}")
print("  by type:", dict(con.execute(
    "SELECT evidence_type, COUNT(*) FROM evidence_units WHERE task_id=? GROUP BY 1", (TASK_ID,)).fetchall()))
print("  by method:", dict(con.execute(
    "SELECT extraction_method, COUNT(*) FROM evidence_units WHERE task_id=? GROUP BY 1", (TASK_ID,)).fetchall()))

print("\nphase_runs (completed, with duration):")
for r in con.execute("""
    SELECT pr.phase_name, pr.status, pr.round_number, t.duration_ms
    FROM phase_runs pr
    LEFT JOIN agent_traces t ON t.task_id = pr.task_id
        AND t.step_name = pr.phase_name AND t.step_type = 'phase_duration'
    WHERE pr.task_id = ?
    ORDER BY pr.created_at
""", (TASK_ID,)).fetchall():
    dur = f"{r['duration_ms']/1000:.0f}s" if r["duration_ms"] else "-"
    print(f"  {r['phase_name']:<42} {r['status']:<10} r{r['round_number'] or '-'} {dur}")

print("\nlast 10 traces:")
for r in con.execute(
    "SELECT step_name, step_type, round_number, created_at FROM agent_traces "
    "WHERE task_id=? ORDER BY created_at DESC LIMIT 10", (TASK_ID,)).fetchall():
    print(f"  {r['created_at']} {r['step_name']} [{r['step_type']}] r{r['round_number']}")

print(f"\ngap_candidates: {con.execute('SELECT COUNT(*) FROM gap_candidates WHERE task_id=?', (TASK_ID,)).fetchone()[0]}")
for r in con.execute(
    "SELECT gap_type, status, COUNT(*) n FROM gap_candidates WHERE task_id=? GROUP BY 1,2", (TASK_ID,)).fetchall():
    print(f"  {r['gap_type']}/{r['status']}: {r['n']}")

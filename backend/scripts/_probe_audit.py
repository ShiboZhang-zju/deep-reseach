"""Probe audit + remediation details."""
import json
import sys

import sqlite3

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "df744606-5788-4c8a-9153-f1b9a389fce4"
con = sqlite3.connect("deep_research.db")

print("=== audit_gap_candidate decision ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='audit_gap_candidate' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
if row:
    p = json.loads(row[0])
    for k, v in p.items():
        print(f"  {k} = {str(v)[:160]}")

print("\n=== budgeted_partial_survive ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='budgeted_partial_survive' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
if row:
    print(json.dumps(json.loads(row[0]), ensure_ascii=False)[:600])

print("\n=== gap_audit_text_verdict_conflict ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_audit_text_verdict_conflict' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
if row:
    print(json.dumps(json.loads(row[0]), ensure_ascii=False)[:600])

print("\n=== remediation / targeted research traces ===")
for row in con.execute(
    "SELECT step_name, output_json FROM agent_traces WHERE task_id=? "
    "AND (step_name LIKE '%remediation%' OR step_name LIKE '%targeted%' OR step_name LIKE '%narrow%') "
    "ORDER BY created_at DESC LIMIT 6", (TASK_ID,)).fetchall():
    print(f"  {row[0]}: {str(row[1])[:220]}")

print("\n=== gap_audits table ===")
for r in con.execute("""
    SELECT ga.audit_result, ga.recommended_action, ga.novelty_confidence, ga.remaining_delta
    FROM gap_audits ga JOIN gap_candidates gc ON gc.id = ga.gap_id
    WHERE ga.task_id=?
""", (TASK_ID,)).fetchall():
    print(f"  result={r[0]} action={r[1]} novelty={r[2]} remaining={str(r[3])[:120]}")

print("\n=== O2 round phases (round 6+) ===")
for r in con.execute("""
    SELECT phase_name, status FROM phase_runs WHERE task_id=? AND phase_name LIKE '%_round_6%'
""", (TASK_ID,)).fetchall():
    print(f"  {r[0]}: {r[1]}")

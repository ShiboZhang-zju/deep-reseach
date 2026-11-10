"""Probe downstream: surviving gaps, remediation state, latest decisions."""
import json
import sys

import sqlite3

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "df744606-5788-4c8a-9153-f1b9a389fce4"
con = sqlite3.connect("deep_research.db")

print("=== gap statuses ===")
for g in con.execute(
    "SELECT id, gap_type, status, claimed_delta FROM gap_candidates WHERE task_id=? ORDER BY created_at",
    (TASK_ID,)).fetchall():
    print(f"  {g[0][:8]} {g[1]:<22} {g[2]:<11} {g[3][:100]}")

print("\n=== gap_provisional_survive_after_remediation ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? "
    "AND step_name='gap_provisional_survive_after_remediation' ORDER BY created_at DESC LIMIT 1",
    (TASK_ID,)).fetchone()
if row:
    print(json.dumps(json.loads(row[0]), ensure_ascii=False)[:500])

print("\n=== latest 15 traces ===")
for r in con.execute(
    "SELECT created_at, step_name, step_type FROM agent_traces WHERE task_id=? "
    "ORDER BY created_at DESC LIMIT 15", (TASK_ID,)).fetchall():
    print(f"  {r[0]} {r[1]} [{r[2]}]")

print("\n=== audits ===")
for r in con.execute("""
    SELECT gc.id, gc.gap_type, ga.audit_result, ga.recommended_action, ga.novelty_confidence
    FROM gap_audits ga JOIN gap_candidates gc ON gc.id = ga.gap_id
    WHERE ga.task_id=? ORDER BY ga.created_at
""", (TASK_ID,)).fetchall():
    print(f"  {r[0][:8]} {r[1]:<22} {r[2]:<10} action={r[3]} novelty={r[4]}")

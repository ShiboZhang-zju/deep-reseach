"""Probe mining details: v7.2 gate verdicts + admission funnels."""
import json
import sys

import sqlite3

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "df744606-5788-4c8a-9153-f1b9a389fce4"
con = sqlite3.connect("deep_research.db")

print("=== gap_mining_candidates trace ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_mining_candidates' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
payload = json.loads(row[0])
print("lens_fanout:")
for lf in payload["lens_fanout"]:
    print(f"  {lf['lens']:<22} llm={lf['llm_candidate_count']} typed={lf['type_matched_count']} accepted={lf['accepted_count']}")
print("accepted:")
for a in payload["accepted_candidates"]:
    print(f"  {a['gap_id'][:8]} {a['gap_type']}")
print("rejected:")
for r in payload["rejected_candidates"]:
    extra = f" covering={r.get('covering_paper_id', '')[:8]}" if r.get("covering_paper_id") else ""
    print(f"  {r['reason']:<38} lens={r['lens']}{extra}")

print("\n=== scaling_precedent_check traces ===")
for row in con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='scaling_precedent_check' "
    "ORDER BY created_at", (TASK_ID,)).fetchall():
    p = json.loads(row[0])
    print(f"  verdict={p['verdict']:<8} covering={str(p.get('covering_paper_id'))[:8]} "
          f"queries={p.get('queries')} claim={p['claimed_delta'][:90]}")

print("\n=== question admission (gap_mining_admission) ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_mining_admission' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
if row:
    p = json.loads(row[0])
    for qr in p.get("question_results", []):
        print(f"  {qr['question_id'][:8]} {qr['status']:<8} reasons={qr['reason_codes']} "
              f"papers={qr['supporting_paper_count']} spans={qr['verified_span_count']}")

print("\n=== gap details ===")
for g in con.execute(
    "SELECT id, gap_type, claimed_delta, status FROM gap_candidates WHERE task_id=?", (TASK_ID,)).fetchall():
    print(f"  {g[0][:8]} {g[1]:<22} {g[3]:<10} {g[2][:110]}")

print("\n=== gap_competition trace ===")
row = con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_competition' "
    "ORDER BY created_at DESC LIMIT 1", (TASK_ID,)).fetchone()
if row:
    p = json.loads(row[0])
    print(json.dumps(p, ensure_ascii=False)[:800])

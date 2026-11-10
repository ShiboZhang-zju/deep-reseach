"""Final harvest: ideas, experiments, v7.2 gates, funnel totals, cost."""
import json
import sys

import sqlite3

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "df744606-5788-4c8a-9153-f1b9a389fce4"
con = sqlite3.connect("deep_research.db")

print("=== IDEAS ===")
for r in con.execute("""
    SELECT ri.id, ri.title, ri.decision, ri.idea_status, ri.final_score, ri.quality_reason_codes_json,
           gc.gap_type, gc.claimed_delta
    FROM research_ideas ri LEFT JOIN gap_candidates gc ON gc.id = ri.gap_id
    WHERE ri.task_id=? ORDER BY ri.final_score DESC NULLS LAST
""", (TASK_ID,)).fetchall():
    print(f"  {r[0][:8]} [{r[3]}/{r[2]}] score={r[4]} type={r[6]}")
    print(f"    title: {r[1]}")
    print(f"    reasons: {r[5]}")
    print(f"    gap delta: {str(r[7])[:120]}")

print("\n=== EXPERIMENT PLANS ===")
for r in con.execute("""
    SELECT ep.id, ep.hypothesis, ep.dataset, ep.statistical_analysis
    FROM experiment_plans ep WHERE ep.task_id=?
""", (TASK_ID,)).fetchall():
    print(f"  {r[0][:8]} hyp={str(r[1])[:110]}")
    print(f"    dataset={str(r[2])[:80]} stats={str(r[3])[:80]}")

print("\n=== v7.2 GATE ACTIVITY (all mining cycles) ===")
n = 0
for row in con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='scaling_precedent_check' "
    "ORDER BY created_at", (TASK_ID,)).fetchall():
    p = json.loads(row[0])
    n += 1
    print(f"  [{n}] verdict={p['verdict']} covering={str(p.get('covering_paper_id'))[:8]}")
    print(f"      queries={p.get('queries')}")
    print(f"      claim={p['claimed_delta'][:110]}")
if not n:
    print("  (no scaling_precedent_check traces)")

perf_rej = 0
for row in con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_mining_candidates' "
    "ORDER BY created_at", (TASK_ID,)).fetchall():
    p = json.loads(row[0])
    for rj in p.get("rejected_candidates", []):
        if rj.get("reason") == "PERFORMANCE_TARGET_DELTA":
            perf_rej += 1
            print(f"  PERFORMANCE_TARGET_DELTA rejected: lens={rj['lens']}")
print(f"  PERFORMANCE_TARGET_DELTA total rejections: {perf_rej}")

print("\n=== MINING FUNNEL (3 cycles aggregated) ===")
from collections import Counter
reasons = Counter()
lens_accepted = Counter()
for row in con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='gap_mining_candidates' "
    "ORDER BY created_at", (TASK_ID,)).fetchall():
    p = json.loads(row[0])
    for lf in p["lens_fanout"]:
        lens_accepted[lf["lens"]] += lf["accepted_count"]
    for rj in p.get("rejected_candidates", []):
        reasons[rj["reason"]] += 1
print("  accepted by lens:", dict(lens_accepted))
print("  rejection reasons:", dict(reasons))

print("\n=== LLM COST ===")
row = con.execute("""
    SELECT COUNT(*), SUM(COALESCE(llm_tokens_used,0)) FROM agent_traces WHERE task_id=?
""", (TASK_ID,)).fetchone()
print(f"  traces={row[0]} tokens_recorded={row[1]}")
row = con.execute("""
    SELECT COUNT(DISTINCT step_name) FROM agent_traces WHERE task_id=?
""", (TASK_ID,)).fetchone()
print(f"  distinct step kinds={row[0]}")

print("\n=== REPORTS ===")
for r in con.execute(
    "SELECT id, LENGTH(content_markdown) FROM reports WHERE task_id=?", (TASK_ID,)).fetchall():
    print(f"  report {r[0][:8]} len={r[1]}")

print("\n=== AUDIT OUTCOMES (all) ===")
for r in con.execute("""
    SELECT gc.gap_type, ga.audit_result, ga.recommended_action, ga.novelty_confidence
    FROM gap_audits ga JOIN gap_candidates gc ON gc.id = ga.gap_id
    WHERE ga.task_id=? ORDER BY ga.created_at
""", (TASK_ID,)).fetchall():
    print(f"  {r[0]:<22} {r[1]:<10} action={r[2]:<14} novelty={r[3]}")

print("\n=== IDEA NOVELTY CHECK VERDICTS ===")
verdicts = Counter()
for row in con.execute(
    "SELECT output_json FROM agent_traces WHERE task_id=? AND step_name='idea_novelty_check' "
    "ORDER BY created_at", (TASK_ID,)).fetchall():
    p = json.loads(row[0])
    verdicts[p.get("verdict")] += 1
print(f"  {dict(verdicts)}")

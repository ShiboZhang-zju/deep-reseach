"""Paired two-arm summary of the fixed-budget pilot (descriptive only).

Deliberately computes only SECONDARY metrics. The pre-registered primary
metric (blind-review true-rate, design §6/§7) requires the §8 adjudication
protocol and must not be inferred from these numbers.
"""
import json
import sqlite3

con = sqlite3.connect("deep_research.db")
matrix = json.load(open(r"eval_results\fixed_budget_pilot\run_matrix.json",
                        encoding="utf-8"))

rows = []
for run in matrix["run_order"]:
    for arm in ("A", "B"):
        tid = run["task_ids"].get(arm)
        out = run.get("outcomes", {}).get(arm)
        if not tid or not out:
            continue
        ideas = out.get("ideas", {})
        gaps = out.get("gaps", {})
        stamps = {status: n for k, n in gaps.items()
                  for status in [k.split("/")[1]]}
        audited = sum(
            n for k, n in gaps.items()
            if k.split("/")[1] in
            {"surviving", "confirmed_runner_up", "inconclusive", "rejected"})
        candidate = sum(n for k, n in gaps.items()
                        if k.split("/")[1] == "candidate")
        types = len({k.split("/")[0] for k in gaps})
        rows.append({
            "topic": run["topic_id"], "stratum": run["stratum"], "arm": arm,
            "status": out["status"],
            "papers": out.get("papers", 0),
            "gaps_total": sum(gaps.values()),
            "gaps_audited": audited,
            "gaps_candidate": candidate,
            "gap_types": types,
            "exec": ideas.get("executable_candidate", 0),
            "dir_only": ideas.get("research_direction_only", 0),
            "cond": ideas.get("conditional_review", 0),
        })

print("=== PER-RUN (all 12) ===")
hdr = (f"{'topic':<10}{'arm':<4}{'status':<25}{'papers':>7}{'gaps':>6}"
       f"{'aud':>5}{'cand':>6}{'types':>7}{'exec':>6}{'dir':>5}{'cond':>6}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['topic']:<10}{r['arm']:<4}{r['status']:<25}{r['papers']:>7}"
          f"{r['gaps_total']:>6}{r['gaps_audited']:>5}{r['gaps_candidate']:>6}"
          f"{r['gap_types']:>7}{r['exec']:>6}{r['dir_only']:>5}{r['cond']:>6}")

print("\n=== PAIRED BY TOPIC ===")
print(f"{'topic':<10}{'stratum':<18}{'A:status':<14}{'B:status':<24}"
      f"{'A:exec':>7}{'B:exec':>7}{'A:types':>8}{'B:types':>8}")
print("-" * 96)
for run in matrix["run_order"]:
    a = next((r for r in rows if r["topic"] == run["topic_id"] and r["arm"] == "A"), None)
    b = next((r for r in rows if r["topic"] == run["topic_id"] and r["arm"] == "B"), None)
    if not a or not b:
        continue
    print(f"{run['topic_id']:<10}{run['stratum']:<18}"
          f"{a['status'][:13]:<14}{b['status'][:23]:<24}"
          f"{a['exec']:>7}{b['exec']:>7}{a['gap_types']:>8}{b['gap_types']:>8}")

print("\n=== ARM TOTALS ===")
for arm in ("A", "B"):
    sub = [r for r in rows if r["arm"] == arm]
    n = len(sub)
    ready = sum(1 for r in sub if r["status"] == "waiting_for_user_review")
    print(f"  arm {arm} (n={n}):")
    print(f"    ideas_ready            : {ready}/{n}")
    print(f"    executable_candidate   : {sum(r['exec'] for r in sub)}")
    print(f"    research_direction_only: {sum(r['dir_only'] for r in sub)}")
    print(f"    conditional_review     : {sum(r['cond'] for r in sub)}")
    print(f"    gap candidates total   : {sum(r['gaps_total'] for r in sub)}")
    print(f"    gap candidates audited : {sum(r['gaps_audited'] for r in sub)}")
    print(f"    still 'candidate'      : {sum(r['gaps_candidate'] for r in sub)}")
    print(f"    papers ingested        : {sum(r['papers'] for r in sub)}")
    print(f"    mean gap types / topic : "
          f"{sum(r['gap_types'] for r in sub)/n:.2f}")

print("\n=== CANDIDATE BACKLOG (starvation check) ===")
for arm in ("A", "B"):
    sub = [r for r in rows if r["arm"] == arm]
    total = sum(r["gaps_total"] for r in sub)
    cand = sum(r["gaps_candidate"] for r in sub)
    print(f"  arm {arm}: {cand}/{total} still 'candidate' "
          f"({cand/total:.1%}) — design target <50%")

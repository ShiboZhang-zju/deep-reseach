"""Extraction supply audit — measure the evidence-supply funnel (read-only).

README §19 bottleneck #1: `NO_LIMITATION_SIGNAL` is the top gap-admission
rejection reason, and the root cause is suspected to be section recognition
rate (papers whose PDF failed to parse fall back to abstract-only). The
`extraction_status` column makes this measurable — this script is that
measurement. It answers, for one task or the whole DB:

  A. Extraction status funnel   — how many candidate papers were actually read
                                  (full_text vs abstract_only vs not_assessed),
                                  and why the rest were not (gap reasons).
  B. Section recognition        — chunk section distribution (share of
                                  `unknown`), evidence section distribution,
                                  and the conclusion-reachability rate: of the
                                  papers extracted from full text, how many
                                  yielded at least one conclusion-section
                                  chunk/evidence.
  C. Evidence type mix          — share of limitation / negative_result signals
                                  relative to other types (what gap admission
                                  consumes).
  D. Admission rejection codes  — reason code counts from gap_mining_admission
                                  traces (the downstream symptom).

Usage:
    python scripts/extraction_audit.py [--db sqlite:///./deep_research.db]
                                       [--task-id <id>] [--json <path>]

Read-only: opens the database with a plain engine, never imports app.config
(so the dev .env cannot redirect it), never writes to the DB.
"""

import argparse
import json
import os
import sys
from collections import Counter

from sqlalchemy import create_engine, text as sql_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECTIONS_OF_INTEREST = ("conclusion", "method", "experiment", "result", "abstract", "unknown")


def _rows(conn, query, params=None):
    return [dict(r._mapping) for r in conn.execute(sql_text(query), params or {})]


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):5.1f}%" if total else "  n/a"


def audit_reading_method(conn, task_id=None):
    """A2. Reading funnel anchored on evidence_units.extraction_method.

    task_papers.extraction_status only exists for tasks that ran AFTER the
    0034 migration — older tasks show NULL even though full-text evidence was
    extracted. evidence_units.extraction_method is the authoritative signal
    across all history: pdf_fulltext means the PDF was parsed and read;
    abstract_only means the fallback path.
    """
    where = "WHERE eu.task_id = :task_id" if task_id else ""
    rows = _rows(conn, f"""
        SELECT eu.extraction_method AS method,
               COUNT(*) AS units,
               COUNT(DISTINCT eu.paper_id) AS papers,
               COUNT(DISTINCT eu.task_id) AS tasks
        FROM evidence_units eu
        {where}
        GROUP BY 1 ORDER BY units DESC
    """, {"task_id": task_id} if task_id else None)
    total_units = sum(r["units"] for r in rows)
    # Gap-admission view: of the tasks that reached gap mining, how many had
    # conclusion-section evidence available at all (the v6 question-level
    # limitation gate consumed exactly that).
    where_eu = "AND eu.task_id = :task_id" if task_id else ""
    mining_tasks = _rows(conn, f"""
        SELECT COUNT(DISTINCT t.task_id) AS n
        FROM agent_traces t
        WHERE t.step_name = 'gap_mining_admission'
              {'AND t.task_id = :task_id' if task_id else ''}
    """, {"task_id": task_id} if task_id else None)[0]["n"]
    conc_tasks = _rows(conn, f"""
        SELECT COUNT(DISTINCT eu.task_id) AS n
        FROM evidence_units eu
        WHERE eu.section = 'conclusion' {where_eu}
          AND eu.task_id IN (
              SELECT DISTINCT t.task_id FROM agent_traces t
              WHERE t.step_name = 'gap_mining_admission'
                {'AND t.task_id = :task_id' if task_id else ''}
          )
    """, {"task_id": task_id} if task_id else None)[0]["n"]
    return {
        "breakdown": rows,
        "total_units": total_units,
        "gap_mining_tasks": mining_tasks,
        "gap_mining_tasks_with_conclusion_evidence": conc_tasks,
    }


def audit_section_reconciliation(conn, task_id=None):
    """B2. Cross-pipeline section reconciliation.

    evidence_units.section is attributed by the EXTRACTION pipeline
    (analyze_papers.extract_pdf_text_by_section), while paper_chunks.section
    is attributed by the RAG pipeline (rag_service._classify_section) — two
    different keyword sets and defaults (introduction vs unknown). Auditing
    one and believing it describes the other produced a wrong bottleneck
    diagnosis once (2026-10-29: 'only 25.5% of full-text papers have
    conclusion chunks' measured the RAG view, while the extraction-side
    conclusion evidence coverage was 100%). This section reports both views
    and their overlap so the two can never be conflated again.
    """
    where_eu = "AND eu.task_id = :task_id" if task_id else ""
    rows = _rows(conn, f"""
        WITH ev AS (
            SELECT DISTINCT eu.paper_id FROM evidence_units eu
            WHERE eu.section = 'conclusion' {where_eu}
        ),
        ch AS (SELECT DISTINCT paper_id FROM paper_chunks WHERE section = 'conclusion'),
        both AS (SELECT DISTINCT eu.paper_id, eu.task_id FROM evidence_units eu)
        SELECT
          (SELECT COUNT(*) FROM ev) AS ev_conclusion_papers,
          (SELECT COUNT(*) FROM ch) AS chunk_conclusion_papers,
          (SELECT COUNT(*) FROM ev JOIN ch ON ev.paper_id = ch.paper_id) AS both_conclusion,
          (SELECT COUNT(*) FROM ev WHERE ev.paper_id NOT IN (SELECT paper_id FROM ch)) AS ev_only,
          (SELECT COUNT(*) FROM ch WHERE ch.paper_id NOT IN (SELECT paper_id FROM ev)) AS chunk_only
    """, {"task_id": task_id} if task_id else None)
    recon = rows[0] if rows else {}
    papers_both = _rows(conn, f"""
        SELECT COUNT(DISTINCT eu.paper_id) AS n
        FROM evidence_units eu
        JOIN paper_chunks pc ON pc.paper_id = eu.paper_id
        WHERE 1=1 {where_eu}
    """, {"task_id": task_id} if task_id else None)[0]["n"]
    recon["papers_with_both_evidence_and_chunks"] = papers_both
    return recon


def audit_extraction_status(conn, task_id=None):
    """A. Candidate-paper reading funnel (task_papers.extraction_status)."""
    where = "WHERE tp.task_id = :task_id" if task_id else ""
    rows = _rows(conn, f"""
        SELECT COALESCE(tp.extraction_status, 'NULL') AS status,
               COALESCE(tp.extraction_gap_reason, '-')   AS reason,
               COUNT(*) AS n
        FROM task_papers tp
        {where}
        GROUP BY 1, 2 ORDER BY n DESC
    """, {"task_id": task_id} if task_id else None)
    total = sum(r["n"] for r in rows)
    per_task = _rows(conn, """
        SELECT tp.task_id,
               SUM(CASE WHEN tp.extraction_status = 'full_text' THEN 1 ELSE 0 END) AS full_text,
               SUM(CASE WHEN tp.extraction_status = 'abstract_only' THEN 1 ELSE 0 END) AS abstract_only,
               SUM(CASE WHEN COALESCE(tp.extraction_status, 'not_assessed') = 'not_assessed' THEN 1 ELSE 0 END) AS not_assessed,
               COUNT(*) AS total
        FROM task_papers tp
        {where}
        GROUP BY tp.task_id ORDER BY total DESC LIMIT 30
    """.replace("{where}", "WHERE tp.task_id = :task_id" if task_id else ""),
        {"task_id": task_id} if task_id else None)
    return {"total": total, "breakdown": rows, "per_task": per_task}


def audit_section_recognition(conn, task_id=None):
    """B. Section identification rate on chunks + conclusion reachability."""
    eu_filter = "AND eu.task_id = :task_id" if task_id else ""
    chunk_rows = _rows(conn, f"""
        SELECT pc.section, COUNT(DISTINCT pc.id) AS chunks, COUNT(DISTINCT pc.paper_id) AS papers
        FROM paper_chunks pc
        WHERE EXISTS (SELECT 1 FROM evidence_units eu
                      WHERE eu.paper_id = pc.paper_id {eu_filter})
        GROUP BY pc.section ORDER BY chunks DESC
    """, {"task_id": task_id} if task_id else None)
    total_chunks = sum(r["chunks"] for r in chunk_rows)
    unknown_chunks = sum(r["chunks"] for r in chunk_rows if r["section"] in (None, "", "unknown", "NULL"))

    ev_section_rows = _rows(conn, f"""
        SELECT COALESCE(eu.section, 'NULL') AS section, COUNT(*) AS n
        FROM evidence_units eu WHERE 1=1 {eu_filter}
        GROUP BY section ORDER BY n DESC
    """, {"task_id": task_id} if task_id else None)
    total_ev = sum(r["n"] for r in ev_section_rows)

    # Conclusion reachability: of full-text-extracted papers, how many have
    # at least one conclusion-section chunk / evidence unit.
    reach = _rows(conn, f"""
        WITH ft AS (
            SELECT DISTINCT tp.paper_id
            FROM task_papers tp
            WHERE tp.extraction_status = 'full_text'
              {'AND tp.task_id = :task_id' if task_id else ''}
        )
        SELECT
          (SELECT COUNT(*) FROM ft) AS full_text_papers,
          (SELECT COUNT(*) FROM ft JOIN paper_chunks pc ON pc.paper_id = ft.paper_id
             WHERE pc.section = 'conclusion') AS with_conclusion_chunk,
          (SELECT COUNT(*) FROM ft JOIN evidence_units eu ON eu.paper_id = ft.paper_id
             WHERE eu.section = 'conclusion') AS with_conclusion_evidence
    """, {"task_id": task_id} if task_id else None)
    reach = reach[0] if reach else {"full_text_papers": 0, "with_conclusion_chunk": 0, "with_conclusion_evidence": 0}

    return {
        "chunk_sections": chunk_rows,
        "total_chunks": total_chunks,
        "unknown_chunk_share": round(unknown_chunks / total_chunks, 4) if total_chunks else None,
        "evidence_sections": ev_section_rows,
        "total_evidence": total_ev,
        "conclusion_reachability": reach,
    }


def audit_evidence_types(conn, task_id=None):
    """C. Evidence type mix (what gap admission consumes)."""
    where = "WHERE eu.task_id = :task_id" if task_id else ""
    rows = _rows(conn, f"""
        SELECT eu.evidence_type, COUNT(*) AS n,
               SUM(CASE WHEN eu.verification_status IN ('verified','upgraded') THEN 1 ELSE 0 END) AS verified
        FROM evidence_units eu
        {where}
        GROUP BY eu.evidence_type ORDER BY n DESC
    """, {"task_id": task_id} if task_id else None)
    total = sum(r["n"] for r in rows)
    limitation_signal = sum(r["n"] for r in rows
                            if r["evidence_type"] in ("limitation", "negative_result", "future_work"))
    return {"total": total, "breakdown": rows,
            "limitation_signal_share": round(limitation_signal / total, 4) if total else None}


def audit_admission_rejections(conn, task_id=None):
    """D. Downstream symptom: gap mining admission reason codes.

    Codes are contextualized by gap_candidates.mining_policy_version —
    NO_LIMITATION_SIGNAL / UNRESOLVED_VERIFIED_CONTRADICTION only exist in
    pre-v7 policies (v7 moved the limitation requirement to per-type
    contracts), so their counts describe the v6 era, not current behavior.
    """
    where = "WHERE t.task_id = :task_id" if task_id else ""
    traces = _rows(conn, f"""
        SELECT t.output_json FROM agent_traces t
        WHERE t.step_name = 'gap_mining_admission' AND t.output_json IS NOT NULL
        {where}
    """, {"task_id": task_id} if task_id else None)
    codes = Counter()
    examined = 0
    for t in traces:
        try:
            payload = json.loads(t["output_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        examined += 1
        for qr in payload.get("question_results", []):
            for code in qr.get("reason_codes", []) or []:
                codes[code] += 1
    where_gc = "WHERE gc.task_id = :task_id" if task_id else ""
    policy_rows = _rows(conn, f"""
        SELECT COALESCE(gc.mining_policy_version, 'NULL') AS policy, COUNT(*) AS n
        FROM gap_candidates gc
        {where_gc}
        GROUP BY 1 ORDER BY n DESC
    """, {"task_id": task_id} if task_id else None)
    return {
        "traces_examined": examined,
        "reason_codes": dict(codes.most_common()),
        "mining_policy_versions": policy_rows,
    }


def print_report(report):
    print("=" * 72)
    print("EXTRACTION SUPPLY AUDIT" + (f"  (task {report['task_id']})" if report.get("task_id") else "  (all tasks)"))
    print("=" * 72)

    s = report["extraction_status"]
    print(f"\n[A] Candidate paper reading funnel — {s['total']} task_papers rows")
    print("    NOTE: extraction_status only covers tasks run after the 0034 migration;")
    print("          NULL rows are historical tasks — see [A2] for the all-history view.")
    for r in s["breakdown"]:
        print(f"    {_pct(r['n'], s['total'])}  {r['n']:6d}  {r['status']:<14} reason={r['reason']}")
    print("    per-task (top 30):")
    for t in s["per_task"]:
        ft, ao, na, tot = t["full_text"] or 0, t["abstract_only"] or 0, t["not_assessed"] or 0, t["total"]
        print(f"      {t['task_id'][:12]}  full_text {ft:4d} ({100*ft/tot:4.1f}%)  "
              f"abstract_only {ao:4d}  not_assessed {na:4d}  / {tot}")

    rm = report["reading_method"]
    print(f"\n[A2] Reading funnel (all history, by evidence extraction_method) — "
          f"{rm['total_units']} units")
    for r in rm["breakdown"]:
        print(f"    {_pct(r['units'], rm['total_units'])}  {r['units']:6d} units / "
              f"{r['papers']:5d} papers / {r['tasks']:4d} tasks  {r['method']}")
    print(f"    gap-mining tasks with conclusion-section evidence: "
          f"{rm['gap_mining_tasks_with_conclusion_evidence']}/{rm['gap_mining_tasks']}")

    rec = report["section_reconciliation"]
    print(f"\n[B2] Cross-pipeline section reconciliation (extraction view vs RAG view)")
    print(f"    papers with both evidence and chunks: {rec.get('papers_with_both_evidence_and_chunks', 0)}")
    print(f"    conclusion: evidence-only {rec.get('ev_only', 0)}  "
          f"chunk-only {rec.get('chunk_only', 0)}  both {rec.get('both_conclusion', 0)}  "
          f"(evidence view {rec.get('ev_conclusion_papers', 0)} / RAG view {rec.get('chunk_conclusion_papers', 0)})")
    print("    The two pipelines use different section classifiers; neither view")
    print("    describes the other. Gap admission consumes the EVIDENCE view.")

    sec = report["section_recognition"]
    print(f"\n[B] Section recognition (RAG view — paper_chunks, NOT what extraction saw)")
    print(f"    {sec['total_chunks']} chunks (evidence-linked papers)")
    print(f"    unknown-section share: "
          f"{sec['unknown_chunk_share']*100:.1f}%" if sec["unknown_chunk_share"] is not None else "    no chunks")
    for r in sec["chunk_sections"][:12]:
        print(f"    {_pct(r['chunks'], sec['total_chunks'])}  {r['chunks']:6d} chunks / {r['papers']:4d} papers  section={r['section']}")
    r = sec["conclusion_reachability"]
    ft = r["full_text_papers"]
    print(f"    conclusion reachability on {ft} full_text papers:")
    print(f"      with conclusion chunk:    {r['with_conclusion_chunk']:5d}  ({100.0*r['with_conclusion_chunk']/ft:.1f}%)" if ft else "      no full_text papers")
    if ft:
        print(f"      with conclusion evidence: {r['with_conclusion_evidence']:5d}  ({100.0*r['with_conclusion_evidence']/ft:.1f}%)")

    et = report["evidence_types"]
    print(f"\n[C] Evidence type mix — {et['total']} evidence units")
    if et["limitation_signal_share"] is not None:
        print(f"    limitation-signal share (limitation/negative_result/future_work): "
              f"{et['limitation_signal_share']*100:.1f}%")
    for r in et["breakdown"][:12]:
        print(f"    {_pct(r['n'], et['total'])}  {r['n']:6d}  {r['evidence_type']:<20} verified={r['verified']}")

    ar = report["admission_rejections"]
    print(f"\n[D] Gap mining admission reason codes — {ar['traces_examined']} traces examined")
    for code, n in ar["reason_codes"].items():
        print(f"    {n:6d}  {code}")
    if not ar["reason_codes"]:
        print("    (none)")
    if ar.get("mining_policy_versions"):
        print("    gap_candidates by mining_policy_version:")
        for r in ar["mining_policy_versions"]:
            print(f"      {r['n']:6d}  {r['policy']}")


def main():
    parser = argparse.ArgumentParser(description="Evidence supply funnel audit (read-only)")
    parser.add_argument("--db", default="sqlite:///./deep_research.db")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--json", default=None, help="also write the report as JSON to this path")
    args = parser.parse_args()

    engine = create_engine(args.db)
    with engine.connect() as conn:
        report = {
            "task_id": args.task_id,
            "extraction_status": audit_extraction_status(conn, args.task_id),
            "reading_method": audit_reading_method(conn, args.task_id),
            "section_reconciliation": audit_section_reconciliation(conn, args.task_id),
            "section_recognition": audit_section_recognition(conn, args.task_id),
            "evidence_types": audit_evidence_types(conn, args.task_id),
            "admission_rejections": audit_admission_rejections(conn, args.task_id),
        }
    print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()

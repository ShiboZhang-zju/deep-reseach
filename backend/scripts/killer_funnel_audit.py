"""Killer-funnel audit (read-only): where do prior-art killers get lost?

Question
--------
The fixed-budget pilot's independent reviewer named, for each gap, the papers
that would falsify it (`key_prior_work`). When those papers exist in the task's
own corpus but the audit never saw them, the verdict "confirmed" is a statement
about the pool, not about the literature. This script measures that gap for a
finished adjudication directory.

Funnel (for reviewer-cited killers whose title matches a corpus paper >= 0.9):

    1_selected_neighbor ................. shown to the audit as a neighbour
    2_screened_not_selected ............. gap-relevance scored, not in Top-K
    3_in_audit_pool_capped_before_screen  in the audit pool, cut by the cap
    4_in_task_corpus_never_pooled ....... in the corpus, never admitted at all

Bucket 4 is the structural loss: the pool is built from the gap's adversarial
search plus the evidence-linked papers of its own questions, so a killer that
was never linked to those questions could not be reached. Bucket 3 is the
ordering loss: the cap ranked by TaskPaper.final_score (task relevance) before
any gap-specific screen.

Reported alongside: `novelty_confidence` separation (AUC of PARTIAL > FULL) and
the claimed_delta shape split (measurement-lead vs mechanism vocabulary), which
is what motivated the operationalization-only gate candidate.

Usage
-----
    python scripts/killer_funnel_audit.py \
        --dir eval_results/fixed_budget_pilot \
        --out eval_results/killer_funnel_2026-11-02.json

Read-only: opens the DB with a plain connection and never writes. Missing
inputs degrade to a stated reason rather than a fabricated zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import statistics
import sys

TITLE_MATCH = 0.90
TITLE_WEAK = 0.72


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_verdicts(directory: str) -> dict[str, dict]:
    return {row["anon_id"]: row
            for row in _load_jsonl(os.path.join(directory, "verdicts.jsonl"))}


def _load_mapping(directory: str) -> dict[str, dict]:
    path = os.path.join(directory, "review_mapping.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["units"]


def build_report(db_path: str, directory: str) -> dict:
    verdicts = _load_verdicts(directory)
    units = _load_mapping(directory)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    funnel = collections.Counter()
    conf_rows: list[tuple[str, float | None]] = []
    shape_rows: list[tuple[str, bool]] = []
    no_audit: list[str] = []
    gaps: list[dict] = []

    measurement_lead = re.compile(
        r"^\W*(?:provide|quantify|measure|characteri[sz]e|estimate|report|map|"
        r"extend|establish|offer|deliver|present|evaluate|benchmark)\b", re.I)
    mechanism_vocab = re.compile(
        r"\b(?:mechanism|causal|because|why|attribut\w*|mediate\w*|drives?|"
        r"govern\w*|underlying|explain\w*|intervention|procedure|protocol)\b", re.I)

    for anon_id, meta in units.items():
        if meta.get("unit_type") != "gap":
            continue
        gap_id = meta["real_id"]
        verdict = verdicts.get(anon_id, {}).get("verdict") or ""
        gap = conn.execute(
            "select id, gap_type, status, claimed_delta, question_ids_json "
            "from gap_candidates where id=?", (gap_id,)).fetchone()
        if gap is None:
            no_audit.append(anon_id)
            continue
        audit = conn.execute(
            "select task_id, audit_result, novelty_confidence, "
            "neighbor_paper_ids_json, search_query_ids_json "
            "from gap_audits where gap_id=? order by created_at desc",
            (gap_id,)).fetchone()
        if audit is None:
            no_audit.append(anon_id)
            continue

        neighbors = set(json.loads(audit["neighbor_paper_ids_json"] or "[]"))
        query_ids = json.loads(audit["search_query_ids_json"] or "[]")
        retrieved = set()
        if query_ids:
            placeholders = ",".join("?" * len(query_ids))
            retrieved = {r["paper_id"] for r in conn.execute(
                f"select distinct paper_id from search_query_papers "
                f"where query_id in ({placeholders})", query_ids)}
        screened = {r["paper_id"] for r in conn.execute(
            "select paper_id from gap_paper_relevance where gap_id=?", (gap_id,))}
        corpus = conn.execute(
            "select p.id, p.title from task_papers tp "
            "join papers p on p.id = tp.paper_id where tp.task_id=?",
            (audit["task_id"],)).fetchall()

        retrieved_items = verdicts.get(anon_id, {}).get("retrieved") or []
        cited = []
        for idx in verdicts.get(anon_id, {}).get("key_prior_work") or []:
            try:
                cited.append(retrieved_items[int(idx) - 1])
            except (ValueError, IndexError):
                continue
        for item in cited:
            title = item.get("title") or ""
            best, best_sim = None, 0.0
            for row in corpus:
                sim = _jaccard(title, row["title"])
                if sim > best_sim:
                    best, best_sim = row, sim
            if best_sim < TITLE_WEAK:
                funnel["absent_from_corpus"] += 1
                continue
            if best_sim < TITLE_MATCH:
                funnel["corpus_weakmatch"] += 1
                continue
            pid = best["id"]
            if pid in neighbors:
                funnel["1_selected_neighbor"] += 1
            elif pid in screened:
                funnel["2_screened_not_selected"] += 1
            elif pid in retrieved:
                funnel["3_in_audit_pool_capped_before_screen"] += 1
            else:
                funnel["4_in_task_corpus_never_pooled"] += 1

        conf_rows.append((verdict, audit["novelty_confidence"]))
        gap_text = gap["claimed_delta"] or ""
        shape_rows.append((verdict,
                           bool(measurement_lead.search(gap_text))
                           and not mechanism_vocab.search(gap_text)))
        gaps.append({
            "anon_id": anon_id, "arm": meta.get("arm"),
            "gap_type": gap["gap_type"], "verdict": verdict,
            "status": gap["status"], "audit_result": audit["audit_result"],
            "novelty_confidence": audit["novelty_confidence"],
            "neighbor_count": len(neighbors),
            "cited_killers": len(cited),
        })

    return {
        "directory": directory,
        "db": db_path,
        "gaps_reviewed": len(gaps),
        "gaps_without_audit_row": no_audit,
        "killer_funnel": dict(funnel),
        "killer_funnel_shares": {
            key: round(value / max(sum(funnel.values()), 1), 4)
            for key, value in funnel.items()},
        "novelty_confidence": _confidence_summary(conf_rows),
        "claimed_delta_shape": _shape_summary(shape_rows),
        "neighbor_count_distribution": dict(collections.Counter(
            g["neighbor_count"] for g in gaps)),
        "gaps": gaps,
    }


def _confidence_summary(rows: list[tuple[str, float | None]]) -> dict:
    scored = [(v, c) for v, c in rows if c is not None]
    if not scored:
        return {"n": 0, "note": "no novelty_confidence recorded"}
    by_verdict = collections.defaultdict(list)
    for verdict, conf in scored:
        by_verdict[verdict].append(conf)
    partial = by_verdict.get("PARTIAL", [])
    full = by_verdict.get("FULL", [])
    auc = None
    if partial and full:
        wins = sum(1 for p in partial for f in full if p > f)
        ties = sum(1 for p in partial for f in full if p == f)
        auc = round((wins + 0.5 * ties) / (len(partial) * len(full)), 4)
    return {
        "n": len(scored),
        "median": round(statistics.median(c for _, c in scored), 4),
        "by_verdict": {k: {"n": len(v), "median": round(statistics.median(v), 4)}
                       for k, v in by_verdict.items()},
        # 0.5 means novelty_confidence cannot separate covered from uncovered.
        "auc_partial_gt_full": auc,
    }


def _shape_summary(rows: list[tuple[str, bool]]) -> dict:
    table = collections.defaultdict(collections.Counter)
    for verdict, is_measurement in rows:
        table["measurement_lead" if is_measurement else "other"][verdict] += 1
    out = {}
    for shape, counter in table.items():
        total = sum(counter.values())
        out[shape] = {
            "n": total,
            "verdicts": dict(counter),
            "full_rate": round(counter.get("FULL", 0) / total, 4) if total else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="eval_results/fixed_budget_pilot",
                        help="adjudication output directory")
    parser.add_argument("--db", default="deep_research.db")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    report = build_report(args.db, args.dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)

    print(f"gaps reviewed: {report['gaps_reviewed']}")
    funnel = report["killer_funnel"]
    total = max(sum(funnel.values()), 1)
    for key in sorted(funnel):
        print(f"  {key}: {funnel[key]} ({funnel[key] / total * 100:.1f}%)")
    conf = report["novelty_confidence"]
    print(f"novelty_confidence AUC(PARTIAL>FULL): {conf.get('auc_partial_gt_full')}"
          f"  (0.5 = no signal)")
    for shape, stats in report["claimed_delta_shape"].items():
        print(f"  shape {shape}: n={stats['n']} full_rate={stats['full_rate']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

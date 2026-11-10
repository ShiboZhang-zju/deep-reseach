"""v7 replay pilot — blind prior-art review tool (q2/q4 human columns).

The pilot gate is frozen (replay_report.json):
    GO iff >=1 v7-only opportunity is human-true AND yields a credible idea
    AND new opportunities are not mostly false-open.

`opportunity_replay.py` writes `v7_only_opportunities.jsonl` with EMPTY
`human_prior_art_verdict` fields — the blind audit input. This tool turns
that input into something a human can actually adjudicate, and aggregates
the verdicts back:

    sheet   Generate the blind review material:
            - human_review_blind.md      one anonymized block per opportunity
                                         (claimed_delta + supporting evidence
                                         + candidate prior-art papers pulled
                                         from gap_paper_relevance, falling
                                         back to neighbor_comparisons)
            - human_verdicts.jsonl       template keyed by review_id
            - review_mapping.json        review_id -> gap_id/topic_id
                                         (de-anonymization happens ONLY at
                                         scoring time; do not read while
                                         reviewing)

    score   Read the filled human_verdicts.jsonl, de-anonymize, write
            v7_only_opportunities.adjudicated.jsonl, and compute:
            - q2 marginal opportunity precision (true_gap NONE / adjudicated;
              false_open = FULL; PARTIAL reported separately)
            - q4 per-type human columns
            - the pilot-gate conditions computable from prior-art verdicts
              (>=1 human-true; not mostly false-open). The credible-idea
              condition stays PENDING — it requires linking the replay's
              ideas to these gaps, which this tool does not attempt.

Statistical discipline (production_e2e frozen rules, README §18):
    - Un-adjudicated is NOT a negative result: only rows with a non-empty
      verdict enter the denominator.
    - review_coverage is always printed; partial coverage is flagged
      INCOMPLETE in the header.
    - Candidate prior art is RECALL-oriented: audit-recalled candidates can
      be noisy (historically ~39% share zero literal tokens with the claim),
      so the noise disclosure is inlined in the sheet. Do not filter.

Blinding: the sheet shows NO survival_basis, audit_result, novelty scores,
status, or v6-similarity — the reviewer must not know how the system itself
judged the opportunity. Order is shuffled with a fixed seed so row order
cannot leak status (source rows are grouped by topic and status varies).

Usage:
    cd backend
    python -m eval.production_e2e.v7_blind_review sheet \
        --pilot-dir ../eval_results/v7_replay_pilot
    # ... fill human_verdicts.jsonl ...
    python -m eval.production_e2e.v7_blind_review score \
        --pilot-dir ../eval_results/v7_replay_pilot
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SHUFFLE_SEED = 20260921  # pre-registered pilot date; keep stable or IDs break
VERDICTS = {"FULL", "PARTIAL", "NONE"}

MAX_EVIDENCE_PER_GAP = 4
MAX_CANDIDATES_PER_GAP = 5


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_topic_names() -> dict[str, str]:
    topics_path = Path(__file__).resolve().parent / "topics.jsonl"
    return {t["topic_id"]: t["topic"] for t in _read_jsonl(topics_path)}


def _fetch_gap_context(db, gap_ids: list[str]) -> dict[str, dict]:
    """Evidence claims + candidate prior art per gap, straight from the DB."""
    from sqlalchemy import text as sql_text

    context: dict[str, dict] = {}
    for gap_id in gap_ids:
        evidence = [dict(r._mapping) for r in db.execute(sql_text("""
            SELECT eu.normalized_claim, eu.evidence_type, eu.section,
                   p.title AS paper_title, p.year
            FROM gap_evidence_links gel
            JOIN evidence_units eu ON eu.id = gel.evidence_id
            JOIN papers p ON p.id = eu.paper_id
            WHERE gel.gap_id = :gap_id
            ORDER BY gel.relevance_score DESC
            LIMIT :lim
        """), {"gap_id": gap_id, "lim": MAX_EVIDENCE_PER_GAP})]

        candidates = [dict(r._mapping) for r in db.execute(sql_text("""
            SELECT p.title, p.year, p.venue, gpr.relevance_score,
                   gpr.problem_overlap, gpr.mechanism_overlap, gpr.claim_overlap
            FROM gap_paper_relevance gpr
            JOIN papers p ON p.id = gpr.paper_id
            WHERE gpr.gap_id = :gap_id
            ORDER BY gpr.relevance_score DESC
            LIMIT :lim
        """), {"gap_id": gap_id, "lim": MAX_CANDIDATES_PER_GAP})]
        if not candidates:
            # Older runs predate gap_paper_relevance — the audit's neighbor
            # comparisons carry the same role (recall-oriented candidate list).
            candidates = [dict(r._mapping) for r in db.execute(sql_text("""
                SELECT p.title, p.year, p.venue, nc.overlap_ratio AS relevance_score,
                        nc.why_not_closed,
                        nc.covered_claims_json, nc.uncovered_claims_json
                FROM neighbor_comparisons nc
                JOIN papers p ON p.id = nc.paper_id
                WHERE nc.gap_id = :gap_id
                ORDER BY nc.overlap_ratio DESC
                LIMIT :lim
            """), {"gap_id": gap_id, "lim": MAX_CANDIDATES_PER_GAP})]

        context[gap_id] = {"evidence": evidence, "candidates": candidates}
    return context


def build_sheet(pilot_dir: Path) -> None:
    from app.db.session import SessionLocal

    rows = _read_jsonl(pilot_dir / "v7_only_opportunities.jsonl")
    if not rows:
        raise SystemExit("v7_only_opportunities.jsonl is empty — run the replay first")

    # Deterministic shuffle + anonymized ids. Every field that could reveal
    # the system's own judgment is dropped here.
    order = list(range(len(rows)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    mapping: dict[str, dict] = {}
    anon: list[tuple[str, dict]] = []
    for idx, row_idx in enumerate(order, start=1):
        review_id = f"v7r-{idx:03d}"
        source = rows[row_idx]
        mapping[review_id] = {
            "gap_id": source["gap_id"], "topic_id": source["topic_id"],
            "gap_type": source["gap_type"],
        }
        anon.append((review_id, source))

    topic_names = _load_topic_names()
    db = SessionLocal()
    try:
        context = _fetch_gap_context(db, [r["gap_id"] for r in rows])
    finally:
        db.close()

    lines: list[str] = [
        "# v7 Replay Pilot — Blind Prior-Art Review",
        "",
        f"{len(rows)} v7-only opportunities (claimed discoveries with no v6 counterpart).",
        "",
        "For EACH block below, judge whether existing prior art already covers the",
        "claimed delta. You are given the claim, its supporting evidence, and a",
        "recall-oriented candidate prior-art list retrieved during the audit.",
        "",
        "Verdict semantics (fill human_verdicts.jsonl, keyed by review_id):",
        "- FULL    — a candidate (or other work you know) already covers the claimed",
        "            delta completely  -> false-open gap",
        "- PARTIAL — prior art covers part of the delta; a narrower residual remains",
        "- NONE    — no coverage you can identify  -> true gap",
        "",
        "Candidate-list disclosure: recall is prioritized over precision. Candidate",
        "papers historically share zero literal tokens with the claim ~39% of the",
        "time and are NOT pre-filtered — a weak-looking candidate list is noise,",
        "not evidence of novelty. You may (and should) bring in any prior art you",
        "know yourself; the candidates are a starting point, not a boundary.",
        "",
        "The blocks are anonymized and shuffled. Do not open review_mapping.json",
        "until scoring.",
        "",
        "---",
        "",
    ]
    for review_id, row in anon:
        ctx = context[row["gap_id"]]
        lines.append(f"## {review_id}  [{row['gap_type']}]")
        lines.append("")
        lines.append(f"Topic: {topic_names.get(row['topic_id'], row['topic_id'])}")
        lines.append("")
        lines.append(f"Claimed delta: {row['claimed_delta']}")
        lines.append("")
        if ctx["evidence"]:
            lines.append("Supporting evidence (as extracted from the literature):")
            for ev in ctx["evidence"]:
                year = f" ({ev['year']})" if ev["year"] else ""
                lines.append(f"- [{ev['evidence_type']}/{ev['section'] or 'n/a'}] "
                             f"{ev['normalized_claim']}{year}")
            lines.append("")
        else:
            lines.append("Supporting evidence: (none linked — note this in your verdict)")
            lines.append("")
        if ctx["candidates"]:
            lines.append("Candidate prior art (audit-recalled, unfiltered):")
            for cand in ctx["candidates"]:
                year = f" ({cand['year']})" if cand["year"] else ""
                venue = f" — {cand['venue']}" if cand.get("venue") else ""
                lines.append(f"- {cand['title']}{year}{venue}")
                if "problem_overlap" in cand:
                    lines.append(f"    overlaps: problem={cand['problem_overlap']} "
                                 f"mechanism={cand['mechanism_overlap']} "
                                 f"claim={cand['claim_overlap']}")
                if cand.get("covered_claims_json"):
                    try:
                        covered = json.loads(cand["covered_claims_json"])
                        if covered:
                            lines.append(f"    audit says it covers: {covered[0][:200]}")
                    except (json.JSONDecodeError, TypeError):
                        pass
        else:
            lines.append("Candidate prior art: (audit recalled none — judge from your "
                         "own knowledge of the literature)")
        lines.append("")

    (pilot_dir / "human_review_blind.md").write_text("\n".join(lines), encoding="utf-8")
    _write_jsonl(pilot_dir / "human_verdicts.jsonl", [
        {"review_id": rid, "human_prior_art_verdict": "", "human_notes": ""}
        for rid, _ in anon
    ])
    (pilot_dir / "review_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[v7_blind_review] sheet written to {pilot_dir / 'human_review_blind.md'}")
    print(f"[v7_blind_review] {len(rows)} blocks; fill human_verdicts.jsonl "
          f"(verdict: FULL / PARTIAL / NONE), then run the score subcommand")


def _link_ideas_for_gaps(gap_ids: list[str]) -> dict[str, list[dict]]:
    """Ideas linked to each human-true gap (condition [3] of the pilot gate).

    Credible idea = decision 'executable_candidate' (hard gates passed,
    assigned by code gate, not by score). idea_status is disclosed because
    replay re-runs supersede earlier generations: a superseded generation
    still proves the gap CAN yield a gated idea, but the active generation
    is what the task currently presents.
    """
    if not gap_ids:
        return {}
    from sqlalchemy import text as sql_text

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        linked: dict[str, list[dict]] = {gid: [] for gid in gap_ids}
        for gid in gap_ids:
            linked[gid] = [dict(r._mapping) for r in db.execute(sql_text("""
                SELECT decision, idea_status, final_score, title
                FROM research_ideas WHERE gap_id = :gid
                ORDER BY final_score DESC NULLS LAST
            """), {"gid": gid})]
        return linked
    finally:
        db.close()


def build_score(pilot_dir: Path) -> None:
    rows = _read_jsonl(pilot_dir / "v7_only_opportunities.jsonl")
    mapping = json.loads((pilot_dir / "review_mapping.json").read_text(encoding="utf-8"))
    verdicts = _read_jsonl(pilot_dir / "human_verdicts.jsonl")

    by_review = {}
    for v in verdicts:
        verdict = str(v.get("human_prior_art_verdict", "")).strip().upper()
        by_review[v["review_id"]] = {**v, "_verdict": verdict}

    reverse = {m["gap_id"]: rid for rid, m in mapping.items()}

    adjudicated = true_gap = false_open = partial = invalid = 0
    per_type: dict[str, dict] = {}
    true_gap_ids: list[str] = []
    out_rows: list[dict] = []
    for row in rows:
        rid = reverse.get(row["gap_id"])
        v = by_review.get(rid, {})
        verdict = v.get("_verdict", "")
        out = dict(row)
        out["human_prior_art_verdict"] = verdict
        out["human_notes"] = v.get("human_notes", "")
        out_rows.append(out)
        if verdict == "":
            continue
        if verdict not in VERDICTS:
            invalid += 1
            print(f"[v7_blind_review] WARNING: {rid} has invalid verdict "
                  f"'{verdict}' (expected FULL/PARTIAL/NONE) — excluded")
            continue
        adjudicated += 1
        bucket = per_type.setdefault(row["gap_type"],
                                     {"adjudicated": 0, "true_gap": 0,
                                      "false_open": 0, "partial": 0})
        bucket["adjudicated"] += 1
        if verdict == "NONE":
            true_gap += 1
            true_gap_ids.append(row["gap_id"])
            bucket["true_gap"] += 1
        elif verdict == "FULL":
            false_open += 1
            bucket["false_open"] += 1
        else:
            partial += 1
            bucket["partial"] += 1

    _write_jsonl(pilot_dir / "v7_only_opportunities.adjudicated.jsonl", out_rows)

    total = len(rows)
    judged = adjudicated
    coverage = f"{judged}/{total}"
    print()
    print("=" * 72)
    print("v7 REPLAY PILOT — Q2 MARGINAL OPPORTUNITY PRECISION")
    print(f"Human review: {coverage} judged"
          + ("" if judged == total else "  <-- INCOMPLETE"))
    print("=" * 72)
    if judged == 0:
        print("pending — no verdicts filled yet (empty verdict is NOT a negative "
              "result; only judged rows count)")
        return
    print(f"  adjudicated      {judged}")
    print(f"  true_gap (NONE)  {true_gap}   -> marginal precision "
          f"{100.0 * true_gap / judged:.1f}%")
    print(f"  partial          {partial}")
    print(f"  false_open (FULL){false_open}  -> false-open share "
          f"{100.0 * false_open / judged:.1f}%")
    if invalid:
        print(f"  invalid verdicts excluded: {invalid}")
    print()
    print("Per gap_type (adjudicated only):")
    for gap_type in sorted(per_type):
        b = per_type[gap_type]
        print(f"  {gap_type:<22} adjudicated={b['adjudicated']:2d} "
              f"true={b['true_gap']} partial={b['partial']} false_open={b['false_open']}")
    print()
    print("Pilot gate (pre-registered):")
    cond_true = true_gap >= 1
    cond_not_false = false_open < true_gap + partial or false_open == 0
    print(f"  [1] >=1 v7-only opportunity human-true (NONE): "
          f"{'MET' if cond_true else 'NOT MET'} ({true_gap})")
    print(f"  [2] new opportunities not mostly false-open: "
          f"{'MET' if false_open <= judged - false_open else 'NOT MET'} "
          f"(false_open {false_open} vs non-false {judged - false_open})")

    # Condition [3]: link ideas to the human-true gaps.
    credible_gaps = 0
    try:
        ideas_by_gap = _link_ideas_for_gaps(true_gap_ids)
    except Exception as exc:  # DB unavailable — keep [3] honestly pending
        print(f"  [3] idea linkage unavailable ({exc}) — still PENDING")
        ideas_by_gap = None
    if ideas_by_gap is not None:
        print(f"  [3] >=1 human-true yields a credible idea "
              f"(decision=executable_candidate):")
        for gid in true_gap_ids:
            ideas = ideas_by_gap.get(gid, [])
            credible = [i for i in ideas if i["decision"] == "executable_candidate"]
            if credible:
                credible_gaps += 1
            n_super = sum(1 for i in ideas if i["idea_status"] == "superseded")
            print(f"      gap {gid[:8]}: {len(ideas)} ideas linked, "
                  f"{len(credible)} executable_candidate"
                  + (f" ({n_super} superseded by later generations)" if ideas else ""))
            for i in credible[:3]:
                score = f" score={i['final_score']}" if i["final_score"] is not None else ""
                print(f"        - [{i['idea_status']}] {i['title'][:80]}{score}")
        print(f"      -> {'MET' if credible_gaps >= 1 else 'NOT MET'} "
              f"({credible_gaps} human-true gap(s) with gated ideas)")
    print()
    print("Verdicts written back to "
          f"{pilot_dir / 'v7_only_opportunities.adjudicated.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v7 replay blind prior-art review")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("sheet", "score"):
        p = sub.add_parser(name)
        p.add_argument("--pilot-dir", default="../eval_results/v7_replay_pilot",
                       help="directory containing v7_only_opportunities.jsonl")
    args = parser.parse_args()
    pilot_dir = Path(args.pilot_dir)
    if not (pilot_dir / "v7_only_opportunities.jsonl").exists():
        raise SystemExit(f"no v7_only_opportunities.jsonl under {pilot_dir}")
    if args.cmd == "sheet":
        build_sheet(pilot_dir)
    else:
        build_score(pilot_dir)


if __name__ == "__main__":
    main()

"""§8.1 anonymised export for the fixed-budget pilot adjudication.

Produces two artefacts and NOTHING the reviewer can deanonymise:

  adjudication_input.jsonl  — one record per unit, arm identity stripped,
                             citation evidence redacted of paper identity,
                             shuffled (seeded), sequential anonymous IDs.
  review_mapping.json       — anon_id -> (topic, arm, real id). SEALED: the
                             adjudication must not open it.

Why paper titles/authors are stripped from the evidence block
-------------------------------------------------------------
Design §8.1 accepts ONE known leak: gap_type reveals the arm, because the v6
control arm can only produce two types. That leak is intrinsic to the
experiment and is disclosed. A second, avoidable leak would be citing paper
titles: the two arms share the same topics, so a reviewer who recognises
"paper X is in arm B's pool" can infer the arm from the citation. Evidence is
therefore rendered as spans + claim + type, with no paper title, no author, no
year, no arXiv id (those are exactly what a reviewer would search).

What the reviewer must do with an evidence claim
------------------------------------------------
Prior-art checking for novelty needs the CLAIM (what is asserted), not the
paper's identity. If the reviewer cannot verify a claim without the paper
identity, the record says so explicitly via `evidence_identity_redacted: true`
rather than silently degrading the check.

Usage:
    python scripts/export_adjudication.py            # write artefacts
    python scripts/export_adjudication.py --dry-run  # counts only
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from collections import Counter

SEED = 20261102  # fixed in advance; the shuffle must be reproducible
OUT_DIR = r"eval_results\fixed_budget_pilot"
MATRIX = os.path.join(OUT_DIR, "run_matrix.json")

# Units entering adjudication = everything that reached an audit + every idea.
# `candidate` gaps were never audited (no verdict exists), so the pilot's own
# framing does not include them; they stay in the secondary-metric report.
AUDITED_STATUSES = {"surviving", "confirmed_runner_up", "inconclusive",
                    "rejected"}
MAX_EVIDENCE_PER_UNIT = 6


def _short_hash(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _load_matrix() -> dict:
    with open(MATRIX, encoding="utf-8") as fh:
        return json.load(fh)


def _evidence_block(con, gap_id):
    """Redacted evidence: claim + span + type, no paper identity."""
    rows = con.execute("""
        SELECT eu.evidence_type, eu.normalized_claim, eu.original_span,
               eu.verification_status, gel.relation_type
        FROM gap_evidence_links gel
        JOIN evidence_units eu ON eu.id = gel.evidence_id
        WHERE gel.gap_id = ?
        ORDER BY gel.relevance_score DESC
        LIMIT ?
    """, (gap_id, MAX_EVIDENCE_PER_UNIT)).fetchall()
    out = []
    for etype, claim, span, vstatus, rel in rows:
        out.append({
            "evidence_type": etype,
            "claim": (claim or "").strip(),
            "span": (span or "").strip()[:400] or None,
            "verification_status": vstatus,
            "link_relation": rel,
        })
    return out


def _gap_unit(con, gap_row, topic_id, stratum):
    (gid, gtype, description, target_setting, observed_problem,
     existing_coverage, missing_capability, claimed_delta,
     testable_hypothesis, falsification_condition, provenance,
     residual_gap) = gap_row

    audit = con.execute("""
        SELECT audit_result, recommended_action, novelty_confidence,
               nearest_neighbor_summary, differentiation_summary,
               remaining_delta, search_admission_status
        FROM gap_audits WHERE gap_id=? ORDER BY created_at DESC LIMIT 1
    """, (gid,)).fetchone()

    return {
        "unit_type": "gap",
        "anon_id": None,  # assigned after shuffle
        "topic_stratum": stratum,
        "gap_type": gtype,
        "description": description,
        "target_setting": target_setting,
        "observed_problem": observed_problem,
        "existing_coverage": existing_coverage,
        "missing_capability": missing_capability,
        "claimed_delta": claimed_delta,
        "testable_hypothesis": testable_hypothesis,
        "falsification_condition": falsification_condition,
        "residual_gap": residual_gap,
        "provenance_status": provenance,
        "evidence": _evidence_block(con, gid),
        "evidence_identity_redacted": True,
        "audit": None if not audit else {
            "audit_result": audit[0],
            "recommended_action": audit[1],
            "novelty_confidence": audit[2],
            "nearest_neighbor_summary": audit[3],
            "differentiation_summary": audit[4],
            "remaining_delta": audit[5],
            "search_admission_status": audit[6],
        },
        "_topic_id": topic_id,
        "_gap_id": gid,
    }


def _idea_unit(con, idea_row, topic_id, stratum):
    (iid, title, description, motivation, method_sketch,
     expected_contribution, final_score, decision, tier) = idea_row

    return {
        "unit_type": "idea",
        "anon_id": None,
        "topic_stratum": stratum,
        "title": title,
        "description": description,
        "motivation": motivation,
        "method_sketch": method_sketch,
        "expected_contribution": expected_contribution,
        # decision/tier are the PIPELINE's own verdict — deliberately kept,
        # because the reviewer is asked whether the Idea is worth doing, not
        # whether the pipeline liked it. Including them lets the adjudication
        # also report how often the pipeline's own judgement was right.
        "pipeline_decision": decision,
        "pipeline_confidence_tier": tier,
        "pipeline_final_score": final_score,
        "_topic_id": topic_id,
        "_idea_id": iid,
    }


def collect(con, matrix):
    gaps, ideas = [], []
    for run in matrix["run_order"]:
        topic_id, stratum = run["topic_id"], run["stratum"]
        for arm in ("A", "B"):
            tid = run["task_ids"].get(arm)
            if not tid:
                continue
            placeholders = ",".join("?" * len(AUDITED_STATUSES))
            for row in con.execute(f"""
                SELECT id, gap_type, description, target_setting,
                       observed_problem, existing_coverage, missing_capability,
                       claimed_delta, testable_hypothesis,
                       falsification_condition, provenance_status, residual_gap
                FROM gap_candidates
                WHERE task_id=? AND status IN ({placeholders})
                ORDER BY created_at
            """, (tid, *sorted(AUDITED_STATUSES))).fetchall():
                unit = _gap_unit(con, row, topic_id, stratum)
                unit["_arm"] = arm
                gaps.append(unit)

            for row in con.execute("""
                SELECT id, title, description, motivation, method_sketch,
                       expected_contribution, final_score, decision,
                       confidence_tier
                FROM research_ideas
                WHERE task_id=? AND idea_status='active'
                ORDER BY created_at
            """, (tid,)).fetchall():
                unit = _idea_unit(con, row, topic_id, stratum)
                unit["_arm"] = arm
                ideas.append(unit)
    return gaps, ideas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = _load_matrix()
    con = sqlite3.connect("deep_research.db")
    try:
        gaps, ideas = collect(con, matrix)
    finally:
        con.close()

    units = gaps + ideas
    print(f"collected: {len(gaps)} gap units, {len(ideas)} idea units "
          f"({len(units)} total)")

    print("\nper topic × arm (gap units / idea units):")
    tally = Counter((u["_topic_id"], u["_arm"], u["unit_type"]) for u in units)
    for run in matrix["run_order"]:
        row = []
        for arm in ("A", "B"):
            row.append(f"{arm}: {tally[(run['topic_id'], arm, 'gap')]}g/"
                       f"{tally[(run['topic_id'], arm, 'idea')]}i")
        print(f"  {run['topic_id']}: " + "   ".join(row))

    # Deterministic shuffle: seed fixed above, order independent of collection
    # order, so a re-run of the export reproduces the same anon_id -> unit map.
    units.sort(key=lambda u: _short_hash(u["unit_type"],
                                        u.get("_gap_id") or u.get("_idea_id")))
    rng = random.Random(SEED)
    rng.shuffle(units)
    for i, unit in enumerate(units, start=1):
        unit["anon_id"] = f"fbp-{i:03d}"

    mapping = {
        u["anon_id"]: {
            "unit_type": u["unit_type"],
            "topic_id": u["_topic_id"],
            "arm": u["_arm"],
            "real_id": u.get("_gap_id") or u.get("_idea_id"),
        } for u in units
    }

    # Strip every internal field before writing the reviewer-facing file.
    clean = []
    for u in units:
        rec = {k: v for k, v in u.items() if not k.startswith("_")}
        clean.append(rec)

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    inp = os.path.join(OUT_DIR, "adjudication_input.jsonl")
    with open(inp, "w", encoding="utf-8") as fh:
        for rec in clean:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    map_path = os.path.join(OUT_DIR, "review_mapping.json")
    with open(map_path, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": SEED,
            "sealed_at": "2026-11-02",
            "warning": ("SEALED until the adjudication is complete. Opening "
                        "this file before verdicts are written breaks blindness."),
            "units": mapping,
        }, fh, ensure_ascii=False, indent=2)

    # Leak check: nothing in the reviewer file may name a task, arm, or stamp.
    forbidden = ("task_id", "policy_version", "v6", "v7.2", "opportunity-ontology",
                 "evidence-admission", "_arm")
    blob = json.dumps(clean, ensure_ascii=False)
    hits = [f for f in forbidden if f in blob]
    print(f"\nwritten {len(clean)} units -> {inp}")
    print(f"written mapping ({len(mapping)} entries) -> {map_path}")
    print(f"leak check: {'CLEAN' if not hits else 'LEAKED ' + str(hits)}")

    types = Counter(u["gap_type"] for u in clean if u["unit_type"] == "gap")
    print(f"\ngap type distribution in the review set (the disclosed leak):")
    for t, n in types.most_common():
        print(f"  {t:<24} {n}")


if __name__ == "__main__":
    main()

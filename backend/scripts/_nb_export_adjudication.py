"""Anonymised export for the neighbour-budget adjudication (3 vs 8).

Reuses the fixed-budget pilot's blindness discipline (seeded shuffle, sequential
anon ids, sealed mapping, paper identity stripped) with two changes forced by
THIS experiment's design:

1. No audit free text is exported.
   Measured before writing this: 8 of 94 audit text fields name the neighbour
   count outright ("None of the THREE neighbors ...", "the EIGHT prior works"),
   which is the arm label in disguise. Redacting phrases would still leak the
   number semantically ("none of the prior works we compared" is not
   arm-neutral either) — and it would leak MORE than blindness: a reviewer who
   reads the pipeline's own novelty verdict is anchored by it, while the primary
   metric is explicitly the reviewer's INDEPENDENT judgement. So the audit's
   prose is dropped entirely. Machine fields that carry no neighbour count
   (audit_result, novelty_confidence) are also NOT exported for the same
   anchoring reason; they are reported separately as secondary metrics.

2. Paper short-hashes inside audit prose are moot (no prose exported), but the
   evidence block still strips titles/authors/years, and any 8-hex token is
   scrubbed from the remaining text, because our own audit prompts embedded
   paper ids as "(2235e7d7)" and those resolved to real papers.

Leak check is enforced, not advisory: the export aborts if the reviewer-facing
blob contains an arm token, a neighbour-count phrase, or an id-shaped token.

Usage:
    python scripts/_nb_export_adjudication.py --dry-run
    python scripts/_nb_export_adjudication.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3

SEED = 20261103          # fixed before any run; shuffle must be reproducible
PREFIX = "nb-"           # anon id prefix for this experiment
OUT_DIR = os.path.join("eval_results", "neighbor_budget")
MATRIX = os.path.join(OUT_DIR, "run_matrix.json")

AUDITED_STATUSES = {"surviving", "confirmed_runner_up", "inconclusive", "rejected"}
MAX_EVIDENCE_PER_UNIT = 6

# Phrases that would tell the reviewer which arm a unit came from. Matched after
# lowercasing; the export refuses to write a file containing any of them.
_COUNT_PHRASE = re.compile(
    r"\b(three|eight|3|8)\s+(neighbou?rs?|papers?|works?|prior\s+works?|"
    r"candidate\s+papers?|comparisons?)\b", re.I)
_NONE_OF_N = re.compile(r"\bnone\s+of\s+the\s+(three|eight|3|8)\b", re.I)
_HEX8 = re.compile(r"\b[0-9a-f]{8}\b")

# Some Ideas embed the audit's own prose in `motivation` (measured: 4 of 78
# active Ideas in this experiment). That block both states the neighbour count
# ("None of the supplied neighbors ...") and hands the reviewer the pipeline's
# novelty verdict — the exact anchoring this export exists to prevent. The
# generator writes it as a trailing "Audit differentiation: ..." section, so the
# section is cut from the reviewer-facing text. The pipeline's verdict is not
# lost: it is reported as a secondary metric from the audit tables, where it
# cannot contaminate the primary judgement.
_AUDIT_PROSE = re.compile(r"\n*\s*Audit differentiation:.*\Z", re.S)


def _strip_audit_prose(text: str | None) -> str | None:
    if not text or "Audit differentiation:" not in text:
        return text
    return _AUDIT_PROSE.sub("", text).strip() or None


def _short_hash(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _scrub(text: str | None) -> str | None:
    """Remove audit prose and id-shaped tokens from any text we do export."""
    if not text:
        return text
    return _HEX8.sub("[paper]", _strip_audit_prose(text))


def _evidence_block(con, gap_id):
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
            "claim": _scrub((claim or "").strip()),
            "span": _scrub((span or "").strip()[:400]) or None,
            "verification_status": vstatus,
            "link_relation": rel,
        })
    return out


def _gap_unit(con, row, topic_id, stratum):
    (gid, gtype, description, target_setting, observed_problem,
     existing_coverage, missing_capability, claimed_delta,
     testable_hypothesis, falsification_condition, provenance,
     residual_gap) = row
    return {
        "unit_type": "gap",
        "anon_id": None,
        "topic_stratum": stratum,
        "gap_type": gtype,
        "description": _scrub(description),
        "target_setting": _scrub(target_setting),
        "observed_problem": _scrub(observed_problem),
        "existing_coverage": _scrub(existing_coverage),
        "missing_capability": _scrub(missing_capability),
        "claimed_delta": _scrub(claimed_delta),
        "testable_hypothesis": _scrub(testable_hypothesis),
        "falsification_condition": _scrub(falsification_condition),
        "residual_gap": _scrub(residual_gap),
        "provenance_status": provenance,
        "evidence": _evidence_block(con, gid),
        "evidence_identity_redacted": True,
        # Audit prose and the pipeline's own verdict are deliberately ABSENT —
        # see the module docstring. The reviewer must judge novelty unaided.
        "_topic_id": topic_id,
        "_gap_id": gid,
    }


def _idea_unit(row, topic_id, stratum):
    (iid, title, description, motivation, method_sketch,
     expected_contribution) = row
    return {
        "unit_type": "idea",
        "anon_id": None,
        "topic_stratum": stratum,
        "title": _scrub(title),
        "description": _scrub(description),
        "motivation": _scrub(motivation),
        "method_sketch": _scrub(method_sketch),
        "expected_contribution": _scrub(expected_contribution),
        "_topic_id": topic_id,
        "_idea_id": iid,
    }


def collect(con, matrix):
    gaps, ideas = [], []
    for run in matrix["run_order"]:
        if run.get("pair_valid") is False:
            # The invalidated pair is excluded from the adjudication entirely:
            # sending it to review would produce verdicts that cannot be
            # attributed to the treatment (its arm C carried a resumed run).
            continue
        topic_id, stratum = run["topic_id"], run["stratum"]
        for arm in ("C", "D"):
            tid = (run.get("task_ids") or {}).get(arm)
            if not tid:
                continue
            ph = ",".join("?" * len(AUDITED_STATUSES))
            for row in con.execute(f"""
                SELECT id, gap_type, description, target_setting,
                       observed_problem, existing_coverage, missing_capability,
                       claimed_delta, testable_hypothesis,
                       falsification_condition, provenance_status, residual_gap
                FROM gap_candidates
                WHERE task_id=? AND status IN ({ph})
                ORDER BY created_at
            """, (tid, *sorted(AUDITED_STATUSES))).fetchall():
                u = _gap_unit(con, row, topic_id, stratum)
                u["_arm"] = arm
                gaps.append(u)

            for row in con.execute("""
                SELECT id, title, description, motivation, method_sketch,
                       expected_contribution
                FROM research_ideas
                WHERE task_id=? AND idea_status='active'
                ORDER BY created_at
            """, (tid,)).fetchall():
                u = _idea_unit(row, topic_id, stratum)
                u["_arm"] = arm
                ideas.append(u)
    return gaps, ideas


def _leak_check(clean_units) -> list[str]:
    blob = json.dumps(clean_units, ensure_ascii=False)
    hits = []
    for token in ("task_id", "policy_version", "_arm", "neighbor_budget",
                  "gap_audit_neighbor_budget"):
        if token in blob:
            hits.append(f"token:{token}")
    for rx, name in ((_COUNT_PHRASE, "count_phrase"),
                     (_NONE_OF_N, "none_of_N")):
        m = rx.search(blob)
        if m:
            hits.append(f"{name}:{m.group(0)!r}")
    m = _HEX8.search(blob)
    if m:
        hits.append(f"id_shaped:{m.group(0)}")
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with open(MATRIX, encoding="utf-8") as fh:
        matrix = json.load(fh)

    con = sqlite3.connect("file:deep_research.db?mode=ro", uri=True)
    try:
        gaps, ideas = collect(con, matrix)
    finally:
        con.close()

    units = gaps + ideas
    print(f"collected {len(gaps)} gap units, {len(ideas)} idea units")
    for run in matrix["run_order"]:
        if run.get("pair_valid") is False:
            print(f"  {run['topic_id']}: EXCLUDED (invalid pair)")
            continue
        line = []
        for arm in ("C", "D"):
            g = sum(1 for u in gaps if u["_topic_id"] == run["topic_id"]
                    and u["_arm"] == arm)
            i = sum(1 for u in ideas if u["_topic_id"] == run["topic_id"]
                    and u["_arm"] == arm)
            line.append(f"{arm}: {g}g/{i}i")
        print(f"  {run['topic_id']}: " + "   ".join(line))

    units.sort(key=lambda u: _short_hash(u["unit_type"],
                                        u.get("_gap_id") or u.get("_idea_id")))
    random.Random(SEED).shuffle(units)
    for i, u in enumerate(units, start=1):
        u["anon_id"] = f"{PREFIX}{i:03d}"

    mapping = {
        u["anon_id"]: {
            "unit_type": u["unit_type"],
            "topic_id": u["_topic_id"],
            "arm": u["_arm"],
            "real_id": u.get("_gap_id") or u.get("_idea_id"),
        } for u in units
    }
    clean = [{k: v for k, v in u.items() if not k.startswith("_")} for u in units]

    hits = _leak_check(clean)
    print(f"\nleak check: {'CLEAN' if not hits else 'LEAKED -> ' + str(hits)}")
    if hits and not args.dry_run:
        raise SystemExit("refusing to write a leaking adjudication file")

    if args.dry_run:
        print("[dry-run] no files written")
        return

    inp = os.path.join(OUT_DIR, "adjudication_input.jsonl")
    with open(inp, "w", encoding="utf-8") as fh:
        for rec in clean:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    map_path = os.path.join(OUT_DIR, "review_mapping.json")
    with open(map_path, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": SEED,
            "sealed_at": "2026-11-06",
            "warning": ("SEALED until all verdicts are written. Opening this "
                        "before verdicts exist breaks blindness."),
            "excluded_pairs": [r["topic_id"] for r in matrix["run_order"]
                               if r.get("pair_valid") is False],
            "units": mapping,
        }, fh, ensure_ascii=False, indent=2)

    print(f"written {len(clean)} units -> {inp}")
    print(f"written sealed mapping ({len(mapping)}) -> {map_path}")
    arms = {}
    for u in units:
        arms[u["_arm"]] = arms.get(u["_arm"], 0) + 1
    print(f"arm balance (sealed): {arms}")


if __name__ == "__main__":
    main()

"""Pass 2: retrieval-repaired re-adjudication of units whose search failed.

Why this exists (disclosed measurement repair, NOT an outcome edit)
------------------------------------------------------------------
Pass 1 produced a -4.0pp arm difference that the whole §7 NO-GO rests on, but
that difference is carried by exactly one unit (fbp-089). That unit's own
`notes` say the retrieval was "largely off-target" and missed the canonical
literature for its topic; `key_prior_work` is empty. Its NONE is therefore an
artifact of a failed search, not evidence of novelty. Six units in total show
failed or off-target retrieval (five flagged `search_useful=false`, plus
fbp-089 whose prose admits the failure while its boolean says True).

The repair changes ONLY the retrieval, never the rubric and never the
verdict criterion. Queries are rebuilt from the pipeline's OWN nearest-
neighbour paper titles (stored in the DB for that gap) — the exact fix already
prescribed for the killer-search mechanism, because the failure mode is
identical: LLM-coined query terms do not occur in the real literature, whereas
real neighbour titles do. The reviewer still never sees article authorship,
provenance or arm.

Disclosure requirement: pass-2 verdicts are reported SEPARATELY and the final
report must state both the pass-1 and pass-2 metrics. Substituting pass 2 for
pass 1 without disclosure would be outcome editing.
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.adjudicate_pilot import (  # noqa: E402
    GapVerdict, IdeaVerdict, RUBRIC, _run_search, _unit_brief,
)

IN_DIR = r"eval_results\fixed_budget_pilot"
IN_PATH = os.path.join(IN_DIR, "adjudication_input.jsonl")
MAP_PATH = os.path.join(IN_DIR, "review_mapping.json")
OUT_PATH = os.path.join(IN_DIR, "verdicts.pass2.jsonl")

# Units whose pass-1 retrieval failed or was declared off-target.
FAILED_RETRIEVAL = ["fbp-037", "fbp-058", "fbp-071", "fbp-085",
                    "fbp-087", "fbp-089"]


def _neighbour_titles(gap_id: str) -> list[str]:
    """Real neighbour paper titles the pipeline recorded for this gap."""
    import sqlite3
    con = sqlite3.connect("deep_research.db")
    try:
        row = con.execute(
            "SELECT neighbor_paper_ids_json FROM gap_audits "
            "WHERE gap_id=? ORDER BY created_at DESC LIMIT 1",
            (gap_id,)).fetchone()
        if not row or not row[0]:
            return []
        ids = json.loads(row[0])[:6]
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        return [r[0] for r in con.execute(
            f"SELECT title FROM papers WHERE id IN ({q})", ids).fetchall()
            if r[0]]
    finally:
        con.close()


async def repair_one(llm, unit: dict, real_gap_id: str) -> dict:
    titles = _neighbour_titles(real_gap_id)

    # Domain-vocabulary queries derived from real neighbour titles: strip
    # subtitle after ':' and keep the head, which is where the method/domain
    # words live.
    title_queries = []
    for t in titles:
        head = t.split(":")[0].strip()
        head = " ".join(head.split()[:10])
        if len(head) > 12:
            title_queries.append(head)
    title_queries = title_queries[:2]

    domain_note = ""
    if title_queries:
        domain_note = (
            "\n\nADDITIONAL CONTEXT — titles of the nearest works the pipeline "
            "itself recorded for this claim (identity of the *reviewer's* "
            "sources is still withheld; these are only given so your queries "
            "use vocabulary that occurs in this literature):\n"
            + "\n".join(f"  - {t}" for t in titles))

    brief = _unit_brief(unit)
    hits = []
    queries_used = []
    for attempt_queries in (title_queries, None):
        if attempt_queries:
            queries = attempt_queries
        else:
            # Fall back to LLM-proposed queries with an explicit instruction
            # to avoid coined terms (the pass-1 failure mode).
            from scripts.adjudicate_pilot import SearchPlan
            plan = await llm.chat_json([
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": (
                    f"{brief}{domain_note}\n\nPropose 3 short academic search "
                    "queries using vocabulary that ACTUALLY OCCURS in this "
                    "literature (method names, task names, standard terms). "
                    "Do NOT coin phrases from the claim wording.")},
            ], SearchPlan)
            queries = [q.strip() for q in plan.queries if q.strip()][:3]
        queries_used = queries
        hits = await _run_search(queries)
        if len(hits) >= 5:
            break

    rendered = "\n".join(
        f"  [{i}] {h['title']} ({h.get('year') or 'n/a'}"
        f"{', ' + h['venue'] if h.get('venue') else ''})\n      "
        f"{(h.get('abstract') or '(no abstract)')[:420]}"
        for i, h in enumerate(hits, start=1)) or "  (no results retrieved)"

    schema = GapVerdict if unit["unit_type"] == "gap" else IdeaVerdict
    scale = ("NONE / PARTIAL / FULL" if unit["unit_type"] == "gap"
             else "fund / maybe / no")
    verdict = await llm.chat_json([
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"{brief}{domain_note}\n\nSEARCH QUERIES USED (repaired pass):\n"
            + "\n".join(f"  - {q}" for q in queries)
            + f"\n\nSEARCH RESULTS:\n{rendered}\n\n"
              f"Answer with the scale only: {scale}. Cite works by index.")},
    ], schema)

    return {
        "anon_id": unit["anon_id"],
        "unit_type": unit["unit_type"],
        "topic_stratum": unit.get("topic_stratum"),
        "gap_type": unit.get("gap_type"),
        "queries": queries,
        "retrieved": hits,
        "verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "key_prior_work": verdict.key_prior_work,
        "residual_novelty": verdict.residual_novelty,
        "notes": verdict.notes,
        "search_useful": verdict.search_useful,
        "blinded_context": True,
        "reviewer": "llm_web_verified_repair",
        "repair_reason": "pass-1 retrieval off-target; queries rebuilt from "
                         "real neighbour titles",
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    units = {json.loads(l)["anon_id"]: json.loads(l)
             for l in open(IN_PATH, encoding="utf-8") if l.strip()}
    mapping = json.load(open(MAP_PATH, encoding="utf-8"))["units"]

    from app.llm.factory import get_llm
    llm = get_llm()

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for aid in FAILED_RETRIEVAL:
            unit, meta = units[aid], mapping[aid]
            print(f"  {aid} (arm {meta['arm']}, {unit.get('gap_type') or 'idea'})"
                  f" ...", end=" ", flush=True)
            try:
                res = await repair_one(llm, unit, meta["real_id"])
            except Exception as exc:
                print(f"ERROR {type(exc).__name__}: {str(exc)[:100]}")
                continue
            print(f"{res['verdict']} ({res['confidence']}, "
                  f"hits={len(res['retrieved'])})")
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()

    print(f"\nwrote -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

"""§8.2 LLM adjudicator for the fixed-budget pilot (prior-art verification).

Independence requirements this script enforces
----------------------------------------------
The v7 replay pilot's known weakness was that the SAME LLM that generated the
material also judged it. This adjudicator is still an LLM, so it cannot be
made independent by construction — but three things reduce the overlap and
one makes the overlap measurable:

1. It never sees which arm produced a unit (§8.1 export strips it).
2. It does NOT trust the pipeline's own audit verdict. The audit's
   nearest-neighbour summary is passed only as `pipeline_claim_summary`
   (labelled "the pipeline's own assertion, not evidence") so the reviewer can
   notice contradiction, but the verdict must rest on ITS OWN retrieval.
3. It performs its own search over Semantic Scholar / OpenAlex / arXiv using
   queries derived from the CLAIM, not from the pipeline's adversarial
   queries (those were optimised to close the gap, not to open it).
4. `blinded_context: true` is recorded per verdict, and the human spot-check
   (§8.2) is the instrument that measures whether all this actually worked.

Verdict rubric (inherited verbatim from the pilot so the numbers are
comparable — do not "improve" it mid-experiment):

  NONE    — no prior work covers the claimed delta; a researcher could
            plausibly publish the claimed contribution.
  PARTIAL — some prior work covers part of the delta, or covers it in a
            different setting, or the delta is a re-parameterisation of known
            work. The claim needs narrowing but retains residual novelty.
  FULL    — prior work already establishes the claimed delta (false open).

For ideas:
  fund / maybe / no — "would you fund this experiment", judged on whether the
  claimed contribution is worth the stated cost given the retrieved prior art.

Usage:
    python scripts/adjudicate_pilot.py --dry-run          # 3 units
    python scripts/adjudicate_pilot.py --limit 10         # first N
    python scripts/adjudicate_pilot.py                    # all
    python scripts/adjudicate_pilot.py --resume           # skip done ids
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Literal

import httpx
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.config import settings  # noqa: E402
from app.services.semantic_recall import semantic_recall  # noqa: E402

IN_DIR = r"eval_results\fixed_budget_pilot"
IN_PATH = os.path.join(IN_DIR, "adjudication_input.jsonl")
OUT_PATH = os.path.join(IN_DIR, "verdicts.jsonl")

SEARCH_SOURCES = ["semantic_scholar", "openalex", "arxiv"]
SEARCH_LIMIT = 8
MAX_QUERIES = 3
# How many filtered hits the reviewer is shown. Deliberately larger than the
# pre-filter display cap: the filter already removed the noise, so more genuine
# neighbours can be shown without burying the reviewer.
MAX_HITS = 12

# The rubric text is part of the experiment's measurement instrument: it must
# stay byte-identical to the pilot's, otherwise the true-rates are not
# comparable across pilots.
RUBRIC = """You are an independent prior-art reviewer for a research-gap
adjudication. You will be shown ONE claimed research gap (or ONE research
idea) extracted from a literature review, together with a set of evidence
claims that the pipeline used.

Your job: decide whether the claimed contribution is already covered by prior
work. Use the search results supplied to you — they were retrieved from
Semantic Scholar / OpenAlex / arXiv with queries derived from the claim.

Verdict rubric for GAPS (use exactly one):
- NONE: no prior work covers the claimed delta. A researcher could plausibly
  publish the claimed contribution as new.
- PARTIAL: prior work covers PART of the delta, covers it in a different
  setting, or the delta is a re-parameterisation of known work. The claim
  needs narrowing but retains residual novelty.
- FULL: prior work already establishes the claimed delta. The gap is a false
  open.

Verdict rubric for IDEAS (use exactly one):
- fund: the experiment is worth running; the contribution would be new and
  the cost is proportionate.
- maybe: worth considering but the novelty or the cost is questionable.
- no: not worth running; already done, or not a real contribution.

Rules:
- Base your verdict on the SEARCH RESULTS, not on the pipeline's own audit
  verdict. The pipeline's assertion is shown to you labelled as an assertion.
- If the search results are irrelevant or empty, say so explicitly in notes
  and use your own knowledge, flagging `search_useful: false`.
- Be conservative about NONE: absence of a hit in these results is weak
  evidence of absence in the literature.
- Quantify your confidence (high / medium / low) and say which retrieved work
  (by its number) drove the verdict."""


class SearchPlan(BaseModel):
    """Queries the reviewer wants run before judging."""

    queries: list[str] = Field(
        description="2-3 short academic search queries (English) that would "
                    "find prior work covering the claimed delta. Prefer "
                    "domain vocabulary over coined terms.")


class _VerdictBase(BaseModel):
    confidence: Literal["high", "medium", "low"]
    key_prior_work: list[str] = Field(
        default_factory=list,
        description="Indices (1-based) of the retrieved works that drove the "
                    "verdict, e.g. ['2', '5'].")
    residual_novelty: str = Field(
        default="",
        description="What (if anything) remains uncovered by prior work.")
    notes: str = Field(default="", description="Free-form reviewer notes.")
    search_useful: bool = True


class GapVerdict(_VerdictBase):
    """Gap rubric only. Separate schema so an idea cannot be scored NONE."""

    verdict: Literal["NONE", "PARTIAL", "FULL"]


class IdeaVerdict(_VerdictBase):
    """Idea rubric only. Separate schema so a gap cannot be scored fund."""

    verdict: Literal["fund", "maybe", "no"]


def _load_units(path=IN_PATH):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _unit_brief(unit: dict) -> str:
    """Render the unit WITHOUT its pseudonymous processor identity.

    `pipeline_*` fields stay (the design keeps them on purpose: the
    adjudication also reports how often the pipeline's own judgement matched),
    but they are explicitly framed as assertions, not evidence.
    """
    if unit["unit_type"] == "gap":
        ev = "\n".join(
            f"  [E{i}] ({e['evidence_type']}, {e['verification_status']}) "
            f"{e['claim']}" + (f" | span: {e['span']}" if e.get("span") else "")
            for i, e in enumerate(unit.get("evidence") or [], start=1))
        audit = unit.get("audit") or {}
        return f"""CLAIMED GAP
type: {unit['gap_type']}
description: {unit['description']}
target setting: {unit.get('target_setting') or '(not stated)'}
observed problem: {unit.get('observed_problem') or '(not stated)'}
existing coverage: {unit.get('existing_coverage') or '(not stated)'}
missing capability: {unit.get('missing_capability') or '(not stated)'}
claimed delta: {unit.get('claimed_delta') or '(not stated)'}
testable hypothesis: {unit.get('testable_hypothesis') or '(not stated)'}
falsification condition: {unit.get('falsification_condition') or '(not stated)'}

EVIDENCE CLAIMS (paper identity withheld by the export)
{ev or '  (none)'}

PIPELINE'S OWN ASSERTION (not evidence — do not rely on it):
  audit_result: {audit.get('audit_result')}
  recommended_action: {audit.get('recommended_action')}
  its novelty_confidence: {audit.get('novelty_confidence')}
  its nearest-neighbour summary: {audit.get('nearest_neighbor_summary') or '(none)'}
  its differentiation summary: {audit.get('differentiation_summary') or '(none)'}"""

    return f"""RESEARCH IDEA
title: {unit['title']}
description: {unit.get('description') or '(not stated)'}
motivation: {unit.get('motivation') or '(not stated)'}
method sketch: {unit.get('method_sketch') or '(not stated)'}
expected contribution: {unit.get('expected_contribution') or '(not stated)'}

PIPELINE'S OWN ASSERTION (not evidence — do not rely on it):
  decision: {unit.get('pipeline_decision')}
  confidence tier: {unit.get('pipeline_confidence_tier')}
  its final score: {unit.get('pipeline_final_score')}"""


async def _run_search(queries: list[str]) -> list[dict]:
    from app.services.search_service import SearchService

    svc = SearchService()
    try:
        papers = await svc.search_multiple_queries(
            queries, limit=SEARCH_LIMIT, allowed_sources=SEARCH_SOURCES)
    except Exception as exc:  # infra failure must degrade, never crash the run
        print(f"    [search failed: {type(exc).__name__}: {exc}]")
        return []

    seen, out = set(), []
    for p in papers:
        key = (p.title or "").strip().lower()[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "title": p.title,
            "year": getattr(p, "year", None),
            "venue": getattr(p, "venue", None),
            "abstract": (getattr(p, "abstract", None) or "")[:600],
        })
        if len(out) >= 12:
            break
    return out


async def adjudicate_unit(llm, unit: dict) -> dict:
    brief = _unit_brief(unit)

    plan = await llm.chat_json([
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"{brief}\n\nBefore judging, propose {MAX_QUERIES} short academic "
            "search queries that would find prior work covering the claimed "
            "delta. Use domain vocabulary, not coined terms.")},
    ], SearchPlan)

    queries = [q.strip() for q in plan.queries if q.strip()][:MAX_QUERIES]

    # Pass 3 retrieval (2026-11-02): keyword search alone returned pools where
    # ZERO papers were about the claim (fbp-089: 12 statistical-mediation and
    # medical-GNN papers for an MoE-routing claim). Semantic recall filters and
    # reranks by embedding distance to the claim, so the reviewer ranks
    # relevance instead of reading keyword noise. The kept/filtered counts are
    # recorded and disclosed below — a hidden filter would be the same bug in a
    # new form.
    if settings.semantic_recall_enabled:
        claim = " ".join(filter(None, [
            unit.get("description"), unit.get("observed_problem"),
            unit.get("missing_capability"), unit.get("claimed_delta"),
            unit.get("title"), unit.get("method_sketch")])).strip()
        hits, recall_diag = await semantic_recall(
            claim, queries, limit=MAX_HITS,
            allowed_sources=SEARCH_SOURCES,
            overfetch=settings.semantic_recall_overfetch)
    else:
        hits = await _run_search(queries)
        recall_diag = {"semantic_filtered": False, "raw_count": len(hits),
                       "kept_count": len(hits), "filtered_count": 0}

    rendered = "\n".join(
        f"  [{i}] {h['title']} ({h.get('year') or 'n/a'}"
        f"{', ' + h['venue'] if h.get('venue') else ''})"
        f"{'  sim=' + str(h['similarity']) if h.get('similarity') is not None else ''}\n      "
        f"{(h.get('abstract') or '(no abstract)')[:420]}"
        for i, h in enumerate(hits, start=1)) or "  (no results retrieved)"

    # Disclose the retrieval funnel to the reviewer: without the raw count, an
    # empty filtered list reads as "no prior art" when it means "everything was
    # off-topic" or "the filter was too strict".
    retrieval_note = ""
    if recall_diag.get("semantic_filtered"):
        retrieval_note = (
            f"\n\nRETRIEVAL NOTE: {recall_diag['raw_count']} keyword matches "
            f"were retrieved and {recall_diag['kept_count']} passed a semantic "
            f"relevance filter (threshold {recall_diag['threshold']}, "
            f"{recall_diag['filtered_count']} discarded as off-topic). "
            "Weigh absence accordingly: a discarded item was judged unrelated "
            "to the claim, not merely unhelpful.")
    elif recall_diag.get("degraded"):
        retrieval_note = (
            f"\n\nRETRIEVAL NOTE: semantic filtering was unavailable "
            f"({recall_diag['degraded']}); results are unfiltered keyword "
            "matches, so some may be entirely off-topic.")

    # The rubric text contains BOTH scales (that is how the pilot's instrument
    # is written), so the schema is what keeps them apart: a gap unit can only
    # receive NONE/PARTIAL/FULL and an idea only fund/maybe/no. Passing one
    # shared six-value schema let the model answer PARTIAL for an idea
    # (observed in the dry run), which would silently merge two scales.
    verdict_schema = GapVerdict if unit["unit_type"] == "gap" else IdeaVerdict
    scale_note = ("Answer with the GAP scale only: NONE / PARTIAL / FULL."
                  if unit["unit_type"] == "gap"
                  else "Answer with the IDEA scale only: fund / maybe / no.")
    verdict = await llm.chat_json([
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"{brief}\n\nSEARCH QUERIES USED:\n"
            + "\n".join(f"  - {q}" for q in queries)
            + f"\n\nSEARCH RESULTS (retrieved from Semantic Scholar / OpenAlex "
              f"/ arXiv):\n{rendered}{retrieval_note}\n\n"
              f"Now give your verdict. {scale_note} "
              "Cite retrieved works by their index.")},
    ], verdict_schema)

    return {
        "anon_id": unit["anon_id"],
        "unit_type": unit["unit_type"],
        "topic_stratum": unit.get("topic_stratum"),
        "gap_type": unit.get("gap_type"),
        "queries": queries,
        "retrieved": hits,
        # Retrieval funnel, recorded so a future reader can tell "no prior art"
        # from "everything retrieved was off-topic" from "the filter was too
        # strict" — the distinction the pass-1 data could not make.
        "retrieval": recall_diag,
        "verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "key_prior_work": verdict.key_prior_work,
        "residual_novelty": verdict.residual_novelty,
        "notes": verdict.notes,
        "search_useful": verdict.search_useful,
        "blinded_context": True,
        "reviewer": "llm_web_verified_semantic",
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="judge 3 units and print, write nothing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip anon_ids already present in verdicts.jsonl")
    ap.add_argument("--only", default="", help="comma-separated anon_ids")
    args = ap.parse_args()

    units = _load_units()
    done = set()
    if args.resume and os.path.exists(OUT_PATH):
        done = {json.loads(l)["anon_id"]
                for l in open(OUT_PATH, encoding="utf-8") if l.strip()}

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        units = [u for u in units if u["anon_id"] in wanted]
    elif done:
        units = [u for u in units if u["anon_id"] not in done]

    # Gaps first: the pre-registered primary metric is the gap true-rate, so
    # the gap set must complete even if a long run is interrupted. Ideas are
    # a secondary outcome and can be finished later via --resume.
    units.sort(key=lambda u: (u["unit_type"] != "gap", u["anon_id"]))

    if args.limit:
        units = units[:args.limit]
    if args.dry_run:
        units = units[:3]

    print(f"adjudicating {len(units)} unit(s) "
          f"({len(done)} already done, {len(units)} in this pass)")

    from app.llm.factory import get_llm
    llm = get_llm()

    # NEVER truncate an existing verdict file. The previous `mode = "w"` when
    # --resume was absent destroyed a complete 91-unit adjudication when it was
    # re-run with --only for a single unit (2026-11-02): the run wrote 1 record
    # over 91, and no backup existed. Accumulating is also the correct
    # semantics — verdicts are keyed by anon_id and later duplicates are
    # resolved by the summary scripts reading last-wins.
    #
    # To start a genuinely fresh pass, write to a different --out path.
    if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 0:
        _backup = OUT_PATH + ".bak"
        if not os.path.exists(_backup):
            import shutil
            shutil.copy2(OUT_PATH, _backup)
            print(f"  (backed up existing verdicts -> {_backup})")
    mode = "a"
    fh = None if args.dry_run else open(OUT_PATH, mode, encoding="utf-8")

    try:
        for idx, unit in enumerate(units, start=1):
            print(f"  [{idx}/{len(units)}] {unit['anon_id']} "
                  f"({unit['unit_type']}"
                  f"{': ' + unit['gap_type'] if unit.get('gap_type') else ''}) ...",
                  end=" ", flush=True)
            try:
                result = await adjudicate_unit(llm, unit)
            except Exception as exc:
                print(f"ERROR {type(exc).__name__}: {str(exc)[:120]}")
                continue
            print(f"{result['verdict']} ({result['confidence']}, "
                  f"hits={len(result['retrieved'])})")
            if fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()

    if args.dry_run:
        print("\n[dry-run] nothing written")
    else:
        print(f"\nwrote -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

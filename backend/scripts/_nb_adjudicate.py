"""LLM adjudicator for the neighbour-budget experiment (3 vs 8).

This reuses the fixed-budget pilot's instrument verbatim, because the two
experiments must be comparable and an "improved" rubric would make them not so.
What is inherited, unchanged:

  * the RUBRIC text (imported from adjudicate_pilot, not retyped, so it cannot drift),
  * the verdict schema (NONE / PARTIAL / FULL for gaps; fund / maybe / no for ideas),
  * the reviewer's OWN retrieval (Semantic Scholar / OpenAlex / arXiv, queries
    derived from the CLAIM, then semantic filtering),
  * blindness: the unit record contains no arm, no task id, no pipeline verdict.

Two things this experiment must NOT hide
----------------------------------------
1. Independence. The reviewer is still an LLM, so "independent" cannot be true
   by construction — it is reduced, not achieved. The human spot-check
   (`human_spotcheck.md`, judged separately) is the instrument that measures
   how well the adjudicator agrees with a careful human, and it is run BEFORE
   the main verdicts drive any decision.

2. The pilot's known failure mode. Its pass-1 retrieval was broken (one unit
   returned 12 results and 0 relevant ones, and carried the whole -4.0pp
   delta). The repair was semantic recall with a calibrated threshold. This
   script records `search_useful` and `n_search_hits` per verdict so a repeat of
   that failure is visible in the data rather than inferred afterwards.

The one deliberate difference from the pilot: our export contains NO audit
prose and NO pipeline verdict (the pilot showed the pipeline's assertion as
`pipeline_claim_summary`). For a 3-vs-8 neighbour comparison that would be
fatal — the audit text names the neighbour count ("None of the three
neighbors..."), which is the arm label, and it also anchors the reviewer on
exactly the judgement under test.

Usage:
    python scripts/_nb_adjudicate.py --dry-run
    python scripts/_nb_adjudicate.py --limit 5
    python scripts/_nb_adjudicate.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings  # noqa: E402
from app.services.semantic_recall import semantic_recall  # noqa: E402
from adjudicate_pilot import RUBRIC  # noqa: E402

IN_DIR = os.path.join("eval_results", "neighbor_budget")
# Overridable so the out-of-sample validation set can reuse this exact
# instrument (same rubric, same retrieval, same blindness) instead of a copy
# that could drift from it.
IN_PATH = os.environ.get("NB_IN_PATH",
                         os.path.join(IN_DIR, "adjudication_input.jsonl"))
OUT_PATH = os.environ.get("NB_OUT_PATH",
                          os.path.join(IN_DIR, "verdicts.jsonl"))

SEARCH_SOURCES = ["semantic_scholar", "openalex", "arxiv"]
SEARCH_LIMIT = 8
MAX_QUERIES = 3
MAX_HITS = 12


class SearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)


class _VerdictBase(BaseModel):
    anon_id: str
    unit_type: str
    verdict: str
    confidence: Literal["high", "medium", "low"]
    notes: str
    residual_novelty: str = ""
    search_useful: bool = True
    key_prior_work: list[str] = Field(default_factory=list)
    blinded_context: bool = True


class GapVerdict(_VerdictBase):
    verdict: Literal["NONE", "PARTIAL", "FULL"]


class IdeaVerdict(_VerdictBase):
    verdict: Literal["fund", "maybe", "no"]


def _load_units(path=None):
    # Bound at call time, not at def time: the default must pick up the env
    # override, otherwise the validation run silently reads the main set.
    with open(path or IN_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _unit_brief(unit: dict) -> str:
    """Render the unit for the reviewer. No arm, no task id, no pipeline verdict."""
    if unit["unit_type"] == "gap":
        parts = [
            f"TOPIC STRATUM: {unit.get('topic_stratum')}",
            f"GAP TYPE: {unit.get('gap_type')}",
            f"DESCRIPTION: {unit.get('description') or ''}",
        ]
        for label, key in (("TARGET SETTING", "target_setting"),
                           ("OBSERVED PROBLEM", "observed_problem"),
                           ("EXISTING COVERAGE", "existing_coverage"),
                           ("MISSING CAPABILITY", "missing_capability"),
                           ("CLAIMED DELTA", "claimed_delta"),
                           ("TESTABLE HYPOTHESIS", "testable_hypothesis"),
                           ("FALSIFICATION CONDITION", "falsification_condition")):
            if unit.get(key):
                parts.append(f"{label}: {unit[key]}")
        ev = unit.get("evidence") or []
        if ev:
            parts.append("EVIDENCE CLAIMS (paper identity redacted):")
            for i, e in enumerate(ev, 1):
                parts.append(f"  {i}. [{e.get('evidence_type')}] "
                             f"{(e.get('claim') or '')[:400]}")
        return "\n".join(parts)
    return "\n".join([
        f"TOPIC STRATUM: {unit.get('topic_stratum')}",
        f"IDEA TITLE: {unit.get('title')}",
        f"DESCRIPTION: {unit.get('description') or ''}",
        f"MOTIVATION: {unit.get('motivation') or ''}",
        f"METHOD SKETCH: {unit.get('method_sketch') or ''}",
        f"EXPECTED CONTRIBUTION: {unit.get('expected_contribution') or ''}",
    ])


async def _run_search(claim: str, queries: list[str]) -> tuple[list[dict], dict]:
    """The reviewer's own retrieval, with the shared semantic filter.

    Returns (hits, diagnostics). Diagnostics are not optional decoration: the
    pilot's pass-1 delta was produced by a retriever that returned 0 relevant
    results, and that was only visible because failures were recorded. A silent
    empty list would let the same failure recur undetected.
    """
    try:
        hits, diag = await semantic_recall(
            claim, queries[:MAX_QUERIES], limit=MAX_HITS,
            allowed_sources=SEARCH_SOURCES,
            overfetch=settings.semantic_recall_overfetch)
    except Exception as exc:  # noqa: BLE001 - a failed search is data, not a crash
        return [], {"semantic_filtered": False,
                    "degraded": f"search_failed: {str(exc)[:150]}",
                    "raw_count": 0, "kept_count": 0}
    return list(hits or []), dict(diag or {})


async def adjudicate_unit(llm, unit: dict) -> dict:
    brief = _unit_brief(unit)
    plan = await llm.chat_json([
        {"role": "system", "content": (
            "You design literature-search queries to CHECK whether a claimed "
            "research contribution is already covered by prior work. Derive "
            "queries from the CLAIM itself (its mechanism, setting, and "
            "evaluation), not from the pipeline's framing. Return 1-3 short "
            "keyword queries in English.")},
        {"role": "user", "content": brief},
    ], SearchPlan)

    claim = " ".join(filter(None, [unit.get("description"),
                                   unit.get("claimed_delta"),
                                   unit.get("missing_capability"),
                                   unit.get("title"),
                                   unit.get("method_sketch")]))
    hits, recall_diag = await _run_search(claim.strip(), plan.queries)
    rendered = "\n".join(
        f"[{i}] {h.get('title')} ({h.get('year') or 'n.d.'})\n"
        f"    {(h.get('abstract') or '')[:700]}"
        for i, h in enumerate(hits, 1)) or "(no search results)"

    schema = GapVerdict if unit["unit_type"] == "gap" else IdeaVerdict
    verdict = await llm.chat_json([
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"UNIT ({unit['unit_type']}): {unit['anon_id']}\n\n{brief}\n\n"
            f"SEARCH RESULTS (retrieved by you, {len(hits)} hits):\n{rendered}")},
    ], schema)
    out = verdict.model_dump()
    out["anon_id"] = unit["anon_id"]
    out["unit_type"] = unit["unit_type"]
    out["blinded_context"] = True
    out["n_search_hits"] = len(hits)
    out["recall_diagnostics"] = recall_diag
    return out


def _load_done(path=None) -> set[str]:
    path = path or OUT_PATH          # bound at call time (env override)
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    done.add(json.loads(line)["anon_id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ids", default="", help="comma-separated anon_ids")
    args = ap.parse_args(argv)

    units = _load_units()
    if args.ids:
        keep = {x.strip() for x in args.ids.split(",") if x.strip()}
        units = [u for u in units if u["anon_id"] in keep]
    if args.resume:
        done = _load_done()
        units = [u for u in units if u["anon_id"] not in done]
    if args.limit:
        units = units[:args.limit]

    print(f"units to adjudicate: {len(units)}")
    if args.dry_run:
        for u in units[:3]:
            print("-" * 70)
            print(_unit_brief(u)[:600])
        return 0

    from app.llm.factory import get_llm
    llm = get_llm()

    written = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, unit in enumerate(units, 1):
            try:
                rec = await adjudicate_unit(llm, unit)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                print(f"[{i}/{len(units)}] {unit['anon_id']} FAILED: {exc}")
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"[{i}/{len(units)}] {unit['anon_id']} "
                  f"{rec['unit_type']:4} -> {rec['verdict']:8} "
                  f"hits={rec.get('n_search_hits')} "
                  f"useful={rec.get('search_useful')}")
            await asyncio.sleep(1.0)

    print(f"\nwrote {written} verdicts -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

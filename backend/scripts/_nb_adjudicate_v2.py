"""Adjudicator with selectable rubric version (v1 default, v2 opt-in).

Everything except the rubric and verdict schema is inherited from
`_nb_adjudicate` (same retrieval, same semantic filter, same blindness, same
output paths). The rubric is the only variable, so a v1-vs-v2 difference can
only be attributed to the instrument.

Usage:
    python scripts/_nb_adjudicate_v2.py --rubric v2 --probes   # gate check FIRST
    python scripts/_nb_adjudicate_v2.py --rubric v2 --resume   # then real units
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("_nba", "_nb_adjudicate.py")
v2 = _load("_nbv2", "adjudicate_rubric_v2.py")

# Pydantic resolves annotations from the module namespace; when a module is
# executed via importlib instead of a normal import, `Literal` is not visible in
# the way the forward-ref machinery expects, so the models raise
# "not fully defined" on first use. Rebuilding them explicitly fixes it without
# changing the schemas.
for _model in (v2.GapVerdictV2, v2.IdeaVerdictV2):
    _model.model_rebuild(_types_namespace={"Literal": __import__("typing").Literal})

PROBE_PATH = os.path.join("eval_results", "neighbor_budget", "rubric_v2_probes.jsonl")


def _rubric(version: str):
    if version == "v1":
        return base.RUBRIC, base.GapVerdict, base.IdeaVerdict
    return v2.RUBRIC_V2, v2.GapVerdictV2, v2.IdeaVerdictV2


async def adjudicate_unit(llm, unit: dict, version: str) -> dict:
    """Same flow as v1, with the rubric and schema chosen by version."""
    rubric, gap_schema, idea_schema = _rubric(version)
    brief = base._unit_brief(unit)

    plan = await llm.chat_json([
        {"role": "system", "content": (
            "You design literature-search queries to CHECK whether a claimed "
            "research contribution is already covered by prior work. Derive "
            "queries from the CLAIM itself (its mechanism, setting, and "
            "evaluation), not from the pipeline's framing. Return 1-3 short "
            "keyword queries in English.")},
        {"role": "user", "content": brief},
    ], base.SearchPlan)

    claim = " ".join(filter(None, [unit.get("description"),
                                   unit.get("claimed_delta"),
                                   unit.get("missing_capability"),
                                   unit.get("title"),
                                   unit.get("method_sketch")]))
    hits, recall_diag = await base._run_search(claim.strip(), plan.queries)
    rendered = "\n".join(
        f"[{i}] {h.get('title')} ({h.get('year') or 'n.d.'})\n"
        f"    {(h.get('abstract') or '')[:700]}"
        for i, h in enumerate(hits, 1)) or "(no search results)"

    schema = gap_schema if unit["unit_type"] == "gap" else idea_schema
    verdict = await llm.chat_json([
        {"role": "system", "content": rubric},
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
    out["rubric_version"] = v2.RUBRIC_VERSION if version == "v2" else "v1-pilot"
    return out


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rubric", choices=["v1", "v2"], default="v1")
    ap.add_argument("--probes", action="store_true",
                    help="run the pre-registered calibration probes")
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--out", dest="out", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    in_path = args.inp or (PROBE_PATH if args.probes else base.IN_PATH)
    out_path = args.out or (
        os.path.join("eval_results", "neighbor_budget",
                     f"probe_verdicts_{args.rubric}.jsonl") if args.probes
        else os.path.join("eval_results", "neighbor_budget",
                          f"verdicts_{args.rubric}.jsonl"))

    units = base._load_units(in_path)
    if args.resume:
        done = base._load_done(out_path)
        units = [u for u in units if u["anon_id"] not in done]
    if args.limit:
        units = units[:args.limit]

    print(f"rubric={args.rubric} units={len(units)} in={in_path}")
    if not units:
        return 0

    from app.llm.factory import get_llm
    llm = get_llm()

    written = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, unit in enumerate(units, 1):
            try:
                rec = await adjudicate_unit(llm, unit, args.rubric)
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}/{len(units)}] {unit['anon_id']} FAILED: {exc}")
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            diag = ""
            if args.rubric == "v2":
                diag = (f" established={rec.get('delta_established')}"
                        f" adjacent={rec.get('adjacent_work_exists')}")
            print(f"[{i}/{len(units)}] {unit['anon_id']} -> {rec['verdict']} "
                  f"hits={rec.get('n_search_hits')}{diag}")
            await asyncio.sleep(1.0)

    print(f"\nwrote {written} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

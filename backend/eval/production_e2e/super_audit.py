"""production_e2e_v1 — Independent Super Audit (design point 3).

The production Gap Audit must NOT be re-run and called independent validation:
it would share query family, retrieval sources, model preference and retrieval
blind spots with the system under test — both could miss the same killer paper.

This module instead freezes the three systems' outputs and searches for
candidate "killer" prior art with:
  - a DIFFERENT prompt template (wider query families: exact / paraphrase /
    adjacent-domain / survey-style),
  - multi-source retrieval (OpenAlex + Semantic Scholar + arXiv),
  - (v1.1 TODO) citation snowballing from the idea's cited papers.

It never makes the FULL/PARTIAL/NONE verdict itself: candidates go to
human_review.md for cheap manual adjudication (only machine-found nearest
prior art plus a small sample of "no killer found" controls).

Usage:
    cd backend
    python -m eval.production_e2e.super_audit --run-id pe2e_v1_audit \
        --systems ../eval_results/pe2e_v1_direct ../eval_results/pe2e_v1_retellm \
                  ../eval_results/pe2e_v1_fullv2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from eval.common import CallStats, EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.baseline_direct import GENERATION_TEMPERATURE

BENCHMARK = "production_e2e"
MODE = "super_audit"
CANDIDATES_PER_TARGET = 12
PER_QUERY_PER_SOURCE = 5

SUPER_AUDIT_SYSTEM_PROMPT = (
    "You are an adversarial literature auditor. Given a research claim, generate "
    "diverse search queries to find papers that may ALREADY have implemented or "
    "covered it. You are deliberately broader than a novelty checker."
)


class AuditQueries(BaseModel):
    queries: list[str] = Field(min_length=4, max_length=8)


SUPER_AUDIT_USER_PROMPT = """Research topic: {topic}

Claim to adversarially audit (does prior art already cover this?):
\"\"\"{claim}\"\"\"

Generate 4-8 search queries covering ALL of these angles:
1. exact implementations of the same mechanism for the same purpose;
2. close method neighbours (same technique, different framing);
3. adjacent domains where this mechanism may have been published first;
4. survey / benchmark style queries that would list the relevant prior work.

Queries must be standalone search strings (no context needed). Do not repeat
the same wording across queries."""


def _iter_targets(run_dirs: list[Path]):
    """Yield (system, topic_id, topic, claim, target_type) audit targets.

    - every proposed idea from every system (target_type=idea)
    - every surviving gap of full_v2 (target_type=gap) — the direct object of
      the false-open-gap metric.
    """
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        system = "unknown"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            system = str(cfg.get("system") or "unknown")
        pred_path = run_dir / "predictions.jsonl"
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("parse_status") not in (None, "ok"):
                    continue
                if rec.get("decision") == "propose_idea" and rec.get("idea"):
                    idea = rec["idea"]
                    claim = idea.get("research_question") or idea.get("title") or ""
                    yield system, rec["topic_id"], rec.get("topic", ""), str(claim), "idea"
        gaps_path = run_dir / "gaps_export.jsonl"
        if gaps_path.exists():
            for line in gaps_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for gap in rec.get("gaps") or []:
                    if gap.get("status") == "surviving" and gap.get("claimed_delta"):
                        yield "full_v2", rec["topic_id"], "", str(gap["claimed_delta"]), "gap"


async def gen_queries(llm, topic: str, claim: str, stats: CallStats) -> AuditQueries:
    from eval.common import chat_json

    return await chat_json(
        llm,
        [
            {"role": "system", "content": SUPER_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": SUPER_AUDIT_USER_PROMPT.format(
                topic=topic or "(unspecified)", claim=claim)},
        ],
        AuditQueries,
        temperature=0.0,   # audit-side determinism; NOT the generation temperature
        stats=stats,
    )


async def search_candidates(queries: list[str]) -> list[dict]:
    """Multi-source retrieval with per-source failure isolation."""
    from app.paper_sources.arxiv import ArxivSource
    from app.paper_sources.openalex import OpenAlexSource
    from app.paper_sources.semantic_scholar import SemanticScholarSource

    sources = [("openalex", OpenAlexSource()),
               ("semantic_scholar", SemanticScholarSource()),
               ("arxiv", ArxivSource())]
    seen_titles: set[str] = set()
    candidates: list[dict] = []
    for source_name, source in sources:
        for query in queries:
            try:
                papers = await source.search(query, limit=PER_QUERY_PER_SOURCE)
            except Exception as exc:  # source-level isolation: 429s are expected
                print(f"    [super_audit] {source_name} failed on "
                      f"{query[:40]!r}: {type(exc).__name__}: {exc}")
                continue
            for paper in papers:
                title = (getattr(paper, "title", None) or "").strip()
                if not title:
                    continue
                key = " ".join(title.lower().split())
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                candidates.append({
                    "title": title,
                    "year": getattr(paper, "year", None),
                    "venue": getattr(paper, "venue", None) or None,
                    "url": getattr(paper, "url", None) or None,
                    "source": source_name,
                    "query": query,
                })
    return candidates[:CANDIDATES_PER_TARGET]


def human_review_row(target: dict, candidates: list[dict]) -> str:
    lines = [
        f"## [{target['target_type']}] {target['system']} / {target['topic_id']}",
        "",
        f"Claim: {target['claim']}",
        "",
        "Candidate prior art (mark each: FULL / PARTIAL / NONE coverage of the claim):",
        "",
    ]
    if not candidates:
        lines.append("- (no candidates found — control sample: please judge the "
                     "claim's novelty yourself)")
    for idx, cand in enumerate(candidates, 1):
        lines.append(
            f"- [ ] {idx}. {cand['title']} ({cand.get('year') or 'n.d.'}, "
            f"{cand.get('venue') or cand['source']}) — verdict: ____")
    lines.append("")
    lines.append("Overall verdict for this claim: [ ] false-open (already covered) "
                 "/ [ ] genuinely open")
    lines.append("")
    return "\n".join(lines)


async def _run(args) -> None:
    from app.llm.factory import get_llm

    run_dirs = [Path(p) for p in args.systems]
    for run_dir in run_dirs:
        if not (run_dir / "predictions.jsonl").exists():
            raise FileNotFoundError(f"{run_dir} has no predictions.jsonl")

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_audit")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="super_audit", mode=MODE, split="topics_v1",
        sample_count=len(run_dirs), seed=None, limit=None,
        extra={
            "systems": [str(p.resolve()) for p in run_dirs],
            "query_generation_temperature": 0.0,
            "candidates_per_target": CANDIDATES_PER_TARGET,
            "independence": "different prompt template + multi-source retrieval; "
                            "production query family NOT reused; FULL/PARTIAL/NONE "
                            "verdict reserved for human review",
            "citation_snowball": "TODO v1.1",
        },
    )
    run.write_config(cfg, overwrite=True)

    llm = get_llm()
    targets = list(_iter_targets(run_dirs))
    print(f"[super_audit] {len(targets)} audit targets "
          f"({sum(1 for t in targets if t[4] == 'idea')} ideas, "
          f"{sum(1 for t in targets if t[4] == 'gap')} surviving gaps)")

    candidates_path = run.dir / "candidate_killer_papers.jsonl"
    review_md = ["# Super Audit — human review sheet",
                 "",
                 "Mark each candidate FULL / PARTIAL / NONE; then the overall verdict.",
                 "FULL = the candidate already implements the claimed mechanism+purpose.",
                 ""]
    for idx, (system, topic_id, topic, claim, target_type) in enumerate(targets):
        stats = CallStats()
        print(f"[{idx+1}/{len(targets)}] ({target_type}) {system}/{topic_id}")
        try:
            queries = await gen_queries(llm, topic, claim, stats)
            candidates = await search_candidates(queries.queries)
        except Exception as exc:  # per-target isolation
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            candidates = []
            queries = AuditQueries(queries=[])
        target_record = {
            "system": system, "topic_id": topic_id, "topic": topic,
            "target_type": target_type, "claim": claim,
            "queries": queries.queries,
            "candidate_papers": candidates,
            "llm": stats.as_dict(),
        }
        with candidates_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(target_record, ensure_ascii=False, default=str) + "\n")
        review_md.append(human_review_row(target_record, candidates))
    (run.dir / "human_review.md").write_text("\n".join(review_md), encoding="utf-8")
    print(f"[super_audit] run dir: {run.dir}")
    print(f"[super_audit] next: fill human_review.md verdicts, then run evaluate.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e super audit")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--systems", nargs="+", required=True,
                        help="run dirs of the three systems (predictions.jsonl required)")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

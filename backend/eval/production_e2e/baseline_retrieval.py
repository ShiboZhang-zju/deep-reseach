"""production_e2e_v1 — Baseline B: Retrieval + LLM (same literature supply).

Design point 1 (frozen in README): Baseline B does NOT run its own retrieval.
It consumes the paper export of the Full V2 run (papers_export.jsonl) so the
comparison isolates the value of Evidence + Gap Audit UNDER THE SAME retrieved
literature. self-retrieval variant is deferred to v2.

Usage:
    cd backend
    python -m eval.production_e2e.baseline_retrieval --run-id pe2e_v1_retellm \
        --v2-run-dir ../eval_results/pe2e_v1_fullv2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.common import CallStats, EvalRun, run_samples
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.baseline_direct import (
    ABSTAIN_RULE,
    IDEA_REQUIREMENTS,
    GENERATION_TEMPERATURE,
    load_topics,
    strata_counts,
)
from eval.production_e2e.schema import E2EDecision, build_prediction_record

BENCHMARK = "production_e2e"
MODE = "baseline_retrieval"
TOP_K_PAPERS = 20           # frozen: literature budget for Baseline B
ABSTRACT_MAX_CHARS = 800    # prompt-size bound per paper

BASELINE_B_SYSTEM_PROMPT = (
    "You are a careful research scientist. You are given the list of papers a "
    "retrieval system found for the research topic. Output exactly ONE final "
    "research idea grounded in this literature, or honestly abstain."
)

BASELINE_B_USER_PROMPT = """Research topic: {topic}

Below is the list of papers retrieved for this topic (title / year / venue / abstract).
This is ALL the information you have beyond your general knowledge.

--- Retrieved papers ({n} shown) ---
{papers_block}
--- End of retrieved papers ---

Output your final decision for this topic.

{requirements}

Additional grounding rules:
- Anchor the idea in the retrieved literature: the supporting_rationale must name the
  concrete papers (by title) it builds on and the limitation or gap it exploits.
- If the retrieved literature is too thin, off-topic, or does not support a trustworthy
  idea, choose abstain.

{abstain_rule}
"""


def format_paper(idx: int, paper: dict) -> str:
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) > ABSTRACT_MAX_CHARS:
        abstract = abstract[:ABSTRACT_MAX_CHARS] + " [...]"
    year = paper.get("year") or "n.d."
    venue = (paper.get("venue") or "").strip() or "unknown venue"
    title = (paper.get("title") or "").strip()
    return f"[{idx}] {title} ({year}, {venue})\nAbstract: {abstract or '(no abstract available)'}"


def build_papers_block(papers: list[dict]) -> str:
    return "\n\n".join(format_paper(i + 1, p) for i, p in enumerate(papers))


def load_v2_papers_export(v2_run_dir: Path) -> dict[str, list[dict]]:
    """topic_id -> papers sorted by final_score desc (already the export order)."""
    export_path = v2_run_dir / "papers_export.jsonl"
    if not export_path.exists():
        raise FileNotFoundError(
            f"papers_export.jsonl not found in {v2_run_dir} — run run_full_v2 first "
            "(Baseline B consumes the SAME literature the Full V2 pipeline retrieved)")
    mapping: dict[str, list[dict]] = {}
    for line in export_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        mapping[record["topic_id"]] = record.get("papers") or []
    return mapping


async def run_one_topic(sample: dict, llm, v2_papers: dict[str, list[dict]]) -> dict:
    from eval.common import chat_json

    topic_id = sample["topic_id"]
    papers = [p for p in (v2_papers.get(topic_id) or [])
              if (p.get("title") or "").strip()]
    # Defensive re-sort by final_score desc, then cap at TOP_K_PAPERS.
    papers = sorted(papers, key=lambda p: (p.get("final_score") or 0.0), reverse=True)
    papers = papers[:TOP_K_PAPERS]

    stats = CallStats()
    decision = await chat_json(
        llm,
        [
            {"role": "system", "content": BASELINE_B_SYSTEM_PROMPT},
            {"role": "user", "content": BASELINE_B_USER_PROMPT.format(
                topic=sample["topic"], n=len(papers),
                papers_block=build_papers_block(papers),
                requirements=IDEA_REQUIREMENTS, abstain_rule=ABSTAIN_RULE)},
        ],
        E2EDecision,
        temperature=GENERATION_TEMPERATURE,
        stats=stats,
    )
    record = build_prediction_record(
        topic_id=topic_id,
        stratum=sample["stratum"],
        topic=sample["topic"],
        system="retrieval_llm",
        decision=decision,
        extra={
            "decision_raw": decision.model_dump(),
            "papers_consumed": len(papers),
            "papers_available": len(v2_papers.get(topic_id) or []),
            "top_k_papers": TOP_K_PAPERS,
        },
    )
    record.update(stats.as_dict())
    return record


async def _run(args) -> None:
    from app.llm.factory import get_llm

    topics = load_topics()
    if args.limit:
        topics = topics[: args.limit]

    v2_run_dir = Path(args.v2_run_dir)
    v2_papers = load_v2_papers_export(v2_run_dir)

    missing = [t["topic_id"] for t in topics if not v2_papers.get(t["topic_id"])]
    runnable = [t for t in topics if v2_papers.get(t["topic_id"])]
    if missing:
        print(f"[baseline_retrieval] WARNING: {len(missing)} topics have no V2 paper "
              f"export and are SKIPPED (need run_full_v2 to cover them): {missing}")

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_retellm")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="idea_e2e", mode=MODE, split="topics_v1",
        sample_count=len(runnable), seed=None, limit=args.limit,
        extra={
            "system": "retrieval_llm",
            "generation_temperature": GENERATION_TEMPERATURE,
            "top_k_papers": TOP_K_PAPERS,
            "v2_run_dir": str(v2_run_dir.resolve()),
            "strata": strata_counts(runnable),
            "fairness": "same provider/temperature/schema; consumes the SAME papers "
                        "the Full V2 run retrieved (design point 1); retrieval cost "
                        "attributed to Full V2, not to this baseline",
            "topics_skipped_no_export": missing,
        },
    )
    run.write_config(cfg, overwrite=True)

    llm = get_llm()

    async def run_one(sample: dict) -> dict:
        return await run_one_topic(sample, llm, v2_papers)

    await run_samples(runnable, run_one, run, resume=args.resume)
    print(f"[baseline_retrieval] run dir: {run.dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e Baseline B (Retrieval + LLM)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--v2-run-dir", required=True,
                        help="eval_results run dir of the Full V2 run (papers_export.jsonl)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

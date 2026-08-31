"""production_e2e_v1 — Baseline A: Direct LLM (zero information).

One structured LLM call per topic, same provider / temperature (0.3, matching
the production idea-generation chain) / output schema as the other two systems.
The only variable vs Full V2 is information and workflow: no retrieval, no
evidence, no audit.

Usage:
    cd backend
    python -m eval.production_e2e.baseline_direct --run-id pe2e_v1_direct
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.common import CallStats, EvalRun, run_samples
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.schema import SYSTEM_NAMES, E2EDecision, build_prediction_record

BENCHMARK = "production_e2e"
MODE = "baseline_direct"
GENERATION_TEMPERATURE = 0.3  # frozen: equals production idea-chain chat_json default

TOPICS_PATH = Path(__file__).resolve().parent / "topics.jsonl"

# Shared instruction block; Baseline B reuses it and prepends the literature.
IDEA_REQUIREMENTS = """Requirements for the idea:
- research_question: ONE falsifiable claim this idea would test (not a goal statement, not a topic restatement).
- method_sketch: the concrete mechanism — what is built/changed and why it should work (not a list of techniques).
- experiment_outline: a minimal executable experiment — dataset or data source, baselines, metric, protocol.
- The idea must NOT be a mere "combine area A with area B" bridge; name the specific mechanism or evidence gap it exploits.
- Do not invent datasets, numbers, or prior results you are not confident exist."""

ABSTAIN_RULE = """Abstaining is a valid scientific outcome, not a failure:
- If your knowledge of this area is insufficient to ground a trustworthy idea (you cannot name concrete prior work, methods, or evaluation practices), choose abstain and give an honest abstain_reason.
- Do not propose an idea just to have one; an ungrounded idea is worse than abstention."""

BASELINE_A_SYSTEM_PROMPT = (
    "You are a careful research scientist. For the given research topic you output "
    "exactly ONE final research idea, or you honestly abstain. Follow the schema strictly."
)

BASELINE_A_USER_PROMPT = """Research topic: {topic}

Based ONLY on your internal knowledge of the research literature, output your final
decision for this topic.

{requirements}

{abstain_rule}
"""


def load_topics() -> list[dict]:
    topics: list[dict] = []
    for line in TOPICS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            topics.append(json.loads(line))
    return topics


def strata_counts(topics: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in topics:
        counts[t["stratum"]] = counts.get(t["stratum"], 0) + 1
    return counts


async def run_one_topic(sample: dict, llm) -> dict:
    """One Direct-LLM call per topic; failure granularity = one topic."""
    from eval.common import chat_json

    stats = CallStats()
    decision = await chat_json(
        llm,
        [
            {"role": "system", "content": BASELINE_A_SYSTEM_PROMPT},
            {"role": "user", "content": BASELINE_A_USER_PROMPT.format(
                topic=sample["topic"], requirements=IDEA_REQUIREMENTS,
                abstain_rule=ABSTAIN_RULE)},
        ],
        E2EDecision,
        temperature=GENERATION_TEMPERATURE,
        stats=stats,
    )
    record = build_prediction_record(
        topic_id=sample["topic_id"],
        stratum=sample["stratum"],
        topic=sample["topic"],
        system="direct_llm",
        decision=decision,
        extra={"decision_raw": decision.model_dump()},
    )
    record.update(stats.as_dict())
    return record


async def _run(args) -> None:
    from app.llm.factory import get_llm

    topics = load_topics()
    if args.limit:
        topics = topics[: args.limit]

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_direct")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="idea_e2e", mode=MODE, split="topics_v1",
        sample_count=len(topics), seed=None, limit=args.limit,
        extra={
            "system": "direct_llm",
            "generation_temperature": GENERATION_TEMPERATURE,
            "strata": strata_counts(topics),
            "fairness": "same provider/temperature/schema as retrieval_llm and full_v2; "
                        "no retrieval, no evidence, no audit",
        },
    )
    run.write_config(cfg, overwrite=True)

    llm = get_llm()

    async def run_one(sample: dict) -> dict:
        return await run_one_topic(sample, llm)

    await run_samples(topics, run_one, run, resume=args.resume)
    print(f"[baseline_direct] run dir: {run.dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e Baseline A (Direct LLM)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

"""production_e2e_v1 — Independent Evaluation Procedure (design points 3 & 4).

Terminology note (frozen): this is an *independent evaluation procedure*, NOT an
independent retrieval corpus — OpenAlex / Semantic Scholar / arXiv overlap with
the production data sources. What makes it independent enough for v1:
  - a DIFFERENT prompt template (wider query families: exact / paraphrase /
    adjacent-domain / survey-style),
  - a fresh multi-source retrieval pass,
  - human prior-art adjudication of the machine-found candidates.
The production Gap Audit must NOT be re-run and called independent validation:
it would share query family, retrieval sources, model preference and retrieval
blind spots with the system under test — both could miss the same killer paper.
Citation snowballing is deferred until the pilot shows the miss rate.

Blind review (design point 4): the human review sheet exposes ONLY
research topic / claim / candidate prior art. System identity (A/B/C),
target_type, topic stratum, V2 internal scores and production audit verdicts
are hidden behind random submission_ids; mapping.json restores identities
only after review. Order is shuffled.

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
import random
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from eval.common import CallStats, EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config

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
                if rec.get("protocol_flag"):
                    continue  # protocol violations are not audited outcomes
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


def blind_review_section(submission_id: str, topic: str, claim: str,
                         candidates: list[dict]) -> str:
    """Blind sheet: NO system identity, NO target_type, NO internal scores."""
    lines = [
        f"## Submission {submission_id}",
        "",
        f"Research topic: {topic or '(unspecified)'}",
        "",
        f"Claim under audit: {claim}",
        "",
        "For each candidate below, mark whether it already covers the claim:",
        "FULL (implements the same mechanism for the same purpose) / "
        "PARTIAL (overlapping but not the same) / NONE.",
        "",
    ]
    if not candidates:
        lines.append("- (no candidates found — control sample: judge from your "
                     "own knowledge)")
    for idx, cand in enumerate(candidates, 1):
        lines.append(
            f"- [ ] {idx}. {cand['title']} ({cand.get('year') or 'n.d.'}, "
            f"{cand.get('venue') or cand['source']}) — FULL / PARTIAL / NONE: ____")
    lines += [
        "",
        "Overall: does prior art already cover this claim (false-open)? [ ] yes [ ] no",
        "Novelty of the claim vs the nearest prior art (1-5): ____",
        "Feasibility of testing this claim as a research direction (1-5): ____",
        "Credible as a research idea/direction? [ ] yes [ ] no",
        "",
        "---",
        "",
    ]
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
            "independence": "independent evaluation procedure (NOT an independent "
                            "retrieval corpus: sources overlap with production); "
                            "different prompt template + fresh multi-source "
                            "retrieval + human prior-art adjudication; production "
                            "query family NOT reused",
            "citation_snowball": "deferred until pilot shows the miss rate",
            "blind_review": {
                "shuffle_seed": args.shuffle_seed,
                "hidden_fields": ["system", "target_type", "stratum",
                                  "internal scores", "production audit verdicts"],
                "protocol_flagged_samples": "excluded from audit",
            },
        },
    )
    run.write_config(cfg, overwrite=True)

    llm = get_llm()
    targets = list(_iter_targets(run_dirs))
    # Blind protocol: shuffle order, hide identities behind submission_ids.
    random.Random(args.shuffle_seed).shuffle(targets)
    submissions = []
    for system, topic_id, topic, claim, target_type in targets:
        submissions.append({
            "submission_id": uuid.uuid4().hex[:12],
            "system": system,
            "topic_id": topic_id,
            "target_type": target_type,
            "claim": claim,
            "_topic": topic,
        })
    print(f"[super_audit] {len(submissions)} audit targets "
          f"({sum(1 for s in submissions if s['target_type'] == 'idea')} ideas, "
          f"{sum(1 for s in submissions if s['target_type'] == 'gap')} surviving gaps), "
          f"shuffled with seed={args.shuffle_seed}")

    candidates_path = run.dir / "candidate_killer_papers.jsonl"
    review_md = [
        "# Super Audit — BLIND human review sheet",
        "",
        "Review each submission WITHOUT knowing which system produced it.",
        "Fill every field; identities are restored automatically afterwards.",
        "",
        "",
    ]
    template_path = run.dir / "human_verdicts.jsonl"
    for idx, sub in enumerate(submissions):
        stats = CallStats()
        print(f"[{idx+1}/{len(submissions)}] {sub['submission_id']}")
        try:
            queries = await gen_queries(llm, sub["_topic"], sub["claim"], stats)
            candidates = await search_candidates(queries.queries)
        except Exception as exc:  # per-target isolation
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            candidates = []
            queries = AuditQueries(queries=[])
        sub["queries"] = queries.queries
        sub["candidate_papers"] = candidates
        sub["llm"] = stats.as_dict()
        with candidates_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sub, ensure_ascii=False, default=str) + "\n")
        review_md.append(blind_review_section(
            sub["submission_id"], sub["_topic"], sub["claim"], candidates))
        # Fill-in template: reviewer edits the nulls, keyed by submission_id only.
        with template_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "submission_id": sub["submission_id"],
                "false_open": None,
                "novelty": None,
                "feasibility": None,
                "credible": None,
                "notes": "",
            }, ensure_ascii=False) + "\n")

    # Identity mapping: DO NOT share with the reviewer before review is done.
    mapping = {
        "shuffle_seed": args.shuffle_seed,
        "submissions": [
            {k: s[k] for k in ("submission_id", "system", "topic_id",
                               "target_type", "claim")}
            for s in submissions
        ],
    }
    (run.dir / "submission_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (run.dir / "human_review_blind.md").write_text("\n".join(review_md), encoding="utf-8")
    print(f"[super_audit] run dir: {run.dir}")
    print("[super_audit] next: fill human_verdicts.jsonl from human_review_blind.md "
          "(keyed by submission_id), then run evaluate.py --audit-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e super audit")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--systems", nargs="+", required=True,
                        help="run dirs of the three systems (predictions.jsonl required)")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

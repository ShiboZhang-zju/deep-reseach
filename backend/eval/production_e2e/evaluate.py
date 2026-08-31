"""production_e2e_v1 — headline table aggregation.

Computes what the harness can compute automatically (abstention, idea counts,
cost) and merges human verdicts (human_verdicts.jsonl, filled from
super_audit's human_review.md) into the frozen headline table:

| System | False-open-gap ↓ | Credible Idea Yield ↑ | Novelty ↑ | Feasibility ↑ | Abstention | Cost |

Dual-headline rule (design point 2): False-open-gap rate alone is gameable by
abstaining on everything, so Credible Idea Yield is always reported next to it
and Abstention is reported separately. Cells without human verdicts yet show
"pending".

Usage:
    cd backend
    python -m eval.production_e2e.evaluate \
        --runs ../eval_results/pe2e_v1_direct ../eval_results/pe2e_v1_retellm \
               ../eval_results/pe2e_v1_fullv2 \
        --human ../eval_results/pe2e_v1_audit/human_verdicts.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.common import EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.baseline_direct import load_topics

BENCHMARK = "production_e2e"
MODE = "evaluate"

SYSTEM_LABELS = {
    "direct_llm": "Direct LLM",
    "retrieval_llm": "Retrieval + LLM",
    "full_v2": "Full V2",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_system_predictions(run_dir: Path) -> tuple[str, list[dict]]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    system = str(cfg.get("system") or run_dir.name)
    records = [r for r in read_jsonl(run_dir / "predictions.jsonl")
               if r.get("parse_status") in (None, "ok")]
    return system, records


def system_cost_summary(records: list[dict]) -> str:
    llm_calls = sum(int(r.get("llm_calls") or 0) for r in records)
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in records)
    wall = sum(int(r.get("wall_clock_s") or 0) for r in records)
    papers = sum(int(r.get("papers_count") or 0) for r in records)
    parts = []
    if llm_calls:
        parts.append(f"{llm_calls} calls/{total_tokens/1000:.0f}k tok")
    if wall:
        parts.append(f"{wall/3600:.1f}h wall")
    if papers:
        parts.append(f"{papers} papers")
    return "; ".join(parts) if parts else "n/a"


def abstention_stats(records: list[dict], strata: list[str] | None = None) -> dict:
    subset = records
    if strata:
        subset = [r for r in records if r.get("stratum") in strata]
    total = len(subset)
    abstained = sum(1 for r in subset if r.get("decision") == "abstain")
    return {
        "topics": total,
        "abstained": abstained,
        "abstention_rate": round(abstained / total, 4) if total else None,
    }


def _fmt_pct(x: float | None) -> str:
    return "pending" if x is None else f"{x * 100:.1f}%"


def _fmt_score(x: float | None) -> str:
    return "pending" if x is None else f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="production_e2e headline aggregation")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--human", default=None,
                        help="human_verdicts.jsonl (filled from super_audit human_review.md)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    topics = load_topics()
    total_topics = len(topics)

    verdicts = read_jsonl(Path(args.human)) if args.human else []
    # key: (target_type, system, topic_id) -> verdict record
    verdict_map = {
        (str(v.get("target_type")), str(v.get("system")), str(v.get("topic_id"))): v
        for v in verdicts
    }

    systems: dict[str, list[dict]] = {}
    for run_path in args.runs:
        system, records = load_system_predictions(Path(run_path))
        systems[system] = records

    metrics: dict[str, dict] = {}
    for system, records in systems.items():
        idea_records = [r for r in records if r.get("decision") == "propose_idea"]
        # --- human-verdict-derived metrics (pending until verdicts exist) ---
        idea_verdicts = [verdict_map.get(("idea", system, r["topic_id"]))
                         for r in idea_records]
        idea_verdicts = [v for v in idea_verdicts if v]
        gap_verdicts = [v for (t, s, _), v in verdict_map.items()
                        if t == "gap" and s == system and v]

        false_open_ideas = sum(1 for v in idea_verdicts
                               if v.get("verdict") == "false_open")
        gap_total = len(gap_verdicts)
        gap_false_open = sum(1 for v in gap_verdicts
                             if v.get("verdict") == "false_open")
        credible_topics = {
            r["topic_id"] for r in idea_records
            if (verdict_map.get(("idea", system, r["topic_id"])) or {}).get("idea_credible")
        }
        novelty_scores = [float(v["novelty"]) for v in idea_verdicts if v.get("novelty")]
        feasibility_scores = [float(v["feasibility"]) for v in idea_verdicts if v.get("feasibility")]

        abst = abstention_stats(records)
        metrics[system] = {
            "topics_covered": abst["topics"],
            "ideas_proposed": len(idea_records),
            "abstention": abst,
            "abstention_by_stratum": {
                s: abstention_stats(records, [s]) for s in
                ("narrow_mature", "emerging_sparse", "broad_ambiguous")
            },
            "false_open_gap": {
                "idea_level": {
                    "verdicted": len(idea_verdicts),
                    "false_open": false_open_ideas,
                    "rate": round(false_open_ideas / len(idea_verdicts), 4)
                    if idea_verdicts else None,
                },
                "gap_level_full_v2_only": {
                    "verdicted": gap_total,
                    "false_open": gap_false_open,
                    "rate": round(gap_false_open / gap_total, 4) if gap_total else None,
                },
            },
            "credible_idea_yield": {
                "credible_topics": len(credible_topics),
                "total_topics": total_topics,
                "rate": round(len(credible_topics) / total_topics, 4)
                if total_topics else None,
            },
            "novelty_mean": round(sum(novelty_scores) / len(novelty_scores), 3)
            if novelty_scores else None,
            "feasibility_mean": round(sum(feasibility_scores) / len(feasibility_scores), 3)
            if feasibility_scores else None,
            "cost": system_cost_summary(records),
        }

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_eval")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="headline", mode=MODE, split="topics_v1",
        sample_count=total_topics, seed=None, limit=None,
        extra={"runs": args.runs, "human_verdicts": args.human,
               "headline": "False-open-gap + Credible Idea Yield (dual, frozen)"},
    )
    run.write_config(cfg, overwrite=True)
    run.write_metrics(metrics)

    # ---------------- summary.md: the frozen six-column table ----------------
    lines = [
        "# production_e2e_v1 — headline table",
        "",
        f"Topics: {total_topics} (12 narrow_mature / 6 emerging_sparse / 6 broad_ambiguous)",
        "",
        "| System | False-open-gap ↓ | Credible Idea Yield ↑ | Novelty ↑ | Feasibility ↑ | Abstention | Cost |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    order = ["direct_llm", "retrieval_llm", "full_v2"]
    for system in order + [s for s in systems if s not in order]:
        if system not in systems:
            continue
        m = metrics[system]
        lines.append(
            "| {label} | {fog} | {ciy} | {nov} | {fea} | {abst} | {cost} |".format(
                label=SYSTEM_LABELS.get(system, system),
                fog=_fmt_pct(m["false_open_gap"]["idea_level"]["rate"]),
                ciy=_fmt_pct(m["credible_idea_yield"]["rate"]),
                nov=_fmt_score(m["novelty_mean"]),
                fea=_fmt_score(m["feasibility_mean"]),
                abst=_fmt_pct(m["abstention"]["abstention_rate"]),
                cost=m["cost"],
            ))
    lines += [
        "",
        "Notes:",
        "- False-open-gap: idea-level = proposed ideas whose claimed contribution is "
        "already covered by prior art (human verdict from super_audit); gap-level "
        "(full_v2 surviving gaps) in metrics.json.",
        "- Credible Idea Yield: topics with >=1 idea judged credible / all topics. "
        "Reported together with False-open-gap by design (dual headline, "
        "abstain-everything must not win).",
        "- Novelty/Feasibility: human 1-5 means over proposed ideas.",
        "- Cost: baselines = LLM calls/tokens; Full V2 = wall-clock + papers "
        "(full-chain vs single-call; reported separately, never merged).",
        "- 'pending' = waiting for human verdicts (fill human_verdicts.jsonl).",
        "",
        "Decision rule (pre-registered in README): gains from gap correctness -> "
        "strengthen NPA/Audit; novelty weak -> fix retrieval; feasibility weak -> "
        "fix Intervention/Experiment; no significant edge over Retrieval+LLM -> "
        "REMOVE mechanisms, do not add more.",
        "",
    ]
    run.write_summary("\n".join(lines))
    print("\n".join(lines))
    print(f"[evaluate] run dir: {run.dir}")


if __name__ == "__main__":
    main()

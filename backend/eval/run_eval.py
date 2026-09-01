"""Unified eval CLI.

Examples:
    python -m eval.run_eval researchbench --task retrieval --split tiny
    python -m eval.run_eval researchbench --task generation --split tiny
    python -m eval.run_eval researchbench --task ranking --split tiny --limit 3
    python -m eval.run_eval rinobench --split test --limit 20
    python -m eval.run_eval internal
    python -m eval.run_eval compare eval_results/<run_a> eval_results/<run_b>

Outputs per run (backend/eval_results/<run_id>/):
    config.json       benchmark/task/split/mode/sample_count/model/provider/
                      policy versions/git commit/timestamp/seed/limit
    predictions.jsonl per-sample: sample_id, prediction, gold, raw_output,
                      parse_status, latency_ms, error, llm call/token stats
    metrics.json      official scorer output preserved verbatim (+ provenance)
    summary.md        human-readable summary

Resume: pass --output-dir (or --run-id) of an existing run dir; sample_ids
already in predictions.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.common import (
    CallStats,
    EvalRun,
    aggregate_stats,
    flatten_numbers,
    run_samples,
)
from eval.config import DEFAULT_RESULTS_DIR, build_run_config


# --------------------------------------------------------------------------
# ResearchBench
# --------------------------------------------------------------------------

async def _run_researchbench(args) -> None:
    tasks = (["retrieval", "generation", "ranking"]
             if args.task == "all" else [args.task])
    for task in tasks:
        await _run_researchbench_task(args, task)


async def _run_researchbench_task(args, task: str) -> None:
    from app.llm.factory import get_llm
    from eval.researchbench import adapter as rb

    samples = rb.load_samples(task, split=args.split, limit=args.limit, seed=args.seed)
    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix=f"rb_{task}")
    cfg = build_run_config(
        benchmark=rb.BENCHMARK, task=task, mode=rb.MODE, split=args.split,
        sample_count=len(samples), seed=args.seed, limit=args.limit,
    )
    run.write_config(cfg)

    llm = get_llm()
    runner = rb.RUNNERS[task]
    if task == "ranking":
        ranking_kwargs = {
            "ranking_policy": (rb.RANKING_POLICY_V2 if args.ranking_policy == "v2"
                               else rb.RANKING_POLICY_V1)}
        cfg.update({"ranking_policy": ranking_kwargs["ranking_policy"]})
        run.write_config(cfg, overwrite=True)
    else:
        ranking_kwargs = {}

    async def run_one(sample: dict) -> dict:
        stats = CallStats()
        out = await runner(sample, llm, stats, **ranking_kwargs)
        record = {
            "sample_id": str(sample.get("sample_id") or sample.get("source") or ""),
            "prediction": out["prediction"],
            "gold": out.get("gold"),
            "raw_output": None,
            "parse_status": "ok",
            "error": None,
            **stats.as_dict(),
        }
        if out.get("eval_extra"):
            record["extra"] = out["eval_extra"]
        return record

    resume = bool(args.output_dir or args.run_id)
    records = await run_samples(samples, run_one, run, resume=resume)

    # One sample -> one final prediction (latest successful attempt wins).
    final_by_id: dict[str, dict] = {}
    for record in records:
        if record.get("parse_status") == "ok" and record.get("prediction"):
            final_by_id[str(record.get("sample_id"))] = record
    ok_records = list(final_by_id.values())
    metrics = {
        "benchmark": rb.BENCHMARK,
        "task": task,
        "mode": rb.MODE,
        "model": cfg["model"],
        "model_provider": cfg["model_provider"],
        "git_commit": cfg["git_commit"],
    }
    if ok_records:
        by_id = {sample["sample_id"]: sample for sample in samples}
        data_records = [by_id[r["sample_id"]] for r in ok_records]
        predictions = [r["prediction"] for r in ok_records]
        metrics.update(rb.score_official(task, predictions, data_records, llm))
    else:
        metrics["official_scorer_output"] = None
        metrics["note"] = "no successful predictions; official scorer not run"
    if task == "generation":
        from eval.config import judge_policy
        metrics["judge_policy"] = judge_policy()
    metrics["run_stats"] = aggregate_stats(records)
    run.write_metrics(metrics)
    run.write_summary(render_benchmark_summary(cfg, metrics))

    headline = _headline(metrics)
    stats = metrics["run_stats"]
    print(f"ResearchBench {task} [{cfg['split']}] ok={stats['samples_ok']}/{stats['samples_total']}"
          f" parse_fail={stats['parse_failure_rate']:.1%} attempts={stats['attempts_total']}"
          f" llm_calls={stats['llm_calls']}"
          f" tokens={stats['total_tokens']}"
          f" | {headline}\n  -> {run.dir}")


def _headline(metrics: dict) -> str:
    output = metrics.get("official_scorer_output")
    if not isinstance(output, dict):
        return "no official metrics (no successful predictions)"
    summary = output.get("summary") if isinstance(output.get("summary"), dict) else output
    flat = flatten_numbers(summary)
    parts = []
    for key, value in list(flat.items())[:10]:
        parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts) if parts else "see metrics.json"


# --------------------------------------------------------------------------
# RINoBench
# --------------------------------------------------------------------------

async def _run_rinobench(args) -> None:
    from app.llm.factory import get_llm
    from eval.rinobench import adapter as rino

    samples = rino.load_samples(split=args.split, limit=args.limit, seed=args.seed)
    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="rinobench")
    cfg = build_run_config(
        benchmark=rino.BENCHMARK, task=rino.TASK, mode=rino.MODE, split=args.split,
        sample_count=len(samples), seed=args.seed, limit=args.limit,
        extra={"novelty_policy": (rino.NOVELTY_POLICY_V3 if args.novelty_policy == "v3"
                                  else rino.NOVELTY_POLICY_V1)},
    )
    run.write_config(cfg)

    llm = get_llm()
    rubric = rino.load_rubric()

    async def run_one(sample: dict) -> dict:
        stats = CallStats()
        out = await rino.run_novelty_sample(sample, llm, stats, rubric=rubric,
                                            novelty_policy=args.novelty_policy)
        record = {
            "sample_id": sample["source"],
            "prediction": out["prediction"],
            "official_format_element": out["official_element"],
            "gold": out.get("gold"),
            "raw_output": None,
            "parse_status": "ok",
            "error": None,
            **stats.as_dict(),
        }
        return record

    resume = bool(args.output_dir or args.run_id)
    records = await run_samples(samples, run_one, run, resume=resume)

    # One sample -> one final prediction (latest successful attempt wins).
    final_by_id: dict[str, dict] = {}
    for record in records:
        if record.get("parse_status") == "ok" and record.get("prediction"):
            final_by_id[str(record.get("sample_id"))] = record
    ok_records = list(final_by_id.values())
    # Official-format prediction file: array of {reasoning, novelty_score} in
    # dataset order. Directly consumable by the official evaluator when the
    # FULL split was processed (the official script assumes complete coverage).
    official_elements = [r["official_format_element"] for r in ok_records]
    (run.dir / "official_prediction_format.json").write_text(
        json.dumps(official_elements, ensure_ascii=False, indent=4), encoding="utf-8")

    metrics = {
        "benchmark": rino.BENCHMARK,
        "task": rino.TASK,
        "mode": rino.MODE,
        "model": cfg["model"],
        "model_provider": cfg["model_provider"],
        "git_commit": cfg["git_commit"],
        "self_retrieval": False,
    }
    if ok_records:
        metrics.update(rino.build_metrics(ok_records))
        metrics["official_evaluator_ready"] = len(ok_records) == len(samples) and len(
            rino.load_samples(split=args.split)) == len(ok_records)
    else:
        metrics["official_metric_replica"] = None
        metrics["note"] = "no successful predictions; metrics not computed"
    metrics["run_stats"] = aggregate_stats(records)
    run.write_metrics(metrics)
    run.write_summary(render_benchmark_summary(cfg, metrics))

    replica = metrics.get("official_metric_replica") or {}
    stats = metrics["run_stats"]
    f1_macro = replica.get("f1_macro")
    if isinstance(f1_macro, (int, float)):
        headline = f"Macro-F1={f1_macro * 100:.1f}% MAE={replica.get('mean_absolute_error', 'n/a')}"
    else:
        headline = f"Macro-F1=n/a MAE={replica.get('mean_absolute_error', 'n/a')}"
    print(f"RINoBench {rino.TASK} [{cfg['split']}] ok={stats['samples_ok']}/{stats['samples_total']}"
          f" parse_fail={stats['parse_failure_rate']:.1%}"
          f" abstain={metrics.get('abstention_rate', 0):.1%}"
          f" llm_calls={stats['llm_calls']}"
          f" tokens={stats['total_tokens']}"
          f" | {headline}")
    print(f"  -> {run.dir}")


# --------------------------------------------------------------------------
# Internal regression
# --------------------------------------------------------------------------

def cmd_internal(args) -> None:
    from eval.internal import research_validity_regression as ivr

    result = ivr.run_validity_regression()
    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="internal")
    run.write_config(build_run_config(
        benchmark="internal", task="research_validity_regression",
        mode="pytest_aggregate", split="n/a",
        sample_count=result["totals"]["total_tests"],
    ))
    run.write_metrics(result)
    run.write_summary(ivr.render_summary(result))
    totals = result["totals"]
    print(f"Internal validity regression: {totals['passed']}/{totals['total_tests']} PASS"
          f" (failed={totals['failed']} errors={totals['errors']})"
          f" | groups: {result['group_summary']['pass']}/{result['group_summary']['known_groups']}"
          f"\n  -> {run.dir}")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def _load_run(run_dir: Path) -> tuple[dict, dict]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return config, metrics


def _fmt_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def cmd_compare(args) -> None:
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    config_a, metrics_a = _load_run(run_a)
    config_b, metrics_b = _load_run(run_b)

    print(f"run A: {run_a.name}  ({config_a.get('benchmark')} {config_a.get('task')}"
          f" model={config_a.get('model')} commit={config_a.get('git_commit')}"
          f" {config_a.get('timestamp_utc')})")
    print(f"run B: {run_b.name}  ({config_b.get('benchmark')} {config_b.get('task')}"
          f" model={config_b.get('model')} commit={config_b.get('git_commit')}"
          f" {config_b.get('timestamp_utc')})")
    print("")

    flat_a = flatten_numbers(metrics_a.get("official_scorer_output")
                             or metrics_a.get("official_metric_replica") or {})
    flat_b = flatten_numbers(metrics_b.get("official_scorer_output")
                             or metrics_b.get("official_metric_replica") or {})
    ordered_keys = list(dict.fromkeys(list(flat_a) + list(flat_b)))
    if ordered_keys:
        print(f"{'metric':<64}{'run A':>12}{'run B':>12}")
        for key in ordered_keys:
            left = _fmt_value(flat_a[key]) if key in flat_a else "n/a"
            right = _fmt_value(flat_b[key]) if key in flat_b else "n/a"
            print(f"{key:<64}{left:>12}{right:>12}")
    else:
        print("(no official metrics found in either run)")

    groups_a = metrics_a.get("groups")
    groups_b = metrics_b.get("groups")
    if groups_a or groups_b:
        print("")
        print("Validity Regression")
        for run_name in ["run7", "run8", "run10", "run12"]:
            ga = (groups_a or {}).get(run_name) or {}
            gb = (groups_b or {}).get(run_name) or {}
            left = f"{ga.get('status', 'n/a')} ({ga.get('passed', 0)}/{ga.get('total', 0)})"
            right = f"{gb.get('status', 'n/a')} ({gb.get('passed', 0)}/{gb.get('total', 0)})"
            print(f"  {run_name:<10} {left:<28} -> {right}")


# --------------------------------------------------------------------------
# Summary rendering
# --------------------------------------------------------------------------

def render_benchmark_summary(cfg: dict, metrics: dict) -> str:
    stats = metrics.get("run_stats", {})
    official_output = metrics.get("official_scorer_output")
    replica = metrics.get("official_metric_replica")
    lines = [
        f"# Eval summary — {cfg['benchmark']} {cfg['task']}",
        "",
        f"- mode: **{cfg['mode']}**",
        f"- model: {cfg['model']} ({cfg['model_provider']})",
        f"- git commit: {cfg['git_commit']} | run: {cfg['timestamp_utc']}",
        f"- split: {cfg['split']} | seed: {cfg['seed']} | limit: {cfg['limit']} | samples: {cfg['sample_count']}",
        f"- policy versions: `{json.dumps(cfg['policy_versions'])}`",
        "",
        "## Counts",
        f"- ok: {stats.get('samples_ok', 0)} / error: {stats.get('samples_error', 0)}"
        f" / total: {stats.get('samples_total', 0)}",
        f"- llm calls: {stats.get('llm_calls', 0)} | tokens (prompt/completion/total): "
        f"{stats.get('prompt_tokens', 0)}/{stats.get('completion_tokens', 0)}/{stats.get('total_tokens', 0)}"
        f" | llm latency: {stats.get('llm_latency_ms', 0)} ms",
        "",
    ]
    if official_output is not None:
        lines += ["## Official metrics (verbatim)", "```json",
                  json.dumps(official_output, ensure_ascii=False, indent=2), "```", ""]
    if replica is not None:
        lines += ["## Official-metric replica (score-based part)", "```json",
                  json.dumps(replica, ensure_ascii=False, indent=2), "```", ""]
    note = metrics.get("note")
    if note:
        lines += [f"Note: {note}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Frozen-eval harness CLI (ResearchBench / RINoBench / internal regression).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--limit", type=int, default=None, help="first N samples (after optional shuffle)")
    common.add_argument("--seed", type=int, default=None, help="stable shuffle seed for sample selection")
    common.add_argument("--output-dir", default=None,
                        help="results dir to reuse (enables resume of an existing run)")
    common.add_argument("--run-id", default=None, help="explicit run id under the results dir")

    rb_parser = sub.add_parser("researchbench", parents=[common],
                               help="ResearchBench retrieval/generation/ranking (official-compatible)")
    rb_parser.add_argument("--task", choices=["retrieval", "generation", "ranking", "all"],
                           default="all")
    rb_parser.add_argument("--split", choices=["tiny", "full"], default="tiny")
    rb_parser.add_argument("--ranking-policy", choices=["v1", "v2"], default="v1",
                           help="ranking judgment policy: v1=pairwise quality, "
                                "v2=criterion-first (substance over packaging)")
    rb_parser.set_defaults(func=lambda args: asyncio.run(_run_researchbench(args)))

    rino_parser = sub.add_parser("rinobench", parents=[common],
                                 help="RINoBench novelty judgment (gold_related_works, no self retrieval)")
    rino_parser.add_argument("--split", default="test", help="dataset split file (train/test)")
    rino_parser.add_argument("--novelty-policy", choices=["v1", "v3"], default="v1",
                             help="novelty judgment policy: v1=holistic 1-5, "
                                  "v3=criterion-first (closest-work coverage -> residual delta -> score)")
    rino_parser.set_defaults(func=lambda args: asyncio.run(_run_rinobench(args)))

    internal_parser = sub.add_parser(
        "internal", help="aggregate the existing research-validity regression suite")
    internal_parser.add_argument("--output-dir", default=None)
    internal_parser.add_argument("--run-id", default=None)
    internal_parser.set_defaults(func=cmd_internal)

    compare_parser = sub.add_parser("compare", help="compare two eval run dirs")
    compare_parser.add_argument("run_a")
    compare_parser.add_argument("run_b")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

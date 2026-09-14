"""Re-drive topics whose previous attempt was contaminated by infrastructure.

Why this exists
---------------
``run_full_v2.py`` resume semantics are deliberately conservative: a topic is
skipped once its LATEST attempt has ``parse_status == "ok"``. That is the right
default — it stops a resume from re-burning budget on topics that already
produced a protocol-complete record.

But ``parse_status == "ok"`` only means "the driver finished its protocol
interaction". It says nothing about whether the *pipeline* succeeded. On the
2026-09-14 batch, 15 of 24 topics ended in ``failed`` / ``stopped_timeout``
(LLM gateway 502s, a 4h agent timeout, and the ``database is locked`` deaths
that the 2026-09-14 lock-retry contract fixes). Those topics carry a
``parse_status`` of ``ok`` with ``decision == "abstain"`` — an abstention
produced by an infrastructure fault, not a scientific judgement.

Feeding those rows into a "V2 abstains more, therefore it rejects false ideas"
headline would be wrong: the abstention is an artifact, not a result.

What this script does
---------------------
Re-drives a caller-specified subset into a SEPARATE run directory, reusing the
frozen driver's own functions (``run_one_topic``, ``to_prediction_record``,
``export_papers``, ``export_gaps``, ``_rewrite_predictions``) verbatim — no
evaluation logic is forked or reimplemented. The original run stays untouched as
the audit trail; the new run is a clean re-measurement of exactly the
contaminated subset.

Usage:
    cd backend
    python -m eval.production_e2e.rerun_infra_topics \\
        --source-run pe2e_v3_stable --run-id pe2e_v3_stable_clean \\
        --statuses failed stopped_timeout --concurrency 2
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.common import EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.baseline_direct import load_topics, strata_counts
from eval.production_e2e.run_full_v2 import (
    BENCHMARK,
    DEFAULT_API_BASE,
    DEFAULT_POLL_INTERVAL_S,
    MODE,
    ApiClient,
    _rewrite_predictions,
    export_gaps,
    export_papers,
    run_one_topic,
    to_prediction_record,
)

# Statuses that mean "the pipeline did not get a fair chance", as opposed to a
# genuine scientific outcome (abstained / more_research_required) or a success
# (waiting_for_user_review).
DEFAULT_INFRA_STATUSES = ("failed", "stopped_timeout")


def _latest_attempts(run_dir: Path) -> dict[str, dict]:
    """Latest attempt per sample_id, matching run_full_v2's resume rule."""
    latest: dict[str, dict] = {}
    attempts_path = run_dir / "attempts.jsonl"
    if not attempts_path.exists():
        return latest
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = str(rec.get("sample_id") or "")
        if sid:
            latest[sid] = rec
    return latest


def select_contaminated(run_dir: Path, statuses: tuple[str, ...]) -> list[str]:
    """Topic ids whose latest attempt ended in one of ``statuses``."""
    latest = _latest_attempts(run_dir)
    return sorted(
        sid for sid, rec in latest.items()
        if str(rec.get("final_status") or "") in statuses
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-drive infrastructure-contaminated topics into a clean run")
    p.add_argument("--source-run", required=True,
                   help="run_id of the run whose attempts.jsonl is the source of truth")
    p.add_argument("--run-id", required=True, help="new run_id for the clean re-measurement")
    p.add_argument("--statuses", nargs="+", default=list(DEFAULT_INFRA_STATUSES),
                   help="final_status values to treat as infrastructure-contaminated")
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--api-timeout", type=float, default=300.0,
                   help="HTTP read timeout for backend polls. Keep >= the "
                        "backend's worst per-poll latency: at concurrency>1 a "
                        "burst of sync work can stall a response past 120s, "
                        "which is NOT a task failure (the task keeps running).")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    p.add_argument("--timeout-seconds", type=float, default=14400)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--clarify-policy", default="protocol_violation")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print the selected topics and exit without driving them")
    return p


def _run(args: argparse.Namespace) -> None:
    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR

    source_dir = Path(results_dir) / args.source_run
    if not source_dir.exists():
        raise SystemExit(f"[rerun] source run dir not found: {source_dir}")

    statuses = tuple(args.statuses)
    target_ids = select_contaminated(source_dir, statuses)
    all_topics = {t["topic_id"]: t for t in load_topics()}
    unknown = [t for t in target_ids if t not in all_topics]
    if unknown:
        raise SystemExit(f"[rerun] unknown topic ids in source run: {unknown}")
    topics = [all_topics[t] for t in target_ids]

    print(f"[rerun] source run: {source_dir}")
    print(f"[rerun] contaminated statuses {statuses} -> {len(topics)} topic(s): "
          f"{[t['topic_id'] for t in topics]}")
    if args.dry_run or not topics:
        return

    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_fullv2")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="idea_e2e", mode=MODE, split="topics_v1",
        sample_count=len(topics), seed=None, limit=None,
        extra={
            "system": "full_v2",
            "api_base": args.api_base,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout_seconds,
            "clarify_policy": args.clarify_policy,
            "concurrency": args.concurrency,
            "strata": strata_counts(topics),
            "rerun_of": args.source_run,
            "rerun_reason": (
                "topics whose previous attempt ended in "
                f"{list(statuses)} — infrastructure contamination, not a "
                "scientific outcome; protocol identical to run_full_v2"
            ),
            "fairness": "identical frozen protocol to run_full_v2; only the "
                        "topic subset differs (the contaminated ones)",
        },
    )
    run.write_config(cfg, overwrite=True)

    # Cross-process double-start guard, same as the frozen driver.
    run_lock = run.dir / ".driver.lock"
    try:
        lock_fd = os.open(str(run_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"[rerun] refusing to start: {run_lock} already exists — another "
            "driver may be driving this run dir.")

    # 300s, not the 120s used by run_full_v2's own driver: at concurrency>1 the
    # backend event loop stalls in bursts (sync PDF/DB work inside async code)
    # and a single poll read can exceed 120s. That is not a task failure — the
    # task keeps running server-side — so a short client timeout burns the topic
    # as an error while its task row is still alive (observed 2026-09-14:
    # 13/15 topics recorded ReadTimeout at --concurrency 3, yet 7 of their tasks
    # were still in extracting_evidence/searching afterwards).
    api = ApiClient(args.api_base, timeout=args.api_timeout)
    papers_path = run.dir / "papers_export.jsonl"
    gaps_path = run.dir / "gaps_export.jsonl"
    file_lock = threading.Lock()

    def drive_topic(sample: dict) -> dict:
        """Same contract as run_full_v2._run.drive_topic (kept in lockstep)."""
        record = None
        try:
            collected = run_one_topic(api, sample, args.poll_interval,
                                      args.timeout_seconds, args.clarify_policy)
            record = to_prediction_record(sample, collected)
            with file_lock:
                with papers_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(export_papers(sample, collected),
                                        ensure_ascii=False, default=str) + "\n")
                with gaps_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(export_gaps(sample, collected),
                                        ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # per-topic isolation
            record = {
                "sample_id": sample["topic_id"],
                "topic_id": sample["topic_id"],
                "system": "full_v2",
                "parse_status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        record.setdefault("parse_status", "ok")
        record.setdefault("error", None)
        with file_lock:
            with (run.dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            _rewrite_predictions(run)
        print(f"    -> {sample['topic_id']}: "
              f"{record.get('decision') or record.get('parse_status')} "
              f"({record.get('final_status', '')})")
        return record

    try:
        concurrency = max(1, int(args.concurrency or 1))
        if concurrency == 1:
            for sample in topics:
                drive_topic(sample)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(drive_topic, s): s["topic_id"] for s in topics}
                for fut in as_completed(futures):
                    fut.result()
    finally:
        api.close()
        try:
            os.close(lock_fd)
        except OSError:
            pass
        run_lock.unlink(missing_ok=True)

    print(f"[rerun] run dir: {run.dir}")


def main() -> None:
    _run(build_parser().parse_args())


if __name__ == "__main__":
    main()

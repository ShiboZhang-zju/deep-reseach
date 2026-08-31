"""production_e2e_v1 — Full V2 pipeline driver (production, zero code changes).

Drives the production system over HTTP: create task -> start -> poll ->
auto-answer clarification -> terminal state -> collect ideas/gaps/papers ->
export the shared prediction record + papers_export.jsonl.

Requires the backend running (default http://localhost:8000). One topic at a
time (production concurrency is 1 agent per task by design).

Usage:
    cd backend
    python -m eval.production_e2e.run_full_v2 --run-id pe2e_v1_fullv2

Outputs (backend/eval_results/<run_id>/):
    predictions.jsonl   one shared-schema record per topic (idea or abstain)
    papers_export.jsonl topic_id -> full retrieved paper list (consumed by
                        baseline_retrieval: same-literature design point)
    gaps_export.jsonl   topic_id -> gap candidates (surviving claimed_delta is
                        the object of the false-open-gap metric)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from eval.common import EvalRun
from eval.config import DEFAULT_RESULTS_DIR, build_run_config
from eval.production_e2e.baseline_direct import load_topics, strata_counts
from eval.production_e2e.schema import (
    E2EDecision,
    FinalIdea,
    TERMINAL_STATUSES,
    build_prediction_record,
)

BENCHMARK = "production_e2e"
MODE = "full_v2"
DEFAULT_API_BASE = "http://localhost:8000/api"
DEFAULT_POLL_INTERVAL_S = 20
DEFAULT_TIMEOUT_S = 3 * 3600  # 3h per topic; a full run is 20-60 min
PAPERS_PAGE_LIMIT = 200

# Statuses that mean "run finished, outcome collectable". waiting_for_clarification
# is handled inside the poll loop (auto-answer, at most once per topic).
AUTO_CLARIFY_ANSWER = (
    "No further clarification is needed. Please research this direction as stated."
)


class ApiClient:
    def __init__(self, base: str, timeout: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.Client(base_url=self.base, timeout=timeout)

    def create_task(self, user_input: str) -> dict:
        resp = self._client.post("/tasks", json={"user_input": user_input})
        resp.raise_for_status()
        return resp.json()

    def start_task(self, task_id: str) -> dict:
        resp = self._client.post(f"/tasks/{task_id}/start")
        resp.raise_for_status()
        return resp.json()

    def stop_task(self, task_id: str) -> dict:
        resp = self._client.post(f"/tasks/{task_id}/stop")
        resp.raise_for_status()
        return resp.json()

    def get_task(self, task_id: str) -> dict:
        resp = self._client.get(f"/tasks/{task_id}")
        resp.raise_for_status()
        return resp.json()

    def clarify(self, task_id: str, answers: list[str]) -> dict:
        resp = self._client.post(f"/tasks/{task_id}/clarify", json={"answers": answers})
        resp.raise_for_status()
        return resp.json()

    def _get_all(self, path: str) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            resp = self._client.get(path, params={"limit": PAPERS_PAGE_LIMIT, "offset": offset})
            resp.raise_for_status()
            batch = resp.json()
            out.extend(batch)
            if len(batch) < PAPERS_PAGE_LIMIT:
                return out
            offset += PAPERS_PAGE_LIMIT

    def get_ideas(self, task_id: str) -> list[dict]:
        resp = self._client.get(f"/tasks/{task_id}/ideas")
        resp.raise_for_status()
        return resp.json()  # active ideas, already ordered by final_score desc

    def get_gaps(self, task_id: str) -> list[dict]:
        resp = self._client.get(f"/tasks/{task_id}/gaps")
        resp.raise_for_status()
        return resp.json()

    def get_experiments(self, task_id: str) -> list[dict]:
        resp = self._client.get(f"/tasks/{task_id}/experiments")
        resp.raise_for_status()
        return resp.json()

    def get_papers(self, task_id: str) -> list[dict]:
        return self._get_all(f"/tasks/{task_id}/papers")

    def close(self) -> None:
        self._client.close()


def run_one_topic(api: ApiClient, sample: dict, poll_interval: float,
                  timeout_s: float) -> dict:
    """Drive one production task to a terminal state; returns the collection dict."""
    topic_id = sample["topic_id"]
    task = api.create_task(sample["topic"])
    task_id = task["id"]
    api.start_task(task_id)

    clarified = False
    started = time.perf_counter()
    status = "unknown"
    while True:
        time.sleep(poll_interval)
        task = api.get_task(task_id)
        status = str(task.get("status") or "unknown")
        if status == "waiting_for_clarification" and not clarified:
            # Auto-answer once: the E2E protocol treats the topic text as final.
            api.clarify(task_id, [AUTO_CLARIFY_ANSWER])
            clarified = True
            continue
        if status in TERMINAL_STATUSES:
            break
        if (time.perf_counter() - started) > timeout_s:
            try:
                api.stop_task(task_id)
            except Exception:
                pass
            status = "stopped_timeout"
            break
    wall_clock_s = int(time.perf_counter() - started)

    ideas = api.get_ideas(task_id)
    gaps = api.get_gaps(task_id)
    experiments = api.get_experiments(task_id)
    papers = api.get_papers(task_id)

    return {
        "task_id": task_id,
        "final_status": status,
        "stop_reason": task.get("stop_reason"),
        "wall_clock_s": wall_clock_s,
        "clarified_once": clarified,
        "ideas": ideas,
        "gaps": gaps,
        "experiments": experiments,
        "papers": papers,
    }


def map_final_idea(idea: dict, experiments: list[dict]) -> FinalIdea:
    """Map the top active ResearchIdea (+ its experiment plan) to the shared schema."""
    plan = next((e for e in experiments if e.get("idea_id") == idea.get("id")), None)
    if plan:
        experiment_outline = " | ".join(filter(None, [
            f"hypothesis: {plan.get('hypothesis') or ''}".strip(),
            f"metrics: {plan.get('metrics') or ''}".strip(),
        ]))
    else:
        experiment_outline = str(idea.get("expected_contribution") or "").strip()
    return FinalIdea(
        title=str(idea.get("title") or "").strip() or "untitled",
        research_question=str(idea.get("description") or "").strip(),
        method_sketch=str(idea.get("method_sketch") or "").strip(),
        experiment_outline=experiment_outline,
        supporting_rationale=str(idea.get("motivation") or "").strip(),
    )


def to_prediction_record(sample: dict, collected: dict) -> dict:
    ideas = collected["ideas"] or []
    extra = {
        "task_id": collected["task_id"],
        "final_status": collected["final_status"],
        "stop_reason": collected["stop_reason"],
        "wall_clock_s": collected["wall_clock_s"],
        "clarified_once": collected["clarified_once"],
        "papers_count": len(collected["papers"]),
        "active_ideas_count": len(ideas),
        "all_idea_decisions": [
            {"title": i.get("title"), "decision": i.get("decision"),
             "final_score": i.get("final_score"), "confidence_tier": i.get("confidence_tier")}
            for i in ideas
        ],
        "gap_status_counts": _count_by(collected["gaps"], "status"),
        "surviving_gap_claimed_deltas": [
            g.get("claimed_delta") for g in collected["gaps"]
            if g.get("status") == "surviving" and g.get("claimed_delta")
        ],
    }
    if ideas:
        top = ideas[0]  # frozen rule: ONE final idea per topic, highest final_score
        idea_obj = map_final_idea(top, collected["experiments"])
        decision = E2EDecision(decision="propose_idea", idea=idea_obj)
        record = build_prediction_record(
            topic_id=sample["topic_id"], stratum=sample["stratum"],
            topic=sample["topic"], system="full_v2",
            decision=decision,
            extra=extra,
        )
    else:
        abstain_reason = (
            f"terminal_status={collected['final_status']}; "
            f"stop_reason={collected['stop_reason']}; active_ideas=0")
        decision = E2EDecision(decision="abstain", idea=None, abstain_reason=abstain_reason)
        record = build_prediction_record(
            topic_id=sample["topic_id"], stratum=sample["stratum"],
            topic=sample["topic"], system="full_v2",
            decision=decision,
            extra=extra,
        )
    return record


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = str(item.get(key) or "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def export_papers(sample: dict, collected: dict) -> dict:
    return {
        "topic_id": sample["topic_id"],
        "task_id": collected["task_id"],
        # Already ordered by final_score desc by the API (defensive re-sort in consumer).
        "papers": [
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citation_count": p.get("citation_count"),
                "final_score": p.get("final_score"),
                "priority": p.get("priority"),
            }
            for p in collected["papers"]
        ],
    }


def export_gaps(sample: dict, collected: dict) -> dict:
    return {
        "topic_id": sample["topic_id"],
        "task_id": collected["task_id"],
        "gaps": [
            {
                "id": g.get("id"),
                "claimed_delta": g.get("claimed_delta"),
                "existing_coverage": g.get("existing_coverage"),
                "status": g.get("status"),
            }
            for g in collected["gaps"]
        ],
    }


def _run(args) -> None:
    topics = load_topics()
    if args.limit:
        topics = topics[: args.limit]

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    run = EvalRun(results_dir, run_id=args.run_id, run_prefix="pe2e_fullv2")
    cfg = build_run_config(
        benchmark=BENCHMARK, task="idea_e2e", mode=MODE, split="topics_v1",
        sample_count=len(topics), seed=None, limit=args.limit,
        extra={
            "system": "full_v2",
            "api_base": args.api_base,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout_seconds,
            "strata": strata_counts(topics),
            "fairness": "production pipeline unmodified; same provider/temperature "
                        "as baselines; one final idea per topic (top final_score)",
        },
    )
    run.write_config(cfg, overwrite=True)

    # Resume: skip topics already present (parse_status=ok) in predictions.jsonl.
    done = run.existing_sample_ids()

    api = ApiClient(args.api_base, timeout=120.0)
    papers_path = run.dir / "papers_export.jsonl"
    gaps_path = run.dir / "gaps_export.jsonl"
    try:
        for idx, sample in enumerate(topics):
            if sample["topic_id"] in done:
                print(f"[{idx+1}/{len(topics)}] {sample['topic_id']}: SKIP (resume)")
                continue
            print(f"[{idx+1}/{len(topics)}] {sample['topic_id']}: running "
                  f"({sample['stratum']}) -> {sample['topic'][:70]}")
            record = None
            try:
                collected = run_one_topic(api, sample, args.poll_interval, args.timeout_seconds)
                record = to_prediction_record(sample, collected)
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
            with (run.dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            # Keep predictions.jsonl exactly-once per topic: rewrite from attempts.
            _rewrite_predictions(run)
            print(f"    -> {record.get('decision') or record.get('parse_status')} "
                  f"({record.get('final_status', '')})")
    finally:
        api.close()

    print(f"[run_full_v2] run dir: {run.dir}")


def _rewrite_predictions(run: EvalRun) -> None:
    """predictions.jsonl holds one final record per topic (latest attempt wins)."""
    latest: dict[str, dict] = {}
    attempts_path = run.dir / "attempts.jsonl"
    if attempts_path.exists():
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
    with (run.dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for rec in latest.values():
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="production_e2e Full V2 driver")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()

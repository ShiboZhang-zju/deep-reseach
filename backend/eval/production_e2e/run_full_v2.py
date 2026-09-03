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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# is handled inside the poll loop per the frozen clarification protocol.
AUTO_CLARIFY_ANSWER = (
    "No further clarification is needed. Please research this direction as stated."
)
# Frozen fairness protocol (README design point 1): the 24 topics are pre-registered
# to be specific enough. If Full V2 still asks for clarification, that sample is
# flagged as a protocol violation and gets NO auto-generated extra information —
# auto-answering would hand V2 a better problem definition than Baselines A/B get.
CLARIFY_POLICIES = ("protocol_violation", "auto_answer")


class ApiClient:
    def __init__(self, base: str, timeout: float = 300.0) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.Client(base_url=self.base, timeout=timeout)

    # Transient transport failures: under N concurrent agents the backend event
    # loop stalls in bursts (sync PDF parsing / DB work in async context) and
    # API reads time out — transient, NOT per-topic-fatal. Retry with linear
    # backoff; 429/502/503/504 (concurrency reject / gateway) also retry.
    _RETRY_EXC = (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError,
                  httpx.ReadError, httpx.RemoteProtocolError)
    _RETRY_STATUS = (429, 502, 503, 504)

    def _send(self, method: str, path: str, *, retries: int = 5,
              delay: float = 20.0, **kwargs):
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = getattr(self._client, method)(path, **kwargs)
                resp.raise_for_status()
                return resp
            except self._RETRY_EXC as exc:
                last = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in self._RETRY_STATUS:
                    last = exc
                else:
                    raise
            time.sleep(delay * attempt)
        assert last is not None
        raise last

    def create_task(self, user_input: str) -> dict:
        return self._send("post", "/tasks", json={"user_input": user_input}).json()

    def start_task(self, task_id: str) -> dict:
        return self._send("post", f"/tasks/{task_id}/start").json()

    def stop_task(self, task_id: str) -> dict:
        return self._send("post", f"/tasks/{task_id}/stop").json()

    def get_task(self, task_id: str) -> dict:
        return self._send("get", f"/tasks/{task_id}").json()

    def clarify(self, task_id: str, answers: list[str]) -> dict:
        return self._send("post", f"/tasks/{task_id}/clarify",
                          json={"answers": answers}).json()

    def _get_all(self, path: str) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            batch = self._send("get", path, params={
                "limit": PAPERS_PAGE_LIMIT, "offset": offset}).json()
            out.extend(batch)
            if len(batch) < PAPERS_PAGE_LIMIT:
                return out
            offset += PAPERS_PAGE_LIMIT

    def get_ideas(self, task_id: str) -> list[dict]:
        return self._send("get", f"/tasks/{task_id}/ideas").json()

    def get_gaps(self, task_id: str) -> list[dict]:
        return self._send("get", f"/tasks/{task_id}/gaps").json()

    def get_interventions(self, task_id: str) -> list[dict]:
        return self._send("get", f"/tasks/{task_id}/interventions").json()

    def get_traces(self, task_id: str) -> list[dict]:
        return self._send("get", f"/tasks/{task_id}/traces").json()

    def get_experiments(self, task_id: str) -> list[dict]:
        return self._send("get", f"/tasks/{task_id}/experiments").json()

    def get_papers(self, task_id: str) -> list[dict]:
        return self._get_all(f"/tasks/{task_id}/papers")

    def close(self) -> None:
        self._client.close()


def run_one_topic(api: ApiClient, sample: dict, poll_interval: float,
                  timeout_s: float, clarify_policy: str) -> dict:
    """Drive one production task to a terminal state; returns the collection dict."""
    topic_id = sample["topic_id"]
    task = api.create_task(sample["topic"])
    task_id = task["id"]
    api.start_task(task_id)

    clarified = False
    protocol_flag: str | None = None
    started = time.perf_counter()
    status = "unknown"
    while True:
        time.sleep(poll_interval)
        task = api.get_task(task_id)
        status = str(task.get("status") or "unknown")
        if status == "waiting_for_clarification":
            if clarify_policy == "auto_answer" and not clarified:
                # Pilot-observation mode ONLY: auto-answering hands V2 a better
                # problem definition than Baselines A/B get — never use for the
                # frozen 24-topic run.
                api.clarify(task_id, [AUTO_CLARIFY_ANSWER])
                clarified = True
                continue
            # Frozen protocol (design point 1): flag and stop, no extra info.
            protocol_flag = "clarification_triggered"
            break
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

    if protocol_flag:
        # Flagged sample: do not treat it as a system outcome; record minimal info.
        return {
            "task_id": task_id,
            "final_status": status,
            "stop_reason": task.get("stop_reason"),
            "wall_clock_s": wall_clock_s,
            "clarified_once": clarified,
            "protocol_flag": protocol_flag,
            "ideas": [],
            "gaps": [],
            "experiments": [],
            "papers": [],
            "interventions": [],
            "llm_tokens_used_total": None,
            "trace_count": None,
        }

    ideas = api.get_ideas(task_id)
    gaps = api.get_gaps(task_id)
    experiments = api.get_experiments(task_id)
    papers = api.get_papers(task_id)
    interventions = api.get_interventions(task_id)
    # Best-effort LLM accounting from traces (tokens summed; trace count is a
    # step-level proxy, NOT the LLM call count — documented cost caveat).
    llm_tokens_used_total: int | None = None
    trace_count: int | None = None
    try:
        traces = api.get_traces(task_id)
        trace_count = len(traces)
        llm_tokens_used_total = sum(
            int(t.get("llm_tokens_used") or 0) for t in traces)
    except Exception:
        pass

    return {
        "task_id": task_id,
        "final_status": status,
        "stop_reason": task.get("stop_reason"),
        "wall_clock_s": wall_clock_s,
        "clarified_once": clarified,
        "protocol_flag": protocol_flag,
        "ideas": ideas,
        "gaps": gaps,
        "experiments": experiments,
        "papers": papers,
        "interventions": interventions,
        "llm_tokens_used_total": llm_tokens_used_total,
        "trace_count": trace_count,
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
    gaps = collected["gaps"] or []
    interventions = collected.get("interventions") or []
    gap_status_counts = _count_by(gaps, "status")
    extra = {
        "task_id": collected["task_id"],
        "final_status": collected["final_status"],
        "stop_reason": collected["stop_reason"],
        "wall_clock_s": collected["wall_clock_s"],
        "clarified_once": collected["clarified_once"],
        "protocol_flag": collected.get("protocol_flag"),
        "papers_count": len(collected["papers"]),
        "active_ideas_count": len(ideas),
        # best-of-N accounting (design point 3): how many internal candidates
        # the bundle had before the final top-1 pick — kept, NOT normalized away.
        "gap_candidate_count": gap_status_counts.get("candidate", 0),
        "gap_survived_count": gap_status_counts.get("surviving", 0),
        "interventions_count": len(interventions),
        "interventions_passed": sum(
            1 for i in interventions
            if str(i.get("gate_status") or i.get("status") or "") != "FAIL"),
        "llm_tokens_used_total": collected.get("llm_tokens_used_total"),
        "trace_count": collected.get("trace_count"),
        "all_idea_decisions": [
            {"title": i.get("title"), "decision": i.get("decision"),
             "final_score": i.get("final_score"), "confidence_tier": i.get("confidence_tier")}
            for i in ideas
        ],
        "gap_status_counts": gap_status_counts,
        "surviving_gap_claimed_deltas": [
            g.get("claimed_delta") for g in gaps
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
        if collected.get("protocol_flag"):
            abstain_reason = f"protocol_violation ({collected['protocol_flag']}); " + abstain_reason
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
            "clarify_policy": args.clarify_policy,
            "concurrency": args.concurrency,
            "strata": strata_counts(topics),
            "fairness": "production pipeline unmodified; same provider/temperature "
                        "as baselines; one final idea per topic (top final_score); "
                        "clarification protocol: topics are pre-registered specific, "
                        "a clarification request is flagged protocol_violation and "
                        "gets no auto-generated extra info; topic-level driver "
                        "concurrency is an ops knob (backend max_concurrent_agents=2), "
                        "per-topic protocol unchanged",
        },
    )
    run.write_config(cfg, overwrite=True)

    # Resume: skip topics already present (parse_status=ok) in predictions.jsonl.
    done = run.existing_sample_ids()

    api = ApiClient(args.api_base, timeout=120.0)
    papers_path = run.dir / "papers_export.jsonl"
    gaps_path = run.dir / "gaps_export.jsonl"
    concurrency = max(1, int(args.concurrency or 1))
    file_lock = threading.Lock()
    pending = [s for s in topics if s["topic_id"] not in done]
    print(f"[run_full_v2] topics={len(topics)} done={len(done)} "
          f"pending={len(pending)} concurrency={concurrency}")

    def drive_topic(sample: dict) -> dict:
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
            # Keep predictions.jsonl exactly-once per topic: rewrite from attempts.
            _rewrite_predictions(run)
        print(f"    -> {sample['topic_id']}: "
              f"{record.get('decision') or record.get('parse_status')} "
              f"({record.get('final_status', '')})")
        return record

    try:
        if concurrency == 1:
            for sample in pending:
                drive_topic(sample)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(drive_topic, s): s["topic_id"] for s in pending}
                for fut in as_completed(futures):
                    fut.result()
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
    parser.add_argument("--concurrency", type=int, default=1,
                        help="topics driven in parallel (ops knob; per-topic "
                             "protocol unchanged; backend must allow "
                             "max_concurrent_agents >= this)")
    parser.add_argument("--clarify-policy", choices=list(CLARIFY_POLICIES),
                        default="protocol_violation",
                        help="protocol_violation (frozen): a clarification request "
                             "flags the sample and gets no extra info; auto_answer "
                             "is pilot-observation only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()

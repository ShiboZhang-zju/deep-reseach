"""Shared eval-runner utilities.

Eval runs never touch the research DB: the harness drives the LLM provider
directly (same settings/factory as production) and persists everything to
JSON/JSONL artifacts under a run directory:

    config.json / predictions.jsonl / metrics.json / summary.md
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class CallStats:
    """Aggregated per-sample LLM accounting (calls, tokens, latency)."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_latency_ms": self.latency_ms,
        }


async def chat_json(
    llm,
    messages: list[dict],
    schema: Type[T],
    temperature: float = 0.0,
    stats: CallStats | None = None,
) -> T:
    """One structured LLM call with accounting.

    Raises on failure; the caller decides the failure granularity (per window,
    per inspiration, per comparison, or per sample).
    """
    start = time.perf_counter()
    try:
        return await llm.chat_json(messages, schema, temperature=temperature)
    finally:
        if stats is not None:
            stats.calls += 1
            stats.latency_ms += int((time.perf_counter() - start) * 1000)
            usage = llm.get_last_usage() or {}
            stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            stats.completion_tokens += int(usage.get("completion_tokens") or 0)
            stats.total_tokens += int(usage.get("total_tokens") or 0)


class AsyncBridge:
    """Runs async LLM calls from synchronous code.

    The official scorers are synchronous; a dedicated loop thread keeps this
    safe regardless of the caller's context and of per-call client creation.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="eval-async-bridge")
        self._thread.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


def select_samples(samples: list[dict], limit: int | None, seed: int | None) -> list[dict]:
    """Deterministic sample selection.

    seed=None keeps dataset order (first N with a limit); a seed applies a
    stable shuffle so the same seed+limit always picks the same samples.
    """
    picked = list(samples)
    if seed is not None:
        random.Random(seed).shuffle(picked)
    if limit is not None:
        picked = picked[:limit]
    return picked


class EvalRun:
    """One eval run directory with the four standard artifacts."""

    def __init__(self, results_dir: Path, run_id: str | None = None,
                 run_prefix: str = "eval") -> None:
        base = Path(results_dir)
        if run_id:
            run_dir = base / run_id
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_dir = base / f"{stamp}_{run_prefix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.dir = run_dir

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def predictions_path(self) -> Path:
        return self.dir / "predictions.jsonl"

    @property
    def attempts_path(self) -> Path:
        return self.dir / "attempts.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.json"

    @property
    def summary_path(self) -> Path:
        return self.dir / "summary.md"

    def write_config(self, cfg: dict, overwrite: bool = False) -> None:
        # Resume: an explicitly reused run dir keeps its original config.
        if self.config_path.exists() and not overwrite:
            return
        # Atomic replace: a driver killed mid-write used to leave a truncated
        # config.json (provenance gone) behind.
        tmp_path = self.config_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.config_path)

    def existing_sample_ids(self) -> set[str]:
        if not self.predictions_path.exists():
            return set()
        ids: set[str] = set()
        for line in self.predictions_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sid = json.loads(line).get("sample_id")
            except json.JSONDecodeError:
                continue
            if sid:
                ids.add(sid)
        return ids

    def load_attempts(self) -> list[dict]:
        """All historical attempts (including failures/retries), append-only."""
        if not self.attempts_path.exists():
            return []
        records: list[dict] = []
        for line in self.attempts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def append_attempt(self, record: dict) -> None:
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_predictions_final(self, records: list[dict]) -> None:
        """Rewrite predictions.jsonl so each sample_id appears exactly once,
        holding its final successful prediction (dataset order preserved).

        Errored samples appear only in attempts.jsonl, never here.
        """
        final_by_id: dict[str, dict] = {}
        for record in records:
            if record.get("parse_status") == "ok":
                final_by_id[str(record.get("sample_id") or "")] = record
        with self.predictions_path.open("w", encoding="utf-8") as handle:
            for record in final_by_id.values():
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_metrics(self, metrics: dict) -> None:
        self.metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_summary(self, text: str) -> None:
        self.summary_path.write_text(text, encoding="utf-8")


async def run_samples(
    samples: list[dict],
    run_one,
    run: EvalRun,
    resume: bool = False,
) -> list[dict]:
    """Run one adapter call per sample with per-sample error isolation.

    Every attempt (success or failure) is appended to attempts.jsonl;
    predictions.jsonl is rewritten after the loop to hold exactly one final
    successful prediction per sample. A failing sample never aborts the run.
    With resume=True, samples whose LATEST attempt succeeded are skipped;
    errored samples are retried.
    """
    records = run.load_attempts() if resume else []
    latest_status: dict[str, str] = {}
    for record in records:
        sid = str(record.get("sample_id") or "")
        if sid:
            latest_status[sid] = str(record.get("parse_status") or "")
    done = {sid for sid, status in latest_status.items() if status == "ok"}
    for sample in samples:
        sid = str(sample.get("sample_id") or sample.get("source") or "")
        if sid and sid in done:
            continue
        start = time.perf_counter()
        try:
            record = await run_one(sample)
            record.setdefault("parse_status", "ok")
            record.setdefault("error", None)
        except Exception as exc:  # single-sample isolation is the point
            record = {
                "sample_id": sid,
                "prediction": None,
                "parse_status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
        record["latency_ms"] = int((time.perf_counter() - start) * 1000)
        record.setdefault("sample_id", sid)
        run.append_attempt(record)
        records.append(record)
    # One sample id -> one final prediction (errored samples stay in attempts only).
    run.write_predictions_final(records)
    return records


def flatten_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dicts into dotted-key numeric leaves (lists/bools skipped)."""
    flat: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_numbers(value, path))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        flat[prefix] = obj
    return flat


def aggregate_stats(records: list[dict]) -> dict:
    """Run-level accounting over the attempt history.

    Two views are distinguished: attempts (every call/retry, for cost) and
    samples (distinct sample_ids, judged by their LATEST attempt, for
    quality). parse_failure_rate = finally-failed samples / total samples.
    """
    keys = ["llm_calls", "prompt_tokens", "completion_tokens", "total_tokens", "llm_latency_ms"]
    totals: dict = {key: 0 for key in keys}
    for record in records:
        for key in keys:
            totals[key] += int(record.get(key) or 0)

    latest: dict[str, dict] = {}
    for record in records:
        sid = str(record.get("sample_id") or "")
        if sid:
            latest[sid] = record  # last attempt wins
    samples_total = len(latest)
    samples_ok = sum(1 for r in latest.values() if r.get("parse_status") == "ok")
    totals.update({
        "samples_total": samples_total,
        "samples_ok": samples_ok,
        "samples_error": samples_total - samples_ok,
        "attempts_total": sum(1 for r in records if str(r.get("sample_id") or "")),
        "attempts_error": sum(1 for r in records if r.get("parse_status") == "error"),
        "parse_failure_rate": round((samples_total - samples_ok) / samples_total, 4)
        if samples_total else 0.0,
    })
    return totals

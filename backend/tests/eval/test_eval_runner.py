"""Eval runner tests: deterministic selection, limit/resume, error isolation,
attempt/prediction separation, run accounting."""
import asyncio

import pytest

from eval.common import (
    EvalRun,
    aggregate_stats,
    flatten_numbers,
    run_samples,
    select_samples,
)


def _sample(i: int) -> dict:
    return {"sample_id": f"s{i}", "value": i}


async def _run_one_ok(sample: dict) -> dict:
    return {"sample_id": sample["sample_id"],
            "prediction": {"value": sample["value"]}, "gold": None}


async def _run_one_flaky(sample: dict) -> dict:
    if sample["sample_id"] == "s1":
        raise RuntimeError("boom")
    return {"sample_id": sample["sample_id"], "prediction": {"ok": True}, "gold": None}


def test_select_samples_limit_is_first_n_and_deterministic():
    samples = [_sample(i) for i in range(10)]
    picked = select_samples(samples, 3, None)
    assert [s["sample_id"] for s in picked] == ["s0", "s1", "s2"]
    # Same input -> same selection (dataset order is stable).
    assert select_samples([_sample(i) for i in range(10)], 3, None) == picked


def test_select_samples_seed_is_stable():
    samples = [_sample(i) for i in range(20)]
    first = [s["sample_id"] for s in select_samples(samples, 5, seed=123)]
    second = [s["sample_id"] for s in select_samples(
        [_sample(i) for i in range(20)], 5, seed=123)]
    assert first == second
    assert len(set(first)) == 5


def test_run_samples_limit_resume_and_persistence(tmp_path):
    run = EvalRun(tmp_path, run_id="fixed")
    samples = [_sample(i) for i in range(4)]

    first = asyncio.run(run_samples(samples[:2], _run_one_ok, run, resume=False))
    assert [r["sample_id"] for r in first] == ["s0", "s1"]

    # Resume on the same run dir: samples whose latest attempt succeeded are
    # skipped; the rest run.
    second = asyncio.run(run_samples(samples, _run_one_ok, run, resume=True))
    assert {r["sample_id"] for r in second} == {"s0", "s1", "s2", "s3"}

    attempts = run.attempts_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(attempts) == 4  # s0/s1 not retried
    predictions = run.predictions_path.read_text(encoding="utf-8").strip().splitlines()
    # One sample id -> one final prediction, dataset order preserved.
    assert [__import__("json").loads(line)["sample_id"] for line in predictions] == \
        ["s0", "s1", "s2", "s3"]


def test_run_samples_error_isolation(tmp_path):
    run = EvalRun(tmp_path, run_id="fixed")
    records = asyncio.run(
        run_samples([_sample(i) for i in range(3)], _run_one_flaky, run, resume=False))
    assert records[0]["parse_status"] == "ok"
    assert records[1]["parse_status"] == "error"
    assert "boom" in records[1]["error"]
    assert records[1]["prediction"] is None
    assert records[2]["parse_status"] == "ok"

    # attempts.jsonl keeps the full history; predictions.jsonl holds only the
    # final successful predictions (one per sample, errors excluded).
    attempts = run.attempts_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(attempts) == 3
    predictions = run.predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert [__import__("json").loads(line)["sample_id"] for line in predictions] == ["s0", "s2"]

    # Resume skips completed samples and RETRIES errored ones.
    follow_up = asyncio.run(
        run_samples([_sample(i) for i in range(3)], _run_one_ok, run, resume=True))
    assert follow_up[-1]["sample_id"] == "s1"
    assert follow_up[-1]["parse_status"] == "ok"
    attempts = run.attempts_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(attempts) == 4  # s1 retried, s0/s2 untouched
    predictions = run.predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert [__import__("json").loads(line)["sample_id"] for line in predictions] == \
        ["s0", "s2", "s1"]  # final predictions now include the retried s1


def test_aggregate_stats_distinguishes_attempts_and_samples():
    records = [
        {"sample_id": "a", "parse_status": "error", "llm_calls": 1, "total_tokens": 10},
        {"sample_id": "a", "parse_status": "ok", "llm_calls": 2, "total_tokens": 20},
        {"sample_id": "b", "parse_status": "ok", "llm_calls": 3, "total_tokens": 30},
        {"sample_id": "c", "parse_status": "error", "llm_calls": 1, "total_tokens": 5},
    ]
    stats = aggregate_stats(records)
    assert stats["samples_total"] == 3
    assert stats["samples_ok"] == 2
    assert stats["samples_error"] == 1
    assert stats["attempts_total"] == 4
    assert stats["attempts_error"] == 2
    assert stats["parse_failure_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert stats["llm_calls"] == 7  # cost counts every attempt including retries


def test_flatten_numbers_skips_lists_and_bools():
    obj = {"summary": {"round1": {"gold": 0.75}, "n": 2},
           "per_sample": [{"x": 1}], "flag": True}
    assert flatten_numbers(obj) == {"summary.round1.gold": 0.75, "summary.n": 2}

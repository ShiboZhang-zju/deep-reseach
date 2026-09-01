"""RINoBench adapter tests: verdict mapping, official-format serialization,
malformed outputs, and the official-metric replica."""
import asyncio

import pytest

from eval.rinobench import adapter as rino


class FakeEvalLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    async def chat(self, messages, temperature=0.7):
        self.calls.append({"kind": "chat", "messages": messages})
        return "fake chat response"

    async def chat_json(self, messages, schema, temperature=0.3):
        self.calls.append({"kind": "chat_json", "messages": messages})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return schema.model_validate(item)
        return item

    def get_last_usage(self):
        return self.last_usage


def _record(score=4, works=3):
    return {
        "source": f"https://openreview.net/forum?id=sample{score}",
        "venueid": "ICLR.cc/2022/Conference",
        "research_idea": {"objective": "o", "problem_statement": "p", "solution_approach": "s"},
        "novelty_score": score,
        "novelty_reasoning": "gold reasoning",
        "related_works": [{"title": f"W{i}", "abstract": f"A{i}", "authors": [],
                           "url": "", "venue": "", "year": 2020} for i in range(works)],
    }


RUBRIC = ["1: not novel", "2: marginally novel", "3: somewhat novel",
          "4: novel", "5: highly novel"]


def test_internal_verdict_mapping_is_fixed_and_deterministic():
    assert rino.internal_verdict_for(5) == "confirmed"
    assert rino.internal_verdict_for(4) == "confirmed"
    assert rino.internal_verdict_for(3) == "partially_closed"
    assert rino.internal_verdict_for(2) == "uncertain"
    assert rino.internal_verdict_for(1) == "closed"
    with pytest.raises(ValueError):
        rino.internal_verdict_for(0)
    with pytest.raises(ValueError):
        rino.internal_verdict_for(6)


def test_novelty_sample_official_format_element_exact_keys():
    llm = FakeEvalLLM([{"novelty_score": 4, "reasoning": "r",
                        "known_aspects": "k", "novelty_aspects": "n"}])
    out = asyncio.run(rino.run_novelty_sample(_record(4), llm, rubric=RUBRIC))
    # Official evaluator expects exactly these two keys.
    assert out["official_element"] == {"reasoning": "r", "novelty_score": 4}
    assert out["prediction"]["internal_verdict"] == "confirmed"
    assert "never adjusted per run" in out["prediction"]["verdict_mapping"]
    assert out["gold"] == {"novelty_score": 4}
    # The official rubric is passed to the model verbatim.
    user_message = llm.calls[0]["messages"][1]["content"]
    assert "1: not novel" in user_message and "5: highly novel" in user_message
    assert out["prediction"]["related_works_used"] == 3


def test_novelty_out_of_range_score_becomes_sample_error(tmp_path):
    from eval.common import EvalRun, run_samples
    llm = FakeEvalLLM([{"novelty_score": 6, "reasoning": "r"}])
    run = EvalRun(tmp_path, run_id="fixed")
    records = asyncio.run(run_samples(
        [_record(4)],
        lambda sample: rino.run_novelty_sample(sample, llm, rubric=RUBRIC),
        run, resume=False))
    assert records[0]["parse_status"] == "error"
    # The out-of-range score is rejected by the schema (ge=1, le=5) before any
    # verdict mapping could run.
    assert "novelty_score" in records[0]["error"]


def test_related_works_truncation_recorded():
    llm = FakeEvalLLM([{"novelty_score": 2, "reasoning": "r"}])
    out = asyncio.run(rino.run_novelty_sample(
        _record(2, works=45), llm, rubric=RUBRIC, max_related_works=40))
    assert out["prediction"]["related_works_used"] == 40
    assert out["prediction"]["related_works_truncated"] is True
    assert out["prediction"]["abstained"] is False  # default: judgment made


def test_abstained_recorded_in_prediction_and_rate():
    llm = FakeEvalLLM([{"novelty_score": 3, "reasoning": "r", "abstained": True}])
    out = asyncio.run(rino.run_novelty_sample(_record(4), llm, rubric=RUBRIC))
    assert out["prediction"]["abstained"] is True
    # Best-effort score is still emitted in the official element.
    assert out["official_element"]["novelty_score"] == 3

    processed = [
        {"prediction": out["prediction"], "gold": out["gold"]},
        {"prediction": {"novelty_score": 4, "reasoning": "x", "abstained": False},
         "gold": {"novelty_score": 4}},
    ]
    metrics = rino.build_metrics(processed)
    assert metrics["abstention_rate"] == pytest.approx(0.5)


def test_score_metrics_replica_matches_hand_computed():
    predicted, gold = [4, 4, 2, 2], [5, 4, 3, 2]
    metrics = rino.compute_score_metrics(predicted, gold)
    # class 2: tp=1, fp=1 (2 predicted for gold 3), fn=0 -> F1 = 2/3
    assert metrics["f1_scores"]["2"] == pytest.approx(2 / 3)
    # class 4: tp=1, fp=1 (4 predicted for gold 5), fn=0 -> F1 = 2/3
    assert metrics["f1_scores"]["4"] == pytest.approx(2 / 3)
    assert metrics["f1_scores"]["1"] == 0.0
    assert metrics["f1_macro"] == pytest.approx((0 + 2 / 3 + 0 + 2 / 3 + 0) / 5)
    assert metrics["mean_absolute_error"] == pytest.approx(0.5)


def test_pure_fallback_matches_sklearn():
    try:
        from sklearn.metrics import f1_score, mean_absolute_error
    except ImportError:
        pytest.skip("sklearn not installed")
    predicted, gold = [3, 3, 5, 1, 2, 4], [4, 3, 5, 1, 1, 2]
    pure_f1, pure_macro, pure_mae = rino._pure_f1_mae(predicted, gold)
    sk_per_class = f1_score(gold, predicted, labels=[1, 2, 3, 4, 5], average=None,
                            zero_division=0)
    for idx, label in enumerate(["1", "2", "3", "4", "5"]):
        assert pure_f1[label] == pytest.approx(float(sk_per_class[idx]))
    assert pure_macro == pytest.approx(f1_score(
        gold, predicted, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0))
    assert pure_mae == pytest.approx(mean_absolute_error(gold, predicted))


def test_build_metrics_and_official_element_order():
    processed = []
    for score, prediction in [(1, 3), (2, 4), (3, 3)]:
        processed.append({
            "prediction": {"novelty_score": prediction, "reasoning": "r"},
            "gold": {"novelty_score": score},
        })
    metrics = rino.build_metrics(processed)
    assert metrics["benchmark"] == "RINoBench"
    assert metrics["mode"] == "gold_related_works"
    assert metrics["official_metric_replica"]["mean_absolute_error"] == pytest.approx(4 / 3)
    assert "official evaluator" in metrics["note"]


def test_load_rubric_from_benchmark_repo():
    from eval.config import RINOBENCH_DIR
    if not (RINOBENCH_DIR / "data/final_benchmark_dataset/label_descriptions.json").exists():
        pytest.skip("official RINoBench repo not cloned")
    rubric = rino.load_rubric()
    assert len(rubric) == 5
    assert all(str(i + 1) in text.split(":")[0] for i, text in enumerate(rubric))


# --------------------------------------------------------------------------
# V3 criterion-first novelty policy
# --------------------------------------------------------------------------

def _v3_payload(**overrides):
    base = {
        "closest_work": "Prior work X",
        "core_coverage": "partial",
        "residual_delta": "adds a new mechanism",
        "delta_grounded": True,
        "delta_substantive": True,
        "reasoning": "r",
        "novelty_score": 4,
        "abstained": False,
    }
    base.update(overrides)
    return base


def test_v3_valid_result_records_all_criteria():
    llm = FakeEvalLLM([_v3_payload()])
    out = asyncio.run(rino.run_novelty_sample(
        _record(4), llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3))
    p = out["prediction"]
    assert p["novelty_policy"] == "criterion-first-v3"
    assert p["closest_work"] == "Prior work X"
    assert p["core_coverage"] == "partial"
    assert p["delta_grounded"] is True and p["delta_substantive"] is True
    assert p["internal_verdict"] == "confirmed"
    assert out["official_element"] == {"reasoning": "r", "novelty_score": 4}
    # The official rubric is NOT used in v3; the frozen ordinal semantics are.
    user_message = llm.calls[0]["messages"][1]["content"]
    assert "Score 5: The closest prior works do not cover" in user_message
    assert "quality != novelty" in user_message


def test_v3_invalid_core_coverage_becomes_sample_error(tmp_path):
    from eval.common import EvalRun, run_samples
    llm = FakeEvalLLM([_v3_payload(core_coverage="mostly")])
    run = EvalRun(tmp_path, run_id="fixed")
    records = asyncio.run(run_samples(
        [_record(4)], lambda s: rino.run_novelty_sample(
            s, llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3),
        run, resume=False))
    assert records[0]["parse_status"] == "error"
    assert "core_coverage" in records[0]["error"]


def test_v3_missing_residual_delta_becomes_sample_error(tmp_path):
    from eval.common import EvalRun, run_samples
    payload = _v3_payload()
    del payload["residual_delta"]
    llm = FakeEvalLLM([payload])
    run = EvalRun(tmp_path, run_id="fixed")
    records = asyncio.run(run_samples(
        [_record(4)], lambda s: rino.run_novelty_sample(
            s, llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3),
        run, resume=False))
    assert records[0]["parse_status"] == "error"
    assert "residual_delta" in records[0]["error"]


@pytest.mark.parametrize("bad_score", [0, 6])
def test_v3_malformed_score_becomes_sample_error(tmp_path, bad_score):
    from eval.common import EvalRun, run_samples
    llm = FakeEvalLLM([_v3_payload(novelty_score=bad_score)])
    run = EvalRun(tmp_path, run_id="fixed")
    records = asyncio.run(run_samples(
        [_record(4)], lambda s: rino.run_novelty_sample(
            s, llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3),
        run, resume=False))
    assert records[0]["parse_status"] == "error"
    assert "novelty_score" in records[0]["error"]


def test_v3_abstained_keeps_best_effort_score():
    llm = FakeEvalLLM([_v3_payload(abstained=True, novelty_score=2,
                                   delta_substantive=False)])
    out = asyncio.run(rino.run_novelty_sample(
        _record(1), llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3))
    assert out["prediction"]["abstained"] is True
    assert out["prediction"]["novelty_score"] == 2
    # Official element still emitted for scorer compatibility.
    assert out["official_element"]["novelty_score"] == 2


def test_policy_default_is_v1_and_unchanged():
    llm = FakeEvalLLM([{"novelty_score": 4, "reasoning": "r"}])
    out = asyncio.run(rino.run_novelty_sample(_record(4), llm, rubric=RUBRIC))
    assert out["prediction"]["novelty_policy"] == "holistic-v1"
    assert "closest_work" not in out["prediction"]
    assert out["prediction"]["internal_verdict"] == "confirmed"


# Synthetic semantic-regression cases: criterion consistency (pipe level),
# not benchmark scores.
def test_v3_case_a_near_complete_coverage_maps_low():
    llm = FakeEvalLLM([_v3_payload(core_coverage="near_complete", delta_substantive=False,
                                   delta_grounded=False, novelty_score=1)])
    out = asyncio.run(rino.run_novelty_sample(
        _record(1), llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3))
    assert out["prediction"]["core_coverage"] == "near_complete"
    assert out["prediction"]["novelty_score"] == 1
    assert out["prediction"]["internal_verdict"] == "closed"


def test_v3_case_b_partial_coverage_maps_middle():
    llm = FakeEvalLLM([_v3_payload(core_coverage="partial", delta_substantive=True,
                                   delta_grounded=True, novelty_score=3)])
    out = asyncio.run(rino.run_novelty_sample(
        _record(3), llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3))
    assert out["prediction"]["core_coverage"] == "partial"
    assert out["prediction"]["novelty_score"] == 3
    assert out["prediction"]["internal_verdict"] == "partially_closed"


def test_v3_case_c_substantive_delta_maps_high():
    llm = FakeEvalLLM([_v3_payload(core_coverage="none", delta_substantive=True,
                                   delta_grounded=True, novelty_score=5)])
    out = asyncio.run(rino.run_novelty_sample(
        _record(5), llm, rubric=RUBRIC, novelty_policy=rino.NOVELTY_POLICY_V3))
    assert out["prediction"]["core_coverage"] == "none"
    assert out["prediction"]["delta_substantive"] is True
    assert out["prediction"]["novelty_score"] == 5
    assert out["prediction"]["internal_verdict"] == "confirmed"

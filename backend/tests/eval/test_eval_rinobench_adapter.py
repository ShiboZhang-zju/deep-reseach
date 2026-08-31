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

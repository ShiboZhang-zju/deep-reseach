"""ResearchBench adapter tests: parsing, serialization, official rank
semantics, malformed outputs, and an offline official-scorer smoke."""
import asyncio

import pytest

from eval.config import RESEARCHBENCH_DIR
from eval.researchbench import adapter as rb


class FakeEvalLLM:
    """Duck-typed LLMProvider stand-in with scripted structured responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    async def chat(self, messages, temperature=0.7):
        self.calls += 1
        return "fake chat response"

    async def chat_json(self, messages, schema, temperature=0.3):
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeEvalLLM ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return schema.model_validate(item)
        return item

    def get_last_usage(self):
        return self.last_usage


def _retrieve_record(candidates_n=30, gold_n=3):
    candidates = [{
        "title": f"Paper {i}",
        "abstract": f"Abstract {i} " * 20,
        "index": i,
        "label": "gold" if i < gold_n else "negative_t1",
        "source_file": "d0.json" if i < gold_n else "d1.json",
    } for i in range(candidates_n)]
    return {
        "sample_id": "Test/1",
        "research_question": "How do we speed up X?",
        "background_survey": "background",
        "candidates": candidates,
        "gold_titles": [f"Paper {i}" for i in range(gold_n)],
        "label_counts": {"gold": gold_n, "negative_t1": candidates_n - gold_n},
        "default_params": {"window_size": 10, "keep_size": 2, "rounds": 2},
    }


def _pick(indices):
    return {"selections": [{"index": i, "reason": "r"} for i in indices]}


def _ranking_record(n_negatives=2):
    return {
        "sample_id": "Test/1",
        "research_question": "Q?",
        "gold_hypothesis": "GOLD",
        "fake_negative_hypotheses": [f"FAKE{i}" for i in range(n_negatives)],
        "model_negative_hypotheses": [],
    }


def _generation_record():
    return {
        "sample_id": "Test/1",
        "research_question": "Q?",
        "background_survey": "B",
        "gold_inspirations": [{"title": "Insp A", "abstract": "a"},
                              {"title": "Insp B", "abstract": "b"}],
        "gold_hypothesis": "GOLD HYPOTHESIS",
        "default_params": {"num_mutations": 1},
    }


def _draft(hypothesis, scores):
    return {"hypothesis": hypothesis, "mechanism": "m", "measurable_outcome": "o",
            "falsification_condition": "f", "self_validness": scores[0],
            "self_novelty": scores[1], "self_significance": scores[2],
            "self_specificity": scores[3], "reasoning": "why"}


# ----------------------------- retrieval -----------------------------

def test_retrieval_window_funnel_produces_official_schema():
    # window 10 / keep 2 / rounds 2: r1 = 3 windows -> 6 titles, r2 = 1 window -> 2 titles.
    record = _retrieve_record(candidates_n=30)
    llm = FakeEvalLLM([_pick([0, 1]), _pick([1, 2]), _pick([0, 2]), _pick([0, 1])])
    out = asyncio.run(rb.run_retrieval_sample(record, llm))
    pred = out["prediction"]
    assert pred["sample_id"] == "Test/1"
    assert pred["model"]
    assert len(pred["selected_round1_titles"]) == 6
    assert len(pred["selected_round2_titles"]) == 2
    assert {entry["round"] for entry in pred["rounds"]} == {1, 2}
    # One entry per window, holding that window's picks, with official keys.
    entry = pred["rounds"][0]["selected"][0]
    assert set(rb._OFFICIAL_ENTRY_KEYS) <= set(entry)
    assert entry["label"] in {"gold", "negative_t1"}
    assert out["gold"]["gold_titles"] == [f"Paper {i}" for i in range(3)]


def test_retrieval_selection_uses_local_window_index():
    record = _retrieve_record(candidates_n=10, gold_n=2)
    record["default_params"] = {"window_size": 10, "keep_size": 1, "rounds": 2}
    llm = FakeEvalLLM([_pick([1]), _pick([0])])
    out = asyncio.run(rb.run_retrieval_sample(record, llm))
    pred = out["prediction"]
    assert pred["selected_round1_titles"] == ["Paper 1"]
    assert pred["selected_round2_titles"] == ["Paper 1"]


def test_retrieval_window_error_shrinks_funnel_without_aborting():
    record = _retrieve_record(candidates_n=10, gold_n=2)
    record["default_params"] = {"window_size": 5, "keep_size": 1, "rounds": 2}
    # window0 of round1 fails, window1 succeeds, round2 window succeeds.
    llm = FakeEvalLLM([RuntimeError("llm down"), _pick([1]), _pick([0])])
    out = asyncio.run(rb.run_retrieval_sample(record, llm))
    pred = out["prediction"]
    assert len(pred["selected_round1_titles"]) == 1
    assert any("round1/window0" in e for e in out["eval_extra"]["window_errors"])
    assert len(pred["selected_round2_titles"]) == 1


def test_retrieval_total_round_failure_raises_sample_error():
    record = _retrieve_record(candidates_n=10, gold_n=2)
    record["default_params"] = {"window_size": 5, "keep_size": 1, "rounds": 2}
    llm = FakeEvalLLM([RuntimeError("x"), RuntimeError("y")])
    with pytest.raises(RuntimeError):
        asyncio.run(rb.run_retrieval_sample(record, llm))


# ----------------------------- ranking -----------------------------

def test_ranking_matches_official_rank_semantics():
    # 9 negatives: an all-negative sweep must cross the gold_wins threshold
    # (rank 16 -> 7 < 9) in the fan order but not in the res order.
    record = _ranking_record(9)
    llm = FakeEvalLLM([{"selection": 1, "reason": "r"}] * 18)  # 9 negatives x 2 orders
    out = asyncio.run(rb.run_ranking_sample(record, llm))
    pred = out["prediction"]
    res, fan = pred["orders"]["res"], pred["orders"]["fan_1_res"]
    # res order: gold is candidate_1, selection=1 -> gold wins all -> rank 16.
    assert res["rank"] == 16 and res["gold_wins"] is True
    assert all(c["selected"] == "gold" for c in res["comparisons"])
    # fan order: negative is candidate_1, selection=1 -> negative wins all -> rank 7.
    assert fan["rank"] == 7 and fan["gold_wins"] is False
    assert all(c["selected"] == "negative" for c in fan["comparisons"])
    assert set(pred) >= {"sample_id", "model", "orders"}


def test_ranking_parse_failure_defaults_to_1_like_official():
    record = _ranking_record(1)
    llm = FakeEvalLLM([RuntimeError("bad json"), {"selection": 1, "reason": "r"}])
    out = asyncio.run(rb.run_ranking_sample(record, llm))
    pred = out["prediction"]
    res = pred["orders"]["res"]
    assert res["comparisons"][0]["selection"] == 1
    assert "error" in res["comparisons"][0]
    assert out["eval_extra"]["parse_failures"]
    fan = pred["orders"]["fan_1_res"]
    # In fan order candidate_1 is the negative, so the default picks negative.
    assert fan["comparisons"][0]["selected"] == "negative"


# ---------------------------- generation ----------------------------

def test_generation_mechanical_pick_and_official_schema():
    record = _generation_record()
    llm = FakeEvalLLM([
        {"candidates": [_draft("H-A1", [4, 4, 4, 4]), _draft("H-A2", [2, 3, 3, 2])]},
        {"candidates": [_draft("H-B1", [5, 3, 3, 3])]},  # avg 3.5 < 4.0
    ])
    out = asyncio.run(rb.run_generation_sample(record, llm))
    pred = out["prediction"]
    assert pred["final_hypothesis"] == "H-A1"
    assert pred["final_reasoning"] == "inspiration:Insp A"
    # num_mutations=1 caps each inspiration to its first candidate draft.
    assert len(pred["generated_hypotheses"]) == 2
    first = pred["generated_hypotheses"][0]
    assert first["source"] == "inspiration:Insp A"
    assert first["self_eval"]["scores"] == [4, 4, 4, 4]
    assert set(pred) >= {"sample_id", "model", "generated_hypotheses",
                         "final_hypothesis", "final_reasoning"}
    # Internal-shape fields live in eval_extra, not in the official prediction.
    assert out["eval_extra"]["best_mechanism"] == "m"
    assert out["gold"] == {"gold_hypothesis": "GOLD HYPOTHESIS"}


def test_generation_all_inspirations_failing_raises():
    record = _generation_record()
    llm = FakeEvalLLM([RuntimeError("x"), RuntimeError("y")])
    with pytest.raises(RuntimeError):
        asyncio.run(rb.run_generation_sample(record, llm))


# -------------------- official scorer offline smoke --------------------

def test_official_scorer_smoke_on_our_prediction_format():
    if not (RESEARCHBENCH_DIR / "src").exists():
        pytest.skip("official ResearchBench repo not cloned")
    score_retrieve, _score_generation, score_ranking = rb.load_official_scorers()

    record = _ranking_record(9)
    llm = FakeEvalLLM([{"selection": 1, "reason": "r"}] * 18)
    pred = asyncio.run(rb.run_ranking_sample(record, llm))["prediction"]
    result = score_ranking([pred])
    # res order: gold wins; fan order: swept by negatives -> gold_wins False.
    assert result["summary"]["overall_accuracy"] == pytest.approx(0.5)

    retrieve_record = _retrieve_record(candidates_n=10, gold_n=2)
    retrieve_record["default_params"] = {"window_size": 10, "keep_size": 2, "rounds": 2}
    llm2 = FakeEvalLLM([_pick([0, 1]), RuntimeError("r2 skipped")])
    retrieve_pred = asyncio.run(rb.run_retrieval_sample(retrieve_record, llm2))["prediction"]
    retrieve_result = score_retrieve([retrieve_pred], [retrieve_record])
    # Both round1 picks are gold titles (Paper 0 / Paper 1).
    assert retrieve_result["summary"]["round1"]["gold"] == pytest.approx(1.0)


def test_load_official_scorers_missing_repo_message():
    import eval.researchbench.adapter as module
    original = module.RESEARCHBENCH_DIR
    try:
        module.RESEARCHBENCH_DIR = __import__("pathlib").Path("/nonexistent/rb")
        with pytest.raises(RuntimeError, match="git clone"):
            module.load_official_scorers()
    finally:
        module.RESEARCHBENCH_DIR = original

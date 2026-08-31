"""RINoBench adapter — gold_related_works mode (stage 1).

Benchmark: "Is this Idea Novel? An Automated Benchmark for Judgment of
Research Ideas" (arXiv 2603.10303, LREC 2026). Official repo:
https://github.com/TimSchopf/RINoBench, cloned to eval/benchmarks/RINoBench;
the dataset ships inside the repo (data/final_benchmark_dataset/).

Stage-1 scope: research idea + benchmark-provided related works -> novelty
score + justification. The production killer retrieval is intentionally NOT
invoked: this mode isolates the novelty-judgment capability from retrieval
quality. A "self-retrieval extension" mode may be added later and must be
labelled separately.

Label mapping (deterministic, fixed; never adjusted per run):
    benchmark novelty_score 1..5 -> internal audit verdict
        5, 4 -> confirmed          (novel enough to survive an audit)
        3    -> partially_closed   (aspects exist in prior work)
        2    -> uncertain
        1    -> closed             (not novel)
The evaluator asks the model to output the benchmark's own 1-5 scale directly
(rubric text quoted verbatim from the benchmark's label_descriptions.json);
internal_verdict is a derived auxiliary field so results can be correlated
with the production audit vocabulary. The official metric consumes only
novelty_score.

Metrics: the score-based metrics of the official evaluator
(src/eval/evaluate_predicted_novelty.py) are replicated exactly — per-class
and macro F1 (sklearn f1_score semantics, labels 1..5, zero_division=0) and
mean_absolute_error. Marked "official_metric_replica" in metrics.json. The
official justification metrics (alignment / aspect recall / hallucination)
require GPT-4.1 + deepeval and are NOT computed here; the emitted
official_prediction_format.json can be fed to the official evaluator
unchanged.

Official prediction format (verified against the official evaluator):
    JSON array, one element per test sample in dataset order:
        {"reasoning": "<string>", "novelty_score": <1-5 int>}
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from eval.common import CallStats, chat_json, select_samples
from eval.config import RINOBENCH_DIR, model_info

BENCHMARK = "RINoBench"
MODE = "gold_related_works"
TASK = "novelty"

VERDICT_BY_SCORE: dict[int, str] = {
    5: "confirmed",
    4: "confirmed",
    3: "partially_closed",
    2: "uncertain",
    1: "closed",
}
VERDICT_MAPPING_NOTE = (
    "deterministic fixed mapping 5/4->confirmed, 3->partially_closed, "
    "2->uncertain, 1->closed (eval-harness-v1; never adjusted per run)"
)

_NOVELTY_SYSTEM = (
    "You are an expert researcher experienced in judging the novelty of a "
    "research idea."
)


class NoveltyJudgment(BaseModel):
    novelty_score: int = Field(ge=1, le=5)
    reasoning: str
    known_aspects: str = Field(default="", description="what the related works already establish")
    novelty_aspects: str = Field(default="", description="what the idea adds beyond the related works")
    abstained: bool = Field(
        default=False,
        description="true if the related works are insufficient to make a reliable judgment; "
                    "novelty_score is still required as the best-effort estimate")


def data_path(split: str) -> Path:
    path = RINOBENCH_DIR / "data" / "final_benchmark_dataset" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Clone the official repo: "
            f"git clone https://github.com/TimSchopf/RINoBench {RINOBENCH_DIR}"
        )
    return path


def load_rubric() -> list[str]:
    """The official 1-5 rubric, quoted verbatim from the benchmark data."""
    path = RINOBENCH_DIR / "data" / "final_benchmark_dataset" / "label_descriptions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Clone the official repo: "
            f"git clone https://github.com/TimSchopf/RINoBench {RINOBENCH_DIR}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(split: str = "test", limit: int | None = None,
                 seed: int | None = None) -> list[dict]:
    samples = json.loads(data_path(split).read_text(encoding="utf-8"))
    return select_samples(samples, limit, seed)


def internal_verdict_for(score: int) -> str:
    try:
        return VERDICT_BY_SCORE[int(score)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"novelty score {score!r} outside the official 1-5 scale") from exc


def _novelty_prompt(record: dict, works: list[dict], rubric: list[str]) -> str:
    idea = record.get("research_idea") or {}
    works_block = "\n".join(
        f"{idx + 1}. {work.get('title', '')}\n   Abstract: {(work.get('abstract') or '')[:600]}"
        for idx, work in enumerate(works)
    )
    rubric_block = "\n".join(f"{idx + 1}: {text}" for idx, text in enumerate(rubric))
    return (
        "You are judging the novelty of a research idea against a list of related works.\n\n"
        "Research idea:\n"
        f"- Objective: {idea.get('objective', '')}\n"
        f"- Problem statement: {idea.get('problem_statement', '')}\n"
        f"- Solution approach: {idea.get('solution_approach', '')}\n\n"
        f"Related works (title + abstract):\n{works_block}\n\n"
        f"Novelty rubric:\n{rubric_block}\n\n"
        "Instructions: identify what the related works already establish (known aspects) "
        "and what the idea adds beyond them (novelty aspects); then assign the single "
        "best-fitting rubric score. If the related works are clearly insufficient to make "
        "a reliable judgment, set \"abstained\": true and still provide your best-effort "
        "novelty_score.\n\n"
        'Output JSON: {"known_aspects": "<2-4 sentences>", "novelty_aspects": "<2-4 sentences>", '
        '"reasoning": "<2-4 sentence justification for the score>", "novelty_score": <1-5>, '
        '"abstained": <true|false>}'
    )


async def run_novelty_sample(record: dict, llm, stats: CallStats | None = None, *,
                             max_related_works: int = 40,
                             rubric: list[str] | None = None) -> dict:
    rubric = rubric if rubric is not None else load_rubric()
    all_works = list(record.get("related_works") or [])
    works = all_works[:max_related_works]
    if not works:
        raise ValueError("sample has no related works")

    judgment = await chat_json(
        llm,
        [
            {"role": "system", "content": _NOVELTY_SYSTEM},
            {"role": "user", "content": _novelty_prompt(record, works, rubric)},
        ],
        NoveltyJudgment,
        temperature=0.0,
        stats=stats,
    )
    verdict = internal_verdict_for(judgment.novelty_score)

    prediction = {
        "novelty_score": judgment.novelty_score,
        "reasoning": judgment.reasoning,
        "internal_verdict": verdict,
        "verdict_mapping": VERDICT_MAPPING_NOTE,
        "known_aspects": judgment.known_aspects,
        "novelty_aspects": judgment.novelty_aspects,
        "abstained": judgment.abstained,
        "related_works_used": len(works),
        "related_works_truncated": len(all_works) > max_related_works,
    }
    official_element = {"reasoning": judgment.reasoning, "novelty_score": judgment.novelty_score}
    gold = {"novelty_score": record.get("novelty_score")}
    return {"prediction": prediction, "official_element": official_element,
            "gold": gold, "eval_extra": {}}


# --------------------------------------------------------------------------
# Official-metric replica (score-based part of src/eval/evaluate_predicted_novelty.py)
# --------------------------------------------------------------------------

_LABELS = [1, 2, 3, 4, 5]


def _pure_f1_mae(predicted: list[int], gold: list[int]) -> tuple[dict[str, float], float, float]:
    """Same formula as sklearn f1_score(average=None/macro, zero_division=0)
    and mean_absolute_error, without the sklearn dependency."""
    per_class: dict[int, float] = {}
    for label in _LABELS:
        tp = sum(1 for p, g in zip(predicted, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[label] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    macro = sum(per_class[label] for label in _LABELS) / len(_LABELS)
    mae = (sum(abs(p - g) for p, g in zip(predicted, gold)) / len(predicted)) if predicted else 0.0
    return {str(label): value for label, value in per_class.items()}, macro, mae


def compute_score_metrics(predicted: list[int], gold: list[int]) -> dict:
    """Replicates the official evaluator's score-based metrics.

    Uses sklearn (identical to the official script) when available, otherwise
    the pure-python formula above; both are labelled official_metric_replica.
    Justification metrics (GPT-4.1 + deepeval) are out of scope here.
    """
    if len(predicted) != len(gold):
        raise ValueError("predicted/gold length mismatch")
    try:
        from sklearn.metrics import f1_score, mean_absolute_error
        per_class = f1_score(gold, predicted, labels=_LABELS, average=None, zero_division=0)
        macro = f1_score(gold, predicted, labels=_LABELS, average="macro", zero_division=0)
        mae = mean_absolute_error(gold, predicted)
        implementation = "sklearn (identical to official evaluator)"
        f1_scores = {str(label): float(per_class[idx]) for idx, label in enumerate(_LABELS)}
    except ImportError:
        implementation = "pure-python replica of sklearn formulas (sklearn not installed)"
        f1_scores, macro, mae = _pure_f1_mae(predicted, gold)
    return {
        "implementation": implementation,
        "f1_scores": f1_scores,
        "f1_macro": float(macro),
        "mean_absolute_error": float(mae),
    }


def build_metrics(processed: list[dict]) -> dict:
    """Metrics wrapper for a rinobench run. `processed` holds prediction
    records with parse_status == ok, in dataset order."""
    predicted = [int(rec["prediction"]["novelty_score"]) for rec in processed]
    gold = [int(rec["gold"]["novelty_score"]) for rec in processed]
    abstention_rate = (
        sum(1 for rec in processed if rec.get("prediction", {}).get("abstained")) / len(processed)
        if processed else 0.0)
    metrics: dict = {
        "benchmark": BENCHMARK,
        "task": TASK,
        "mode": MODE,
        "metric_source": (
            "official_metric_replica of src/eval/evaluate_predicted_novelty.py "
            "(score-based metrics only; justification metrics need GPT-4.1 + deepeval)"),
        "official_metric_replica": compute_score_metrics(predicted, gold),
        "abstention_rate": round(abstention_rate, 4),
        "note": (
            "F1 values are in [0,1]; the official paper reports the same metric "
            "in percent (e.g. 17.2 == 17.2%), so multiply by 100 for display. "
            "Protocol caveats before comparing to paper numbers: our run uses "
            "the local judge-free score metrics on the same test split and "
            "1-5 labels, but model/preprocessing differ and justification "
            "metrics are not computed — use these numbers for V1->V2 vertical "
            "comparison, not leaderboard comparison. The emitted "
            "official_prediction_format.json can be scored by the official "
            "evaluator unchanged (expects a JSON array of "
            "{reasoning, novelty_score} in dataset order)."),
    }
    return metrics

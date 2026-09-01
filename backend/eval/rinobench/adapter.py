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
from typing import Literal

from pydantic import BaseModel, Field

from eval.common import CallStats, chat_json, select_samples
from eval.config import RINOBENCH_DIR, model_info

BENCHMARK = "RINoBench"
MODE = "gold_related_works"
TASK = "novelty"

NOVELTY_POLICY_V1 = "holistic-v1"
NOVELTY_POLICY_V3 = "criterion-first-v3"
NOVELTY_POLICY_V4A = "forced-binary-coverage-v4a"

# --------------------------------------------------------------------------
# V3 pre-registered hypothesis (frozen before any V3 result was seen):
# H-V3: Direct holistic novelty scoring systematically overweights
# plausibility / proposal completeness and underweights whether the claimed
# contribution is actually distinct from the closest prior work.
# Decomposing novelty judgment into closest-work coverage and residual
# substantive delta before ordinal scoring should improve novelty
# discrimination, especially on low-novelty classes.
# --------------------------------------------------------------------------

# Frozen ordinal semantics for V3 (fixed wording; never tuned per run).
ORDINAL_SEMANTICS_V3 = (
    "Score 1: The essential contribution is already present in the closest "
    "prior work, or the claimed novelty is unsupported.\n"
    "Score 2: Most of the core contribution is already covered. Only a "
    "limited modification, recombination, parameterization, or "
    "implementation-level difference remains.\n"
    "Score 3: There is a partially distinct contribution, but substantial "
    "parts are already covered, or the importance of the residual delta "
    "remains uncertain.\n"
    "Score 4: The general research direction exists, but a major mechanism, "
    "formulation, capability, or contribution remains materially distinct.\n"
    "Score 5: The closest prior works do not cover the core "
    "mechanism/contribution, and there is a clear, grounded, substantive "
    "research delta."
)

# Frozen ban list for V3 (presentation signals must not drive the score).
PRESENTATION_SIGNALS_BAN = (
    "Do NOT judge novelty by: idea length, proposal completeness, the number "
    "of formulas, the number of parameters, specific percentages, complex "
    "framework names, whether it reads like a paper, how plausible it sounds, "
    "or how fluent it is. These indicate only presentation quality / "
    "plausibility. quality != novelty, plausibility != novelty, "
    "specificity != novelty. Ask only: relative to the related works, what "
    "does this idea actually ADD?"
)

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


class NoveltyJudgmentV3(BaseModel):
    """Criterion-first novelty judgment (decomposed before ordinal scoring)."""

    closest_work: str = Field(description="the single closest prior work to the idea")
    core_coverage: Literal["none", "partial", "substantial", "near_complete"] = Field(
        description="how completely the closest work covers the idea's core contribution")
    residual_delta: str = Field(
        description="what the idea adds BEYOND the closest work; explicit if nothing")
    delta_grounded: bool = Field(
        description="the residual delta is anchored and self-consistent, not invented "
                    "details or unsupported parameter claims")
    delta_substantive: bool = Field(
        description="the residual delta is a materially different mechanism/formulation/"
                    "capability, not a parameterization, re-skin, or implementation detail")
    reasoning: str
    novelty_score: int = Field(ge=1, le=5)
    abstained: bool = False


# V4a pre-registered hypothesis (frozen before any V4a result was seen):
# H-V4a: Forcing the coverage head into a discrete YES/NO decision — removing
# the middle-category escape that V3 showed absorbs hedging (coverage
# collapsed to 87% "substantial", none=0, grounded 100% true) — restores
# discrimination at the coverage head, measured by
#     CoverageDiscrimination = P(YES | gold<=2) vs P(YES | gold>=4),
# and improves novelty discrimination without excessively sacrificing
# high-novelty recall. If YES-rate instead collapses toward ~92% across all
# gold levels, forced choice alone is insufficient and evidence anchoring
# (V4b) is the next single-variable experiment.

class NoveltyJudgmentV4A(BaseModel):
    """Forced-binary coverage novelty judgment (single-variable change from
    V3: the hedging-prone 4-way coverage head becomes an unconditional
    YES/NO; everything else follows the V3 structure)."""

    closest_work: str = Field(description="the single closest prior work to the idea")
    covers_core_contribution: bool = Field(
        description="UNCONDITIONAL YES or NO: does the closest prior work already "
                    "contain the idea's core claimed contribution?")
    covered_core_mechanism: str = Field(
        default="",
        description="if YES: which mechanism/formulation of the idea already exists "
                    "there; if NO: the nearest thing the closest work does contain")
    residual_delta: str = Field(
        description="what the idea adds BEYOND the closest work; explicit if nothing")
    delta_grounded: bool = Field(
        description="the residual delta is anchored and self-consistent, not invented "
                    "details or unsupported parameter claims")
    delta_substantive: bool = Field(
        description="the residual delta is a materially different mechanism/formulation/"
                    "capability, not a parameterization, re-skin, or implementation detail")
    reasoning: str
    novelty_score: int = Field(ge=1, le=5)
    abstained: bool = False


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


# Frozen ordinal semantics for V4a (single variable: binary coverage head;
# YES/NO branches are disjoint so every score requires a committed decision).
ORDINAL_SEMANTICS_V4A = (
    "If covers_core_contribution = YES (the closest prior work already contains "
    "the core claimed contribution):\n"
    "  - only a trivial restatement or a pure parameter-level difference remains "
    "-> Score 1\n"
    "  - only a limited modification or recombination remains -> Score 2\n"
    "  - a genuine but incremental contribution remains -> Score 3\n"
    "If covers_core_contribution = NO (the closest prior work does NOT contain "
    "the core claimed contribution):\n"
    "  - the residual delta is ungrounded or non-substantive -> Score 3\n"
    "  - the residual delta is grounded and materially substantive -> Score 4\n"
    "  - the residual delta is grounded, substantive, and opens a materially new "
    "research direction -> Score 5"
)


def _v4a_novelty_prompt(record: dict, works: list[dict]) -> str:
    idea = record.get("research_idea") or {}
    works_block = "\n".join(
        f"{idx + 1}. {work.get('title', '')}\n   Abstract: {(work.get('abstract') or '')[:600]}"
        for idx, work in enumerate(works)
    )
    return (
        "You are judging the novelty of a research idea against related works. "
        "The single most important judgment comes FIRST and is BINARY — you must "
        "commit, there is no middle option:\n\n"
        f"Research idea:\n"
        f"- Objective: {idea.get('objective', '')}\n"
        f"- Problem statement: {idea.get('problem_statement', '')}\n"
        f"- Solution approach: {idea.get('solution_approach', '')}\n\n"
        f"Related works (title + abstract):\n{works_block}\n\n"
        "STEP 1 (forced binary): Identify the single closest prior work "
        "(closest_work), then answer with an unconditional YES or NO: does that "
        "work already contain the idea's CORE claimed contribution? Judge the "
        "contribution itself, not surface similarity of topic.\n"
        "STEP 2: If YES, describe which core mechanism/formulation of the idea "
        "already exists there (covered_core_mechanism). If NO, describe the "
        "nearest thing the closest work does contain.\n"
        "STEP 3: State the residual delta (residual_delta) — what the idea adds "
        "beyond the closest work. If nothing remains, say so explicitly.\n"
        "STEP 4: Judge the residual delta: delta_grounded (anchored, "
        "self-consistent, not invented details) and delta_substantive (a "
        "materially different mechanism/formulation/capability, not a "
        "parameterization or re-skin).\n"
        "STEP 5: Map to the ordinal score using EXACTLY these rules:\n"
        f"{ORDINAL_SEMANTICS_V4A}\n\n"
        f"{PRESENTATION_SIGNALS_BAN}\n\n"
        "Only abstain (abstained=true) if the related works are insufficient to "
        "determine coverage, or the idea itself is materially underspecified; even "
        "then, still output your best-effort novelty_score.\n\n"
        'Output JSON: {"closest_work": "<title or 1-line description>", '
        '"covers_core_contribution": <true|false>, "covered_core_mechanism": '
        '"<1-2 sentences>", "residual_delta": "<1-3 sentences>", '
        '"delta_grounded": <bool>, "delta_substantive": <bool>, "reasoning": '
        '"<2-4 sentence justification for the score>", "novelty_score": <1-5>, '
        '"abstained": <bool>}'
    )


def _v3_novelty_prompt(record: dict, works: list[dict]) -> str:
    idea = record.get("research_idea") or {}
    works_block = "\n".join(
        f"{idx + 1}. {work.get('title', '')}\n   Abstract: {(work.get('abstract') or '')[:600]}"
        for idx, work in enumerate(works)
    )
    return (
        "You are judging the novelty of a research idea against a list of related "
        "works. Work in this EXACT order:\n\n"
        f"Research idea:\n"
        f"- Objective: {idea.get('objective', '')}\n"
        f"- Problem statement: {idea.get('problem_statement', '')}\n"
        f"- Solution approach: {idea.get('solution_approach', '')}\n\n"
        f"Related works (title + abstract):\n{works_block}\n\n"
        "1. Identify the single closest prior work (closest_work) — the work whose "
        "core contribution is nearest to the idea.\n"
        "2. Assess how completely it covers the idea's core contribution "
        "(core_coverage: none | partial | substantial | near_complete).\n"
        "3. State the residual delta: what the idea adds BEYOND that closest work "
        "(residual_delta). If nothing remains, say so explicitly.\n"
        "4. Judge whether the residual delta is grounded (delta_grounded): anchored, "
        "self-consistent, checkable — not invented details or unsupported parameter "
        "claims.\n"
        "5. Judge whether the residual delta is substantive (delta_substantive): a "
        "materially different mechanism, formulation, or capability — not a "
        "parameterization, re-skin, or implementation detail.\n"
        "6. Map to the ordinal novelty score using EXACTLY these definitions:\n"
        f"{ORDINAL_SEMANTICS_V3}\n\n"
        f"{PRESENTATION_SIGNALS_BAN}\n\n"
        "Only abstain (abstained=true) if the related works are insufficient to "
        "determine coverage, or the idea itself is materially underspecified; even "
        "then, still output your best-effort novelty_score.\n\n"
        'Output JSON: {"closest_work": "<title or 1-line description>", '
        '"core_coverage": "<none|partial|substantial|near_complete>", '
        '"residual_delta": "<1-3 sentences>", "delta_grounded": <bool>, '
        '"delta_substantive": <bool>, "reasoning": "<2-4 sentence justification '
        'for the score>", "novelty_score": <1-5>, "abstained": <bool>}'
    )


async def run_novelty_sample(record: dict, llm, stats: CallStats | None = None, *,
                             max_related_works: int = 40,
                             rubric: list[str] | None = None,
                             novelty_policy: str = NOVELTY_POLICY_V1) -> dict:
    if novelty_policy not in (NOVELTY_POLICY_V1, NOVELTY_POLICY_V3, NOVELTY_POLICY_V4A):
        raise ValueError(f"unknown novelty policy: {novelty_policy!r}")
    all_works = list(record.get("related_works") or [])
    works = all_works[:max_related_works]
    if not works:
        raise ValueError("sample has no related works")

    if novelty_policy == NOVELTY_POLICY_V3:
        prompt = _v3_novelty_prompt(record, works)
        schema: type = NoveltyJudgmentV3
    elif novelty_policy == NOVELTY_POLICY_V4A:
        prompt = _v4a_novelty_prompt(record, works)
        schema = NoveltyJudgmentV4A
    else:
        rubric = rubric if rubric is not None else load_rubric()
        prompt = _novelty_prompt(record, works, rubric)
        schema = NoveltyJudgment

    judgment = await chat_json(
        llm,
        [
            {"role": "system", "content": _NOVELTY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        schema,
        temperature=0.0,
        stats=stats,
    )
    verdict = internal_verdict_for(judgment.novelty_score)

    prediction: dict = {
        "novelty_score": judgment.novelty_score,
        "reasoning": judgment.reasoning,
        "internal_verdict": verdict,
        "verdict_mapping": VERDICT_MAPPING_NOTE,
        "abstained": judgment.abstained,
        "related_works_used": len(works),
        "related_works_truncated": len(all_works) > max_related_works,
        "novelty_policy": novelty_policy,
    }
    if novelty_policy == NOVELTY_POLICY_V3:
        v3 = judgment  # type: NoveltyJudgmentV3
        prediction.update({
            "closest_work": v3.closest_work,
            "core_coverage": v3.core_coverage,
            "residual_delta": v3.residual_delta,
            "delta_grounded": v3.delta_grounded,
            "delta_substantive": v3.delta_substantive,
        })
    elif novelty_policy == NOVELTY_POLICY_V4A:
        v4 = judgment  # type: NoveltyJudgmentV4A
        prediction.update({
            "closest_work": v4.closest_work,
            "covers_core_contribution": v4.covers_core_contribution,
            "covered_core_mechanism": v4.covered_core_mechanism,
            "residual_delta": v4.residual_delta,
            "delta_grounded": v4.delta_grounded,
            "delta_substantive": v4.delta_substantive,
        })
    else:
        prediction.update({
            "known_aspects": judgment.known_aspects,
            "novelty_aspects": judgment.novelty_aspects,
        })
    official_element = {"reasoning": judgment.reasoning, "novelty_score": judgment.novelty_score}
    gold = {"novelty_score": record.get("novelty_score")}
    return {"prediction": prediction, "official_element": official_element,
            "gold": gold, "eval_extra": {"novelty_policy": novelty_policy}}


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


def coverage_discrimination(processed: list[dict]) -> dict:
    """Pre-registered V4a diagnostic: does the forced binary coverage head
    actually discriminate?

    Returns P(covers_core_contribution=YES | gold<=2) vs
    P(YES | gold>=4). Under hedging collapse these converge; a working
    coverage head shows a clear gap (low-novelty ideas answered YES more
    often). Policies without a binary coverage field (v1/v3) yield None.
    """
    low = [r for r in processed if (r.get("gold") or {}).get("novelty_score", 99) <= 2]
    high = [r for r in processed if (r.get("gold") or {}).get("novelty_score", -1) >= 4]

    def rate(subset: list[dict]) -> float | None:
        eligible = [r for r in subset if "covers_core_contribution" in r["prediction"]]
        if not eligible:
            return None
        return sum(1 for r in eligible if r["prediction"]["covers_core_contribution"]) / len(eligible)

    return {
        "yes_rate_gold_le2": rate(low),
        "yes_rate_gold_ge4": rate(high),
        "n_low": len(low),
        "n_high": len(high),
    }


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

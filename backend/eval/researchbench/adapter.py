"""ResearchBench adapter — official-compatible mode.

Benchmark: "ResearchBench: Benchmarking LLMs in Scientific Discovery via
Inspiration-Based Task Decomposition" (arXiv 2503.21248, ACL 2026 Findings).
Official repo: https://github.com/ankitala/ResearchBench (MIT), cloned to
eval/benchmarks/ResearchBench. Data license: CC BY-NC-4.0 (non-commercial).

What this adapter does (and does not do):

- retrieval: the OFFICIAL setting is a closed candidate pool (75 candidates,
  2-round window funnel, window_size/keep_size/rounds from default_params).
  We follow that protocol with our own LLM as the windowed selector, emit the
  official prediction schema, and score with the official `score_retrieve`.
  The production web-search stack (SearchService -> OpenAlex/S2/...) is NOT
  used here: running web search against a closed-pool benchmark would not be
  an official score. Web-search ranking is a future "open-web extension
  mode" and must be reported under its own mode label.

- generation: input is research_question + the OFFICIAL gold_inspirations.
  A thin prompt adapter produces hypotheses in the shape of our internal
  hypothesis fields (mechanism / measurable_outcome / falsification_condition
  + 4-way self-eval), then a deterministic mechanical pick selects
  final_hypothesis. Scored with the official `score_generation`; the judge
  client is injectable and is bridged to the app LLM provider (judge model
  name recorded in metrics.json). The production chain (Search -> Evidence ->
  Gap audit -> Intervention -> Experiment) is intentionally NOT started.

- ranking: official pairwise protocol (gold vs each negative, both
  presentation orders, rank starts at 16, each negative win decrements it,
  gold_wins = rank >= 9 — semantics mirrored exactly from the official
  src/researchbench/ranking.py). Our own pairwise prompt is used; this is a
  benchmark-specific thin ranking wrapper, not the production triage. Scored
  with the official `score_ranking`.

Official scorers are imported as pure functions from the cloned official
repo (zero-dependency package; `openai` only needed by their own runner,
which we do not use).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval.common import AsyncBridge, CallStats, chat_json, select_samples
from eval.config import RESEARCHBENCH_DIR, model_info

# Official task name -> data file name / full-data directory.
DATA_FILES = {
    "retrieval": "retrieve.jsonl",
    "generation": "generation.jsonl",
    "ranking": "ranking.jsonl",
}
DATA_DIRS = {"retrieval": "retrieve", "generation": "generation", "ranking": "ranking"}

MODE = "official_compatible"
BENCHMARK = "ResearchBench"

_GATED_DATA_HELP = (
    "Full-split data is gated on HuggingFace (ankilok/ResearchBench). "
    "`huggingface-cli login`, accept the license, then: "
    "huggingface-cli download ankilok/ResearchBench --repo-type dataset "
    "--include \"{retrieve,generation,ranking}/*.jsonl\" --local-dir "
    + str(RESEARCHBENCH_DIR / "data")
)


# --------------------------------------------------------------------------
# Official scorer loading (pure functions from the cloned official repo)
# --------------------------------------------------------------------------

def load_official_scorers():
    """Import the official score_* functions; raise a clear error if the
    official repo is not cloned."""
    src = RESEARCHBENCH_DIR / "src"
    if not src.exists():
        raise RuntimeError(
            f"Official ResearchBench repo not found at {src}. "
            f"Clone it: git clone https://github.com/ankitala/ResearchBench {RESEARCHBENCH_DIR}"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from researchbench.generation import score_generation
    from researchbench.ranking import score_ranking
    from researchbench.retrieve import score_retrieve

    return score_retrieve, score_generation, score_ranking


class OfficialJudgeClient:
    """Bridges our async LLM provider to the official scorer's sync
    ModelClient interface (used by score_generation's LLM judge)."""

    is_mock = False

    def __init__(self, llm) -> None:
        self._llm = llm
        self._bridge = AsyncBridge()

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        return self._bridge.run(
            self._llm.chat([{"role": "user", "content": prompt}], temperature=temperature))

    def close(self) -> None:
        self._bridge.close()


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def data_path(task: str, split: str) -> Path:
    if task not in DATA_FILES:
        raise ValueError(f"unknown ResearchBench task: {task!r}")
    if split == "tiny":
        path = RESEARCHBENCH_DIR / "data" / "tiny" / DATA_FILES[task]
    elif split == "full":
        path = RESEARCHBENCH_DIR / "data" / DATA_DIRS[task] / DATA_FILES[task]
    else:
        raise ValueError(f"unknown split {split!r} (use 'tiny' or 'full')")
    if not path.exists():
        if split == "full":
            raise FileNotFoundError(f"{path} not found.\n{_GATED_DATA_HELP}")
        raise FileNotFoundError(
            f"{path} not found. Clone the official repo: "
            f"git clone https://github.com/ankitala/ResearchBench {RESEARCHBENCH_DIR}"
        )
    return path


def load_samples(task: str, split: str = "tiny", limit: int | None = None,
                 seed: int | None = None) -> list[dict]:
    lines = data_path(task, split).read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    return select_samples(samples, limit, seed)


# --------------------------------------------------------------------------
# Structured-output schemas (our client parses JSON; no regex scraping)
# --------------------------------------------------------------------------

class WindowPick(BaseModel):
    index: int = Field(description="0-based candidate index within this window")
    reason: str = ""


class WindowSelection(BaseModel):
    selections: list[WindowPick] = Field(min_length=1)


class HypothesisDraft(BaseModel):
    hypothesis: str
    mechanism: str = ""
    measurable_outcome: str = ""
    falsification_condition: str = ""
    self_validness: int = Field(ge=1, le=5)
    self_novelty: int = Field(ge=1, le=5)
    self_significance: int = Field(ge=1, le=5)
    self_specificity: int = Field(ge=1, le=5)
    reasoning: str = ""


class InspirationDrafts(BaseModel):
    candidates: list[HypothesisDraft] = Field(min_length=1)


class PairwiseSelection(BaseModel):
    selection: Literal[1, 2]
    reason: str = ""


class CandidateCriteria(BaseModel):
    on_topic: bool = Field(description="directly addresses the research question's core problem")
    grounded: bool = Field(
        description="mechanisms/data/numbers are anchored and self-consistent; invented "
                    "quantities, unsupported parameter claims, or methods asserted without "
                    "basis are red flags")
    mechanism_checkable: bool = Field(
        description="names a causal mechanism one could actually check, not just a list of "
                    "techniques or a goal statement")
    falsifiable: bool = Field(
        description="implies a concrete, distinguishable test or comparison")
    note: str = ""


class CriterionPairwiseSelection(BaseModel):
    candidate_1: CandidateCriteria
    candidate_2: CandidateCriteria
    selection: Literal[1, 2]
    reason: str = ""


# --------------------------------------------------------------------------
# Task A: retrieval (official candidate pool + official window protocol)
# --------------------------------------------------------------------------

_SELECTION_SYSTEM = (
    "You are an expert research assistant judging which papers are the most "
    "relevant inspirations for a research question."
)

_OFFICIAL_ENTRY_KEYS = ("title", "abstract", "index", "label", "source_file")


def _official_entry(candidate: dict, reason: str) -> dict:
    entry = {key: candidate.get(key) for key in _OFFICIAL_ENTRY_KEYS}
    entry["reason"] = reason
    return entry


def _selection_prompt(research_question: str, window: list[dict], keep_size: int) -> str:
    lines = []
    for local_idx, candidate in enumerate(window):
        abstract = (candidate.get("abstract") or "").strip()
        if len(abstract) > 800:
            abstract = abstract[:800] + "..."
        lines.append(f"{local_idx}. Title: {candidate.get('title', '')}\n   Abstract: {abstract}")
    return (
        f"Research question:\n{research_question}\n\n"
        f"Candidate papers (numbered 0-{len(window) - 1}):\n" + "\n".join(lines) + "\n\n"
        f"Select the {keep_size} candidates most likely to serve as productive INSPIRATIONS "
        "for new research on this question (transferable ideas, methods, or findings).\n\n"
        'Output JSON: {"selections": [{"index": <int>, "reason": "<one short sentence>"}]} '
        f"with exactly {keep_size} entries."
    )


async def run_retrieval_sample(record: dict, llm, stats: CallStats | None = None) -> dict:
    """Official window funnel with our LLM as selector -> official schema."""
    params = record.get("default_params") or {}
    window_size = int(params.get("window_size", 15))
    keep_size = int(params.get("keep_size", 3))
    rounds = int(params.get("rounds", 2))

    current: list[dict] = list(record.get("candidates") or [])
    rounds_entries: list[dict] = []
    round_titles: dict[int, list[str]] = {}
    window_errors: list[str] = []

    for round_num in range(1, rounds + 1):
        if not current:
            raise RuntimeError(f"round {round_num}: no candidates carried over from round {round_num - 1}")
        windows = [current[i:i + window_size] for i in range(0, len(current), window_size)]
        selected: list[dict] = []
        for window_idx, window in enumerate(windows):
            picks: list[tuple[dict, str]] = []
            try:
                result = await chat_json(
                    llm,
                    [
                        {"role": "system", "content": _SELECTION_SYSTEM},
                        {"role": "user",
                         "content": _selection_prompt(record["research_question"], window, keep_size)},
                    ],
                    WindowSelection,
                    temperature=0.0,
                    stats=stats,
                )
                for pick in result.selections[:keep_size]:
                    if 0 <= pick.index < len(window):
                        picks.append((window[pick.index], pick.reason))
                    else:
                        window_errors.append(
                            f"round{round_num}/window{window_idx}: index {pick.index} out of range")
            except Exception as exc:
                # A failed window shrinks the funnel for this sample; it does
                # not abort the run (the sample error path handles total loss).
                window_errors.append(
                    f"round{round_num}/window{window_idx}: {type(exc).__name__}: {exc}"[:500])
            if picks:
                # One official entry per window, holding that window's picks.
                rounds_entries.append({
                    "round": round_num,
                    "window": window_idx,
                    "selected": [_official_entry(candidate, reason) for candidate, reason in picks],
                })
            selected.extend(candidate for candidate, _ in picks)
        if not selected and round_num < rounds:
            raise RuntimeError(
                f"round {round_num} produced no selections; window_errors={window_errors[:3]}")
        round_titles[round_num] = [entry.get("title") or "" for entry in selected]
        current = selected

    prediction = {
        "sample_id": record["sample_id"],
        "model": model_info()["model"],
        "rounds": rounds_entries,
        "selected_round1_titles": round_titles.get(1, []),
        "selected_round2_titles": round_titles.get(rounds, []),
    }
    gold = {"gold_titles": record.get("gold_titles"), "label_counts": record.get("label_counts")}
    return {"prediction": prediction, "gold": gold, "eval_extra": {"window_errors": window_errors}}


# --------------------------------------------------------------------------
# Task B: generation (official inspirations + thin hypothesis-prompt adapter)
# --------------------------------------------------------------------------

_GENERATION_SYSTEM = (
    "You are an expert researcher proposing novel, testable scientific "
    "hypotheses grounded in inspiration papers."
)


def _generation_prompt(record: dict, inspiration: dict, num_candidates: int) -> str:
    background = (record.get("background_survey") or "")[:4000]
    return (
        f"Research question:\n{record['research_question']}\n\n"
        f"Background survey (excerpt):\n{background}\n\n"
        f"Inspiration paper: {inspiration.get('title', '')}\n"
        f"Abstract: {(inspiration.get('abstract') or '')[:1200]}\n\n"
        f"Propose {num_candidates} distinct candidate hypotheses inspired by this paper "
        "that could answer the research question. Each must name the causal mechanism, "
        "a measurable outcome, and a falsification condition. Self-score each candidate "
        "1-5 on validness, novelty, significance, specificity.\n\n"
        'Output JSON: {"candidates": [{"hypothesis": "...", "mechanism": "...", '
        '"measurable_outcome": "...", "falsification_condition": "...", '
        '"self_validness": <1-5>, "self_novelty": <1-5>, "self_significance": <1-5>, '
        '"self_specificity": <1-5>, "reasoning": "<one short paragraph>"}]}'
    )


async def run_generation_sample(record: dict, llm, stats: CallStats | None = None, *,
                                num_mutations: int = 2, max_inspirations: int = 3) -> dict:
    params = record.get("default_params") or {}
    num_mutations = int(params.get("num_mutations", num_mutations))
    inspirations = (record.get("gold_inspirations")
                    or record.get("true_retrieve")
                    or [])[:max_inspirations]
    if not inspirations:
        raise ValueError("sample has no gold_inspirations/true_retrieve")

    generated: list[dict] = []
    inspiration_errors: list[str] = []
    for inspiration in inspirations:
        try:
            drafts = await chat_json(
                llm,
                [
                    {"role": "system", "content": _GENERATION_SYSTEM},
                    {"role": "user",
                     "content": _generation_prompt(record, inspiration, num_mutations)},
                ],
                InspirationDrafts,
                temperature=0.0,
                stats=stats,
            )
            for draft in drafts.candidates[:num_mutations]:
                scores = [draft.self_validness, draft.self_novelty,
                          draft.self_significance, draft.self_specificity]
                generated.append({
                    "source": f"inspiration:{inspiration.get('title', '')}",
                    "hypothesis": draft.hypothesis,
                    "self_eval": {
                        "scores": scores,
                        "score_reasons": draft.reasoning,
                        "average_score": round(sum(scores) / 4.0, 4),
                        "raw_response": "",
                    },
                    "_mechanism": draft.mechanism,
                    "_measurable_outcome": draft.measurable_outcome,
                    "_falsification_condition": draft.falsification_condition,
                })
        except Exception as exc:
            inspiration_errors.append(
                f"{(inspiration.get('title') or '')[:60]}: {type(exc).__name__}: {exc}"[:500])
    if not generated:
        raise RuntimeError(f"no hypotheses generated; errors={inspiration_errors[:3]}")

    # Deterministic mechanical pick (our "cheap rank" convention): highest
    # mean self-score, ties broken by order of generation.
    best = max(generated, key=lambda item: item["self_eval"]["average_score"])
    prediction = {
        "sample_id": record["sample_id"],
        "model": model_info()["model"],
        "generated_hypotheses": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in generated
        ],
        "final_hypothesis": best["hypothesis"],
        "final_reasoning": best["source"],
    }
    gold = {"gold_hypothesis": record.get("gold_hypothesis")}
    eval_extra = {
        "inspiration_errors": inspiration_errors,
        "best_mechanism": best["_mechanism"],
        "best_measurable_outcome": best["_measurable_outcome"],
        "best_falsification_condition": best["_falsification_condition"],
    }
    return {"prediction": prediction, "gold": gold, "eval_extra": eval_extra}


# --------------------------------------------------------------------------
# Task C: ranking (official pairwise protocol + our pairwise prompt)
#
# V2 diagnosis (pre-registered before any prompt change, from the V1
# Ranking-50 failure analysis):
#   - gold wins only 14% of pairwise judgments (random baseline ~50%)
#   - outcome is uncorrelated with negative source (fake 0.206 vs
#     model-generated 0.256 gold-win rate) and with length (negatives are
#     longer in 97.5% of WON and 99.9% of LOST comparisons — the dataset
#     simply makes negatives longer)
#   - manual reading of both-orders-lost comparisons shows negatives are
#     "research-proposal style": framework lists, parameter formulas,
#     invented quantified promises (">30%"), while gold hypotheses are
#     compressed, hedged, but anchored claims from real papers
#   => the judge weights PROPOSAL COMPLETENESS / SPECIFICITY THEATER over
#      scientific substance (H-V2 confirmed, mechanism refined: not fluency).
#
# V2 = criterion-first discrimination: score both candidates on substance
# criteria first (anchored mechanisms, falsifiability), explicitly ignore
# packaging (length/detail volume/proposal completeness), then choose.
# --------------------------------------------------------------------------

_RANKING_SYSTEM = "You are an expert researcher judging the quality of research hypotheses."

RANKING_POLICY_V1 = "ranking-pairwise-v1"
RANKING_POLICY_V2 = "ranking-criterion-first-v2"


def _ranking_prompt(research_question: str, candidate_1: str, candidate_2: str) -> str:
    return (
        f"Research question:\n{research_question}\n\n"
        "Two candidate research hypotheses:\n\n"
        f"Candidate 1:\n{candidate_1}\n\n"
        f"Candidate 2:\n{candidate_2}\n\n"
        "Which candidate is the stronger research hypothesis for this question? "
        "Judge by novelty, plausibility, and potential significance.\n\n"
        'Output JSON: {"selection": <1 or 2>, "reason": "<one short sentence>"}'
    )


def _criterion_ranking_prompt(research_question: str, candidate_1: str, candidate_2: str) -> str:
    return (
        f"Research question:\n{research_question}\n\n"
        "Two candidate research hypotheses:\n\n"
        f"Candidate 1:\n{candidate_1}\n\n"
        f"Candidate 2:\n{candidate_2}\n\n"
        "Decide which is the stronger SCIENTIFIC hypothesis. Judge substance only.\n\n"
        "IGNORE completely: length, writing style, formatting, how complete or "
        "detailed the description is, presence of formulas/numbers/framework lists, "
        "and how much it reads like a full research proposal. A longer, more "
        "elaborate proposal is NOT a better hypothesis; real scientific hypotheses "
        "are often short and hedged.\n\n"
        "For EACH candidate judge:\n"
        "1. on_topic — directly addresses the research question's core problem\n"
        "2. grounded — its mechanisms/data/numbers are anchored and self-consistent; "
        "invented quantities, unsupported parameter claims, or methods asserted "
        "without basis are red flags\n"
        "3. mechanism_checkable — names a causal mechanism one could actually check, "
        "not just a list of techniques or a goal statement\n"
        "4. falsifiable — implies a concrete, distinguishable test or comparison\n\n"
        "Then choose the candidate whose CORE CLAIM is more scientifically defensible.\n\n"
        'Output JSON: {"candidate_1": {"on_topic": <bool>, "grounded": <bool>, '
        '"mechanism_checkable": <bool>, "falsifiable": <bool>, "note": "<one short '
        'sentence>"}, "candidate_2": {...}, "selection": <1 or 2>, "reason": "<one '
        'short sentence>"}'
    )


async def run_ranking_sample(record: dict, llm, stats: CallStats | None = None, *,
                             order: str = "both",
                             ranking_policy: str = RANKING_POLICY_V1) -> dict:
    if ranking_policy not in (RANKING_POLICY_V1, RANKING_POLICY_V2):
        raise ValueError(f"unknown ranking policy: {ranking_policy!r}")
    order_names = ["res", "fan_1_res"] if order == "both" else [order]
    negatives = (list(record.get("fake_negative_hypotheses") or [])
                 + list(record.get("model_negative_hypotheses") or []))
    if not negatives:
        raise ValueError("sample has no negative hypotheses")

    parse_failures: list[dict] = []
    orders_out: dict[str, Any] = {}
    for order_name in order_names:
        negative_selection = 2 if order_name == "res" else 1
        rank_count = 16
        comparisons: list[dict] = []
        for negative_index, negative in enumerate(negatives):
            if order_name == "res":
                candidate_1, candidate_2 = record["gold_hypothesis"], negative
            else:
                candidate_1, candidate_2 = negative, record["gold_hypothesis"]
            if ranking_policy == RANKING_POLICY_V2:
                prompt = _criterion_ranking_prompt(
                    record["research_question"], candidate_1, candidate_2)
                schema = CriterionPairwiseSelection
            else:
                prompt = _ranking_prompt(
                    record["research_question"], candidate_1, candidate_2)
                schema = PairwiseSelection
            selection, reason, error = 1, "", None
            try:
                result = await chat_json(
                    llm,
                    [
                        {"role": "system", "content": _RANKING_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    schema,
                    temperature=0.0,
                    stats=stats,
                )
                selection = int(result.selection)
                reason = result.reason
            except Exception as exc:
                # Official parity: the official runner defaults selection to 1
                # when parsing fails; we do the same but record the failure.
                error = f"{type(exc).__name__}: {exc}"[:500]
                parse_failures.append(
                    {"order": order_name, "negative_index": negative_index, "error": error})
            selected = "negative" if selection == negative_selection else "gold"
            if selected == "negative":
                rank_count -= 1
            comparison = {
                "negative_index": negative_index,
                "selection": selection,
                "selected": selected,
                "raw_response": reason,
            }
            if error:
                comparison["error"] = error
            comparisons.append(comparison)
        orders_out[order_name] = {
            "rank": rank_count,
            "gold_wins": rank_count >= 9,
            "comparisons": comparisons,
        }

    prediction = {
        "sample_id": record["sample_id"],
        "model": model_info()["model"],
        "orders": orders_out,
    }
    gold = {"gold_hypothesis": record.get("gold_hypothesis")}
    eval_extra = {"parse_failures": parse_failures, "ranking_policy": ranking_policy}
    return {"prediction": prediction, "gold": gold, "eval_extra": eval_extra}


RUNNERS = {
    "retrieval": run_retrieval_sample,
    "generation": run_generation_sample,
    "ranking": run_ranking_sample,
}


# --------------------------------------------------------------------------
# Official scoring (raw official output is preserved verbatim in metrics)
# --------------------------------------------------------------------------

def score_official(task: str, predictions: list[dict], data_records: list[dict], llm) -> dict:
    """Run the official scorer for a task; returns the raw official output
    plus scorer provenance. Never post-processes official metric values."""
    score_retrieve, score_generation, score_ranking = load_official_scorers()
    if task == "retrieval":
        raw = score_retrieve(predictions, data_records)
        scorer = {"name": "researchbench.score_retrieve (official repo import)", "llm_calls": 0}
    elif task == "ranking":
        raw = score_ranking(predictions)
        scorer = {"name": "researchbench.score_ranking (official repo import)", "llm_calls": 0}
    elif task == "generation":
        calls_before = int(getattr(llm, "call_count", 0) or 0)
        judge = OfficialJudgeClient(llm)
        try:
            raw = score_generation(predictions, data_records, judge)
        finally:
            judge.close()
        from eval.config import judge_policy
        scorer = {
            "name": "researchbench.score_generation (official repo import)",
            "score_kind": "ResearchBench-compatible generation score "
                          "(official formula + local judge; NOT comparable to the "
                          "official paper leaderboard)",
            **judge_policy(),
            "llm_calls": int(getattr(llm, "call_count", 0) or 0) - calls_before,
        }
    else:
        raise ValueError(f"unknown ResearchBench task: {task!r}")
    return {"official_scorer_output": raw, "scorer": scorer}

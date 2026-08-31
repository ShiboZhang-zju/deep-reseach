# RINoBench adapter (gold_related_works mode — stage 1)

Benchmark: [RINoBench](https://arxiv.org/abs/2603.10303) (LREC 2026) —
research idea novelty judgment.
Official repo: https://github.com/TimSchopf/RINoBench, cloned to
`backend/eval/benchmarks/RINoBench/` (gitignored). The dataset ships inside
the repo: `data/final_benchmark_dataset/{train,test,label_descriptions}.json`
(test = 277 samples; novelty gold = 1–5 from ICLR 2022/2023 peer review).
No HF download or gating required. Note: the repo/dataset carry no explicit
open-source license.

## Stage-1 scope

`gold_related_works` mode only: research idea + benchmark-provided related
works (title + abstract, capped at 40 with truncation recorded) → novelty
score + justification.

**The production killer retrieval is NOT invoked.** The purpose of this mode
is to isolate the novelty-judgment capability from retrieval quality. A
`self_retrieval_extension` mode may be added later and must be labelled
separately; `config.json` / `metrics.json` record `self_retrieval: false`
for every stage-1 run.

## Label mapping (deterministic, fixed at eval-harness-v1, never tuned per run)

The evaluator outputs the benchmark's own 1–5 scale directly (rubric text
quoted verbatim from the benchmark's `label_descriptions.json`). An internal
verdict is derived as an auxiliary field:

| benchmark novelty_score | internal verdict |
|---|---|
| 5, 4 | confirmed |
| 3 | partially_closed |
| 2 | uncertain |
| 1 | closed |

Official metrics consume only `novelty_score`; the mapping exists so results
can be correlated with the production audit vocabulary.

## Scoring

`metrics.json` contains `official_metric_replica`: the score-based metrics of
the official evaluator (`src/eval/evaluate_predicted_novelty.py`) replicated
exactly — per-class + macro F1 (sklearn `f1_score`, labels 1..5,
zero_division=0) and `mean_absolute_error`. sklearn is used when installed
(identical to the official script), otherwise a pure-python implementation of
the same formulas (unit-tested to match). These are honest replicas of the
official formulas, not a redefinition.

**Units**: F1 values are stored in [0,1] (sklearn convention); the official
paper reports the same metric in percent (its "17.2" means 17.2%). Display
should show `Macro-F1: 23.4%`. **Protocol caveats**: same test split, same
1-5 labels, judge-free score metrics — but model and preprocessing differ
and the justification metrics are not computed. Use these numbers for
V1→V2 vertical comparison; do NOT claim leaderboard parity without first
aligning the full protocol.

## Abstention

The evaluator may set `"abstained": true` (with a best-effort
`novelty_score`) when the related works are insufficient for a reliable
judgment. `metrics.json` reports `abstention_rate` alongside accuracy so a
future version cannot look better by abstaining more — always read
Quality × Abstention × Cost together. `run_stats.parse_failure_rate`
(unparseable outputs) is reported for every benchmark task.

The official justification metrics (alignment / aspect recall / hallucination)
require GPT-4.1 + `deepeval` and are NOT computed by this harness.

`official_prediction_format.json` is written in the exact format the official
evaluator expects — a JSON array of `{"reasoning", "novelty_score"}` in
dataset order. When the FULL test split (277) is processed, that file can be
scored by the official evaluator unchanged (`official_evaluator_ready: true`
in metrics.json). With `--limit`, the file covers only the processed prefix
and is NOT directly official-evaluator ready (alignment would break).

## Commands

```bash
cd backend
python -m eval.run_eval rinobench --split test --limit 20
python -m eval.run_eval rinobench --split test --limit 100 --seed 42
```

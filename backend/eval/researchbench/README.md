# ResearchBench adapter (official-compatible mode)

Benchmark: [ResearchBench](https://arxiv.org/abs/2503.21248) (ACL 2026 Findings) —
inspiration retrieval / hypothesis generation / hypothesis ranking.
Official repo: https://github.com/ankitala/ResearchBench (MIT), cloned to
`backend/eval/benchmarks/ResearchBench/` (gitignored). Data license:
**CC BY-NC-4.0**, non-commercial research only.

## Data

| split | location | notes |
|---|---|---|
| `tiny` | `data/tiny/{retrieve,generation,ranking}.jsonl` | ships in the official repo, ungated, 12 samples/task — used for smoke runs |
| `full` | `data/{retrieve,generation,ranking}/*.jsonl` | gated on HF (`ankilok/ResearchBench`); download after accepting the license: `huggingface-cli download ankilok/ResearchBench --repo-type dataset --include "{retrieve,generation,ranking}/*.jsonl" --local-dir data` |

## Modes

`official_compatible` (the only mode implemented): official candidate pool /
official protocol / official scorer. Production web search is NOT used —
running web search against a closed-pool benchmark would not yield an
official score. A future `open_web_extension` mode must be labelled
separately and never mixed into official numbers.

| task | input | what runs | scorer (official, imported as pure functions) |
|---|---|---|---|
| retrieval | research_question + fixed 75-candidate pool | official window funnel (window_size=15, keep_size=3, rounds=2 from `default_params`) with our LLM as the windowed selector (structured output, no regex) | `researchbench.score_retrieve` (offline; micro-recall per label, round1/round2; headline = gold hit ratio) |
| generation | research_question + background_survey + OFFICIAL gold_inspirations | thin prompt adapter producing hypotheses in our internal field shape (mechanism / measurable outcome / falsification condition + 4-way self-eval 1-5), then a deterministic mechanical pick | `researchbench.score_generation` (LLM judge; judge is bridged to the app LLM provider, judge model recorded in metrics.json) |
| ranking | research_question + gold + 5 fake + 10 model negatives | official pairwise protocol (both presentation orders; rank starts 16, negative win −1, gold_wins = rank ≥ 9 — mirrored from official `ranking.py`) with our own pairwise prompt | `researchbench.score_ranking` (offline; directional accuracy, both-orders win rates, pair consistency) |

What is intentionally NOT started: the production chain Search → Evidence →
Gap audit → Intervention → Experiment (requires DB/evidence state; not needed
for this benchmark task).

## Prediction formats (official schema, verified against the scorer source)

- retrieval: `{sample_id, model, rounds:[{round, window, selected:[...]}], selected_round1_titles, selected_round2_titles}`
- generation: `{sample_id, model, generated_hypotheses:[{source, hypothesis, self_eval}], final_hypothesis, final_reasoning}`
- ranking: `{sample_id, model, orders:{res:{rank, gold_wins, comparisons}, fan_1_res:{...}}}`

Deviations from the official runner (documented, none affect the scorer):

1. Our client parses structured JSON instead of the official regex markers;
   `raw_response` fields therefore carry our structured reason strings.
2. On a per-window / per-inspiration LLM failure we shrink the funnel for
   that sample (recorded in `window_errors` / `inspiration_errors`) instead
   of aborting; a ranking parse failure defaults `selection=1` exactly like
   the official parser and records the failure explicitly.
3. Generation does not replicate the MOOSE-Chem mutation/self-refine beam
   search — this is OUR generation capability being evaluated; the official
   scorer only consumes `final_hypothesis`.

## Commands

```bash
cd backend
python -m eval.run_eval researchbench --task retrieval --split tiny
python -m eval.run_eval researchbench --task generation --split tiny
python -m eval.run_eval researchbench --task ranking --split tiny --limit 3   # 30 pairwise calls per sample
```

Requires the local LLM endpoint from `.env` to be reachable (same provider
as production). `metrics.json` stores the official scorer output verbatim.

## Artifacts: attempts vs predictions

- `attempts.jsonl` — append-only history of EVERY attempt (success, failure,
  retry). Cost accounting (llm_calls/tokens) covers all attempts.
- `predictions.jsonl` — rewritten after each run: exactly ONE final
  successful prediction per sample_id, dataset order preserved. Errored
  samples appear only in attempts.jsonl. `one sample id -> one final
  prediction` is the invariant the official scorer and compare rely on.

Resume semantics: samples whose latest attempt succeeded are skipped;
errored samples are retried.

## Generation judge freeze

The generation task is scored with the official formula but a LOCAL judge,
so its score is a **ResearchBench-compatible generation score** — it is NOT
comparable to the official paper leaderboard (which differs in generation
pipeline, model, AND judge). The judge configuration is frozen and recorded
in `config.json` / `metrics.json` as `judge_policy`
(`eval-judge-v1`: local venus model, temperature 0). Use these scores for
V1→V2 vertical comparison only; never change judge settings without bumping
`EVAL_JUDGE_POLICY_VERSION`.

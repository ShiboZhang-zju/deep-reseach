"""Frozen-eval harness: external benchmarks + internal validity regression.

Layout:
    eval/config.py       run metadata (policy versions, model, git commit)
    eval/common.py       shared runner utilities (run dir I/O, LLM accounting)
    eval/run_eval.py     unified CLI
    eval/researchbench/  ResearchBench adapter (official-compatible mode)
    eval/rinobench/      RINoBench adapter (gold_related_works mode)
    eval/internal/       internal research-validity regression aggregator
    eval/benchmarks/     cloned official benchmark repos + data (gitignored)

Ground rules: benchmark data / gold labels / official metrics are never
modified; production research pipeline code is never changed to fit a
benchmark; official and extension modes are always labelled in outputs.
"""

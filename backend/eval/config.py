"""Eval run metadata: paths, git commit, production policy-version snapshot."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent          # backend/eval
BACKEND_DIR = EVAL_DIR.parent                       # backend/
BENCHMARKS_DIR = EVAL_DIR / "benchmarks"
RESEARCHBENCH_DIR = BENCHMARKS_DIR / "ResearchBench"
RINOBENCH_DIR = BENCHMARKS_DIR / "RINoBench"
DEFAULT_RESULTS_DIR = BACKEND_DIR / "eval_results"

EVAL_HARNESS_VERSION = "eval-harness-v1"

# Frozen judge configuration for LLM-judged scoring (ResearchBench generation).
# All future eval versions must use the same judge settings so scores are
# comparable across versions (vertical V1 -> V2 comparison only; NOT
# comparable to the official paper leaderboard, which used GPT-4.1-class
# judges and their full generation pipeline).
EVAL_JUDGE_POLICY_VERSION = "eval-judge-v1"


def judge_policy() -> dict:
    """The frozen judge settings, recorded in run config and metrics."""
    return {
        "judge_version": EVAL_JUDGE_POLICY_VERSION,
        "judge_model": model_info()["model"],
        "judge_temperature": 0.0,
    }


def git_commit() -> str:
    """Short commit hash of the working tree (eval reproducibility metadata)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BACKEND_DIR),
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def policy_versions() -> dict[str, str]:
    """Snapshot of the production policy versions the eval ran against.

    Imported live from the production modules so a run config can never claim
    a version the code does not carry.
    """
    from app.agent.steps.audit_gaps import GAP_SEARCH_POLICY_VERSION
    from app.agent.steps.generate_interventions import INTERVENTION_GENERATION_POLICY_VERSION
    from app.agent.steps.generate_minimal_experiments import EXPERIMENT_GENERATION_POLICY_VERSION
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION

    return {
        "gap_mining": GAP_MINING_POLICY_VERSION,
        "gap_search_audit": GAP_SEARCH_POLICY_VERSION,
        "intervention_generation": INTERVENTION_GENERATION_POLICY_VERSION,
        "experiment_generation": EXPERIMENT_GENERATION_POLICY_VERSION,
        "eval_harness": EVAL_HARNESS_VERSION,
    }


def model_info() -> dict[str, str]:
    """The LLM the eval will call (same settings-driven provider as production)."""
    from app.config import settings

    provider = (settings.llm_provider or "").lower()
    model = ""
    if provider == "venus":
        model = settings.venus_llm_model
    elif provider == "openai":
        model = getattr(settings, "openai_model", "")
    return {"model_provider": provider, "model": model}


def build_run_config(
    *,
    benchmark: str,
    task: str,
    mode: str,
    split: str,
    sample_count: int,
    seed: int | None = None,
    limit: int | None = None,
    extra: dict | None = None,
) -> dict:
    cfg: dict = {
        "benchmark": benchmark,
        "task": task,
        "mode": mode,
        "split": split,
        "sample_count": sample_count,
        "seed": seed,
        "limit": limit,
        **model_info(),
        "policy_versions": policy_versions(),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_harness_version": EVAL_HARNESS_VERSION,
    }
    if extra:
        cfg.update(extra)
    return cfg

"""Step: Generate phenomenon-validation plans for surviving gaps (Phase 3F).

Before any intervention or idea is designed, pin down the *phenomenon* each
surviving gap depends on, plus the cheapest experiment that could falsify it and
a kill criterion. This keeps the pipeline from designing a method for a problem
that may not be real or large enough to matter — the single most expensive
failure mode of an auto-generated research idea. The plan does not run the
experiment (the system has no way to), but it forces every direction to carry a
concrete, quantitative falsification target that the user reviews before
committing real effort.
"""

import logging

from pydantic import BaseModel, Field

from app.agent.state import ResearchState
from app.db.models import GapCandidate, GapPhenomenonPlan, ResearchContract
from app.db.repositories import paper_repo

logger = logging.getLogger(__name__)


_PHENOMENON_SYSTEM = """You validate the *phenomenon* a surviving research gap depends on, before any method is designed.

A gap is worth pursuing only if the phenomenon it claims is real and large enough to matter.
Pin that phenomenon down and design the cheapest experiment that could falsify it.

For the gap, produce:
- phenomenon: the concrete, measurable empirical claim the gap rests on (e.g. "models that
  pass sparse tests often fail dense tests on the same program"), NOT a vague statement.
- critical_unknown: the one fact that, if unknown, makes the whole direction uncertain.
- oracle_experiment: the cheapest falsification experiment — what to measure, on what data,
  comparing what against what. It must be runnable without building a new method.
- kill_criterion: the quantitative threshold below which the phenomenon is too small to be
  worth a method paper (e.g. "if fewer than 5% of sparse-pass samples fail dense tests, abandon").
- measurement: the exact metric/quantity that quantifies the phenomenon's size.

Do NOT design a method or a solution. Do NOT invent papers or datasets. If the phenomenon
cannot be made concrete and measurable, say so explicitly rather than padding."""

_PHENOMENON_USER = """Surviving gap:
- Setting: {target_setting}
- Observed problem: {observed_problem}
- Missing capability: {missing_capability}
- Claimed delta: {claimed_delta}
- Nearest prior art: {nearest_prior_art}
- Residual gap: {residual_gap}

Resource constraints:
- GPU available: {gpu_available}
- Allow large benchmark: {allow_large_benchmark}

Produce the phenomenon validation plan."""


class PhenomenonPlanSchema(BaseModel):
    phenomenon: str = Field(min_length=5)
    critical_unknown: str = Field(min_length=5)
    oracle_experiment: str = Field(min_length=5)
    kill_criterion: str = Field(min_length=5)
    measurement: str = Field(min_length=3)


async def generate_phenomenon_plans(
    db,
    state: ResearchState,
    llm,
    task_id: str,
) -> list[str]:
    """Generate and persist one phenomenon plan per surviving gap.

    Idempotent: a gap that already has a plan is skipped, so a resumed run does
    not re-pay the LLM call. Failure of any single plan is non-fatal — the gap
    still proceeds to intervention, just without a pinned-down falsification
    target.
    """
    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    if not contract:
        return []

    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == contract.id,
        GapCandidate.status == "surviving",
    ).all()

    created_ids: list[str] = []
    for gap in gaps:
        existing = db.query(GapPhenomenonPlan).filter(
            GapPhenomenonPlan.gap_id == gap.id,
        ).first()
        if existing:
            created_ids.append(existing.id)
            continue
        try:
            plan = await llm.chat_json([
                {"role": "system", "content": _PHENOMENON_SYSTEM},
                {"role": "user", "content": _PHENOMENON_USER.format(
                    target_setting=gap.target_setting or "(not specified)",
                    observed_problem=gap.observed_problem or "(not specified)",
                    missing_capability=gap.missing_capability or "(not specified)",
                    claimed_delta=gap.claimed_delta or "(not specified)",
                    nearest_prior_art=gap.nearest_prior_art_title or "(none found)",
                    residual_gap=gap.residual_gap or "(not stated)",
                    gpu_available=contract.gpu_available,
                    allow_large_benchmark=contract.allow_large_benchmark,
                )},
            ], PhenomenonPlanSchema)
        except Exception as exc:
            logger.warning("Task %s: phenomenon plan for gap %s failed (%s); non-fatal",
                           task_id[:8], gap.id[:8], exc)
            continue
        row = GapPhenomenonPlan(
            task_id=task_id,
            contract_id=contract.id,
            gap_id=gap.id,
            phenomenon=plan.phenomenon,
            critical_unknown=plan.critical_unknown,
            oracle_experiment=plan.oracle_experiment,
            kill_criterion=plan.kill_criterion,
            measurement=plan.measurement,
            pipeline_version=state.pipeline_version,
        )
        db.add(row)
        db.flush()
        created_ids.append(row.id)

    paper_repo.save_trace(db, task_id, "generate_phenomenon_plans", "decision",
                          output_data={
                              "surviving_gap_count": len(gaps),
                              "plan_count": len(created_ids),
                          })
    db.commit()
    return created_ids

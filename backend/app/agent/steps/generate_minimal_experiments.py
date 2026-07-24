"""Step: Turn gate-approved interventions into minimal decisive experiments."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agent.state import ResearchState
from app.db.models import GapCandidate, InterventionCandidate
from app.db.repositories import paper_repo

logger = logging.getLogger(__name__)

_MIN_EXPERIMENT_SYSTEM = """You design a minimal decisive experiment for a research intervention.
The experiment must test the stated mechanism before proposing a full paper evaluation.
Use only supplied paper IDs as related work. Specify a falsifiable hypothesis, controls, metrics,
success condition, and failure condition. Do not invent dataset or baseline names; use generic
roles when the evidence does not identify a specific valid name."""

_MIN_EXPERIMENT_USER = """Gap:
- Observed problem: {observed_problem}
- Remaining delta: {remaining_delta}
- Hypothesis: {gap_hypothesis}

Intervention:
- Failure mechanism: {failure_mechanism}
- Proposed intervention: {proposed_intervention}
- Intermediate effect: {intermediate_effect}
- Measurable outcome: {measurable_outcome}

Related paper IDs: {paper_ids}
Resource constraints: GPU={gpu_available}; max GPU hours={max_gpu_hours}; runtime minutes={max_runtime_minutes}
"""


class MinimalExperimentSchema(BaseModel):
    title: str = Field(min_length=5)
    summary: str = Field(min_length=10)
    hypothesis: str = Field(min_length=10)
    dataset: str
    baselines: str
    metrics: str
    controls: list[str] = Field(default_factory=list)
    steps: list[str] = Field(min_length=2)
    success_condition: str = Field(min_length=5)
    falsification_condition: str = Field(min_length=5)
    risks: str = ""


@dataclass
class MinimalExperimentResult:
    idea_ids: list[str]
    experiment_ids: list[str]

    def to_phase_payload(self) -> dict:
        return {"idea_ids": self.idea_ids, "experiment_ids": self.experiment_ids}


async def generate_minimal_experiments(db, state: ResearchState, llm, task_id: str) -> MinimalExperimentResult:
    """Persist GO ideas and their small falsifiable experiments from passed interventions."""
    from app.db.models import ResearchContract

    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    if not contract:
        return MinimalExperimentResult([], [])

    interventions = db.query(InterventionCandidate).filter(
        InterventionCandidate.task_id == task_id,
        InterventionCandidate.status == "passed",
    ).all()
    idea_ids = []
    experiment_ids = []
    for intervention in interventions:
        gap = db.get(GapCandidate, intervention.gap_id)
        if not gap:
            continue
        audit = gap_repo.list_gap_audits(db, gap.id)[-1] if gap_repo.list_gap_audits(db, gap.id) else None
        paper_ids = json.loads(audit.neighbor_paper_ids_json or "[]") if audit else []
        plan = await llm.chat_json([
            {"role": "system", "content": _MIN_EXPERIMENT_SYSTEM},
            {"role": "user", "content": _MIN_EXPERIMENT_USER.format(
                observed_problem=gap.observed_problem or "(not specified)",
                remaining_delta=(audit.remaining_delta if audit else gap.claimed_delta) or "(not specified)",
                gap_hypothesis=gap.testable_hypothesis or "(not specified)",
                failure_mechanism=intervention.failure_mechanism,
                proposed_intervention=intervention.proposed_intervention,
                intermediate_effect=intervention.intermediate_effect,
                measurable_outcome=intervention.measurable_outcome,
                paper_ids=paper_ids,
                gpu_available=contract.gpu_available,
                max_gpu_hours=contract.max_gpu_hours,
                max_runtime_minutes=contract.max_runtime_minutes,
            )},
        ], MinimalExperimentSchema)
        idea = paper_repo.save_idea(db, task_id, {
            "title": plan.title,
            "description": plan.summary,
            "motivation": gap.observed_problem,
            "method_sketch": intervention.proposed_intervention,
            "expected_contribution": intervention.measurable_outcome,
            "related_paper_ids_json": json.dumps(paper_ids, ensure_ascii=False),
        })
        paper_repo.update_idea_scores(db, idea.id, {
            "novelty": 1.0,
            "feasibility": 1.0,
            "significance": 1.0,
            "evidence_support": 1.0,
            "differentiation": 1.0,
            "experimentability": 1.0,
            "potential_impact": 1.0,
            "risk": 0.0,
        }, 1.0, "go")
        steps = [
            *plan.steps,
            f"Success condition: {plan.success_condition}",
            f"Falsification condition: {plan.falsification_condition}",
            *[f"Control: {control}" for control in plan.controls],
        ]
        experiment = paper_repo.save_experiment(db, task_id, idea.id, {
            "hypothesis": plan.hypothesis,
            "dataset": plan.dataset,
            "baselines": plan.baselines,
            "metrics": plan.metrics,
            "steps_markdown": "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps)),
            "steps_json": json.dumps({
                "steps": plan.steps,
                "controls": plan.controls,
                "success_condition": plan.success_condition,
                "falsification_condition": plan.falsification_condition,
            }, ensure_ascii=False),
            "risks": plan.risks,
        })
        idea_ids.append(idea.id)
        experiment_ids.append(experiment.id)

    paper_repo.save_trace(db, task_id, "generate_minimal_experiments", "action", output_data={
        "idea_count": len(idea_ids),
        "experiment_count": len(experiment_ids),
    })
    db.commit()
    return MinimalExperimentResult(idea_ids, experiment_ids)


from app.db.repositories import gap_repo

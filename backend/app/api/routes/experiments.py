"""Experiment API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.models import ExperimentPlan, isoformat_utc
from app.schemas.schemas import ExperimentOut

router = APIRouter()


@router.get("/tasks/{task_id}/experiments", response_model=list[ExperimentOut])
def list_experiments(task_id: str, db: Session = Depends(get_db_session)):
    plans = db.query(ExperimentPlan).filter(
        ExperimentPlan.task_id == task_id
    ).order_by(ExperimentPlan.created_at.desc()).all()
    return [_to_out(p) for p in plans]


@router.get("/tasks/{task_id}/experiments/{plan_id}/export")
def export_experiment(task_id: str, plan_id: str, format: str = "markdown",
                       db: Session = Depends(get_db_session)):
    plan = db.get(ExperimentPlan, plan_id)
    if not plan or plan.task_id != task_id:
        raise HTTPException(404, "Experiment plan not found")

    if format == "json":
        import json
        return {
            "id": plan.id,
            "hypothesis": plan.hypothesis,
            "dataset": plan.dataset,
            "baselines": plan.baselines,
            "metrics": plan.metrics,
            "model_spec": plan.model_spec,
            "dataset_provenance": plan.dataset_provenance,
            "oracle": plan.oracle,
            "statistical_analysis": plan.statistical_analysis,
            "resource_budget": plan.resource_budget,
            "scenario_atoms": plan.scenario_atoms_json,
            "steps": plan.steps_json,
            "risks": plan.risks,
        }
    else:
        # Markdown export
        md = f"""# Experiment Plan

## Hypothesis
{plan.hypothesis or ''}

## Dataset
{plan.dataset or ''}

## Baselines
{plan.baselines or ''}

## Metrics
{plan.metrics or ''}

## Model
{plan.model_spec or ''}

## Dataset provenance
{plan.dataset_provenance or ''}

## Oracle
{plan.oracle or ''}

## Statistical analysis
{plan.statistical_analysis or ''}

## Resource budget
{plan.resource_budget or ''}

## Scenario atoms
{plan.scenario_atoms_json or '[]'}

## Steps
{plan.steps_markdown or ''}

## Risks
{plan.risks or ''}
"""
        return {"format": "markdown", "content": md}


def _to_out(plan: ExperimentPlan) -> ExperimentOut:
    return ExperimentOut(
        id=plan.id,
        task_id=plan.task_id,
        idea_id=plan.idea_id,
        hypothesis=plan.hypothesis,
        dataset=plan.dataset,
        baselines=plan.baselines,
        metrics=plan.metrics,
        model_spec=plan.model_spec,
        dataset_provenance=plan.dataset_provenance,
        oracle=plan.oracle,
        statistical_analysis=plan.statistical_analysis,
        resource_budget=plan.resource_budget,
        scenario_atoms_json=plan.scenario_atoms_json,
        steps_markdown=plan.steps_markdown,
        steps_json=plan.steps_json,
        risks=plan.risks,
        created_at=isoformat_utc(plan.created_at),
    )

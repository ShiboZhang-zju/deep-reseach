"""Step: Generate intervention candidates and apply MVP hard gates."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agent.state import ResearchState
from app.db.models import GapCandidate, ResearchContract
from app.db.repositories import gap_repo, paper_repo
from app.db.repositories import intervention_repo

logger = logging.getLogger(__name__)

_INTERVENTION_SYSTEM = """You design bounded research interventions for an audited research gap.
Generate 1-3 interventions, not full paper ideas. Each intervention must explicitly connect:
Observed problem -> failure mechanism -> intervention -> intermediate effect -> measurable outcome.
Do not invent papers, datasets, or evidence IDs. Use only dependency paper IDs supplied in the context."""

_INTERVENTION_USER = """Surviving gap:
- ID: {gap_id}
- Setting: {target_setting}
- Observed problem: {observed_problem}
- Missing capability: {missing_capability}
- Remaining delta: {remaining_delta}
- Testable hypothesis: {hypothesis}

Evidence IDs: {evidence_ids}
Neighbor paper IDs: {neighbor_ids}
Resource constraints:
- GPU available: {gpu_available}
- Max GPU hours: {max_gpu_hours}
- Max API budget: {max_api_budget}
- Max runtime minutes: {max_runtime_minutes}
- Allow model training: {allow_model_training}
- Allow large benchmark: {allow_large_benchmark}

Return only interventions whose mechanism directly addresses the observed failure."""


class InterventionSchema(BaseModel):
    intervention_type: str
    failure_mechanism: str = Field(min_length=5)
    proposed_intervention: str = Field(min_length=5)
    intermediate_effect: str = Field(min_length=5)
    measurable_outcome: str = Field(min_length=5)
    required_components: list[str] = Field(default_factory=list)
    dependency_paper_ids: list[str] = Field(default_factory=list)
    implementation_cost: str = ""
    mechanism_confidence: float = Field(ge=0, le=1)


class InterventionList(BaseModel):
    interventions: list[InterventionSchema] = Field(default_factory=list, max_length=3)


@dataclass
class InterventionGenerationResult:
    intervention_ids: list[str]
    passed_intervention_ids: list[str]

    def to_phase_payload(self) -> dict:
        return {
            "intervention_ids": self.intervention_ids,
            "passed_intervention_ids": self.passed_intervention_ids,
        }


async def generate_interventions(
    db,
    state: ResearchState,
    llm,
    task_id: str,
) -> InterventionGenerationResult:
    """Generate interventions only for surviving gaps and enforce three hard gates."""
    contract = db.get(ResearchContract, state.contract_id) if state.contract_id else None
    if not contract:
        return InterventionGenerationResult([], [])

    created_ids = []
    passed_ids = []
    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.status == "surviving",
    ).all()
    for gap in gaps:
        evidence_ids = [link.evidence_id for link in gap_repo.list_gap_evidence(db, gap.id)]
        audits = gap_repo.list_gap_audits(db, gap.id)
        latest_audit = audits[-1] if audits else None
        neighbor_ids = json.loads(latest_audit.neighbor_paper_ids_json or "[]") if latest_audit else []
        result = await llm.chat_json([
            {"role": "system", "content": _INTERVENTION_SYSTEM},
            {"role": "user", "content": _INTERVENTION_USER.format(
                gap_id=gap.id,
                target_setting=gap.target_setting or "(not specified)",
                observed_problem=gap.observed_problem or "(not specified)",
                missing_capability=gap.missing_capability or "(not specified)",
                remaining_delta=(latest_audit.remaining_delta if latest_audit else gap.claimed_delta) or "(not specified)",
                hypothesis=gap.testable_hypothesis or "(not specified)",
                evidence_ids=evidence_ids,
                neighbor_ids=neighbor_ids,
                gpu_available=contract.gpu_available,
                max_gpu_hours=contract.max_gpu_hours,
                max_api_budget=contract.max_api_budget,
                max_runtime_minutes=contract.max_runtime_minutes,
                allow_model_training=contract.allow_model_training,
                allow_large_benchmark=contract.allow_large_benchmark,
            )},
        ], InterventionList)
        for candidate in result.interventions:
            if not set(candidate.dependency_paper_ids).issubset(set(neighbor_ids)):
                logger.warning("Task %s: skip intervention with unknown dependency paper", task_id[:8])
                continue
            gates = _evaluate_hard_gates(gap, latest_audit, evidence_ids, candidate, contract)
            item = intervention_repo.create_intervention_candidate(db, task_id, gap.id, {
                **candidate.model_dump(),
                **gates,
                "status": "passed" if all(value == "PASS" for value in gates["gate_statuses"].values()) else "rejected",
                "evidence_gate": gates["gate_statuses"]["evidence"],
                "novelty_gate": gates["gate_statuses"]["novelty"],
                "feasibility_gate": gates["gate_statuses"]["feasibility"],
                "gate_rationale": gates["gate_rationale"],
            })
            created_ids.append(item.id)
            if item.status == "passed":
                passed_ids.append(item.id)

    paper_repo.save_trace(db, task_id, "generate_interventions", "decision", output_data={
        "surviving_gap_count": len(gaps),
        "intervention_count": len(created_ids),
        "passed_gate_count": len(passed_ids),
    })
    db.commit()
    return InterventionGenerationResult(created_ids, passed_ids)


def _evaluate_hard_gates(gap, audit, evidence_ids, candidate, contract) -> dict:
    evidence_status = "PASS" if len(evidence_ids) >= 2 else "FAIL"
    evidence_reason = "至少两条可追溯证据支撑 Gap" if evidence_status == "PASS" else "Gap 缺少两条独立可追溯证据"

    novelty_status = "UNKNOWN"
    novelty_reason = "缺少完成的近邻审计"
    if audit:
        if audit.audit_result == "confirmed" and audit.remaining_delta:
            novelty_status = "PASS"
            novelty_reason = "近邻审计确认存在 remaining delta"
        elif audit.audit_result == "closed":
            novelty_status = "FAIL"
            novelty_reason = "近邻审计显示核心 claim 已被覆盖"
        elif audit.audit_result == "partially_closed" and audit.remaining_delta:
            novelty_status = "UNKNOWN"
            novelty_reason = "Gap 需先收窄后再判断新颖性"

    feasibility_status = "PASS"
    feasibility_reason = "符合当前 Contract 约束"
    text = " ".join([candidate.proposed_intervention, candidate.implementation_cost]).lower()
    training_terms = (" fine-tune", " finetune", " training", "训练", "微调")
    if contract.allow_model_training is False and any(token in text for token in training_terms):
        feasibility_status = "FAIL"
        feasibility_reason = "Contract 不允许模型训练"
    elif contract.allow_large_benchmark is False and any(token in text for token in ("benchmark", "large dataset", "基准", "大规模数据集")):
        feasibility_status = "FAIL"
        feasibility_reason = "Contract 不允许构建大型 Benchmark"
    elif contract.gpu_available is False and any(token in text for token in ("gpu", "train", "fine-tune", "训练", "微调")):
        feasibility_status = "FAIL"
        feasibility_reason = "Contract 未提供 GPU 资源"

    gate_statuses = {
        "evidence": evidence_status,
        "novelty": novelty_status,
        "feasibility": feasibility_status,
    }
    return {
        "gate_statuses": gate_statuses,
        "gate_rationale": {
            "evidence": evidence_reason,
            "novelty": novelty_reason,
            "feasibility": feasibility_reason,
        },
    }

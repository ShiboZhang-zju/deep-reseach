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

# v1 (2026-08-28, task d6f64087): the novelty gate now consumes the audit's
# novelty_confidence instead of only the verdict string. Previously
# confirmed + novelty_confidence=0.3 sailed through as PASS (the number was
# even printed in the rationale but never used in the decision), so all three
# interventions of the run reached tier A on an audit that itself reported
# only 30% confidence in the gap's novelty. Changing this rule must
# invalidate previously stamped intervention phases on resumed tasks — this
# constant is hashed into the phase input_version for exactly that purpose.
# v2 (2026-08-28, ARIS-inspired breadth stage): lens fan-out — one LLM call
# per (gap, lens) instead of one call per gap. A single call's 1-3 candidates
# are correlated: they all think along the first mechanism that came to mind
# (task d6f64087: 2 of 3 interventions tested the same hypothesis and merged
# in the cluster gate). Each lens attacks the gap from one named perspective;
# the downstream hard gates + hypothesis clustering remain the arbiter
# (breadth, not verdict).
INTERVENTION_GENERATION_POLICY_VERSION = "intervention-lens-fanout-v2"
# Novelty gate thresholds on audit.novelty_confidence (only when the audit
# verdict itself is "confirmed"): below the WARN bound the confirmed verdict
# still flows downstream but only as tier B (needs confirmation); below the
# FAIL bound the gap is treated as not credibly novel. None (legacy audits
# without the field) keeps PASS so historical tasks are not retro-failed.
_NOVELTY_CONFIDENCE_WARN_BELOW = 0.5
_NOVELTY_CONFIDENCE_FAIL_BELOW = 0.3

# Max candidates accepted from a single lens — the fan-out's total breadth
# stays bounded (6 lenses x 2 = 12 per gap before gates).
_MAX_CANDIDATES_PER_LENS = 2

_INTERVENTION_LENSES: tuple[tuple[str, str], ...] = (
    ("mechanism_correction",
     "Directly counteract the failure mechanism the gap documents. The "
     "intervention changes the component where the mechanism operates so the "
     "failure can no longer produce the observed problem."),
    ("assumption_inversion",
     "Invert a default assumption shared by current approaches. Identify one "
     "assumption everyone inherits (e.g. 'rank what matters'), flip it (e.g. "
     "'detect what is stale instead'), and design the intervention around the "
     "inverted assumption."),
    ("diagnostic_first",
     "Build a measurement instrument, not a method: an observable score, "
     "probe, or test that quantifies the failure mechanism itself. Do NOT "
     "restate the phenomenon plan's oracle experiment — extend it into a "
     "reusable diagnostic that later work can adopt as a standard measure."),
    ("boundary_scaling",
     "Characterize WHEN the failure starts and stops: which regime boundary "
     "(scale, frequency, density, budget) governs it. The intervention maps "
     "the boundary, not just improves the average."),
    ("cross_domain_transfer",
     "Import a mechanism proven in a neighbouring field (databases, systems, "
     "signal processing, control theory, statistics, ...) that solves an "
     "isomorphic problem. Name the source field and the mapping explicitly."),
    ("contradiction_resolver",
     "Resolve a documented tension: two results or claims that both hold but "
     "conflict, or one quantity improving while another degrades. The "
     "intervention tests or reconciles the contradiction."),
)

_INTERVENTION_SYSTEM = """You design bounded research interventions for an audited research gap.
Each user message names ONE research lens; generate only interventions for that lens
(respecting the lens's candidate cap), not full paper ideas. Each intervention must
explicitly connect:
Observed problem -> failure mechanism -> intervention -> intermediate effect -> measurable outcome.
Do not invent papers, datasets, or evidence IDs.
dependency_paper_ids may only contain IDs taken verbatim from the "Neighbor paper IDs"
list in the context; the "Evidence IDs" are evidence units, not papers, and must never
appear there. Leave dependency_paper_ids empty when no neighbor paper is required.

CRITICAL — diversify the failure mechanisms: when you produce more than one
intervention, each one MUST target a DISTINCT failure mechanism. Do not emit
multiple variants of the same mechanism (e.g. three flavours of "early stopping").
If the lens only supports one genuinely distinct mechanism, generate one
intervention rather than padding — and if it supports none, return an empty list."""

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

_LENS_USER_SECTION = """

Research lens — every candidate MUST attack the gap from this one angle:
- Lens: {lens_name}
- Directive: {lens_directive}

Generate 1-2 interventions for THIS lens only. If this lens genuinely does
not fit the gap (no sound, evidence-respecting intervention exists from this
angle), return an empty interventions list rather than padding."""


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
    dropped_dependencies = []
    lens_fanout = []
    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == contract.id,
        GapCandidate.status == "surviving",
    ).all()
    for gap in gaps:
        evidence_ids = [link.evidence_id for link in gap_repo.list_gap_evidence(db, gap.id)]
        audits = gap_repo.list_gap_audits(db, gap.id)
        latest_audit = audits[-1] if audits else None
        neighbor_ids = json.loads(latest_audit.neighbor_paper_ids_json or "[]") if latest_audit else []
        base_user = _INTERVENTION_USER.format(
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
        )
        for lens_name, lens_directive in _INTERVENTION_LENSES:
            result = await llm.chat_json([
                {"role": "system", "content": _INTERVENTION_SYSTEM},
                {"role": "user", "content": base_user + _LENS_USER_SECTION.format(
                    lens_name=lens_name, lens_directive=lens_directive)},
            ], InterventionList)
            # Bounded breadth: a lens that ignores the 1-2 instruction still
            # cannot flood the candidate pool.
            candidates = result.interventions[:_MAX_CANDIDATES_PER_LENS]
            lens_fanout.append({
                "gap_id": gap.id, "lens": lens_name,
                "candidate_count": len(candidates),
            })
            for candidate in candidates:
                # dependency_paper_ids is supporting context, not the basis of the
                # intervention's validity (the three hard gates below decide that).
                # Discarding the whole intervention over one unoffered ID threw away
                # otherwise sound proposals — observed in production, where every
                # intervention for a surviving gap was dropped this way and the run
                # ended with no idea at all. Drop the unoffered IDs and record them.
                unoffered = [item for item in candidate.dependency_paper_ids
                             if item not in set(neighbor_ids)]
                if unoffered:
                    candidate.dependency_paper_ids = [
                        item for item in candidate.dependency_paper_ids
                        if item in set(neighbor_ids)]
                    dropped_dependencies.append({"gap_id": gap.id, "dropped_paper_ids": unoffered})
                    logger.warning(
                        "Task %s: dropped %d unoffered dependency paper ID(s) from an "
                        "intervention for gap %s: %s",
                        task_id[:8], len(unoffered), gap.id[:8], unoffered)
                gates = _evaluate_hard_gates(gap, latest_audit, evidence_ids, candidate, contract)
                tier = _compute_confidence_tier(gap, gates["gate_statuses"])
                item = intervention_repo.create_intervention_candidate(db, task_id, gap.id, {
                    **candidate.model_dump(),
                    "contract_id": contract.id,
                    **gates,
                    # O1: graded output. A/B tiers flow downstream (idea synthesis);
                    # C is kept as a speculative direction but does not pass the gate.
                    "status": "passed" if tier in ("A", "B") else "rejected",
                    "confidence_tier": tier,
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
        "dropped_dependency_paper_ids": dropped_dependencies,
        "lens_fanout": lens_fanout,
    })
    db.commit()
    return InterventionGenerationResult(created_ids, passed_ids)


def _compute_confidence_tier(gap, gate_statuses: dict) -> str:
    """O1: derive an A/B/C confidence tier from gate results + gap provenance.

    A: every hard gate PASS and the gap is backed by full-text evidence.
    B: no gate FAILed but at least one is UNKNOWN/WARN, or the gap is only
       abstract-strength (provenance_status !='complete'). Still actionable,
       flagged as needing confirmation.
    C: at least one gate FAILed — speculative direction, kept for the user but
       not promoted downstream.
    """
    statuses = set(gate_statuses.values())
    if "FAIL" in statuses:
        return "C"
    full_text = getattr(gap, "provenance_status", "partial") == "complete"
    if statuses <= {"PASS"} and full_text:
        return "A"
    return "B"


def _evaluate_hard_gates(gap, audit, evidence_ids, candidate, contract) -> dict:
    evidence_status = "PASS" if len(evidence_ids) >= 2 else "FAIL"
    # E2E 2026-08-26: rationale must be instantiated (counts/verdicts), not a
    # fixed template — a static string proved identical for every intervention
    # and carried no reviewable basis.
    evidence_reason = (
        f"{len(evidence_ids)} 条可追溯证据支撑 Gap（门槛 ≥2）"
        if evidence_status == "PASS"
        else f"Gap 仅 {len(evidence_ids)} 条可追溯证据（门槛 ≥2）")

    neighbor_ids_json = getattr(audit, "neighbor_paper_ids_json", None) if audit else None
    neighbor_count = len(json.loads(neighbor_ids_json or "[]"))
    novelty_confidence = getattr(audit, "novelty_confidence", None) if audit else None
    novelty_status = "UNKNOWN"
    novelty_reason = "缺少完成的近邻审计"
    if audit:
        if audit.audit_result == "confirmed" and audit.remaining_delta:
            # P0-1a (task d6f64087): a "confirmed" verdict only means no
            # neighbor covered the claim — absence of a hit, not presence of
            # novelty. The auditor's own novelty_confidence quantifies how
            # much it trusts its search; the gate must consume it instead of
            # printing it. confirmed+0.3 previously produced tier-A
            # interventions and would have produced an executable_candidate.
            if novelty_confidence is not None and novelty_confidence < _NOVELTY_CONFIDENCE_FAIL_BELOW:
                novelty_status = "FAIL"
                novelty_reason = (
                    f"近邻审计判 confirmed 但 novelty_confidence={novelty_confidence:.2f}"
                    f"（<{_NOVELTY_CONFIDENCE_FAIL_BELOW}）：审计自身不确信新颖性主张，"
                    f"视为不可信新颖性")
            elif novelty_confidence is not None and novelty_confidence < _NOVELTY_CONFIDENCE_WARN_BELOW:
                novelty_status = "WARN"
                novelty_reason = (
                    f"近邻审计判 confirmed（{neighbor_count} 篇近邻），但 "
                    f"novelty_confidence={novelty_confidence:.2f}"
                    f"（<{_NOVELTY_CONFIDENCE_WARN_BELOW}）偏低，新颖性主张需复审确认")
            else:
                novelty_status = "PASS"
                novelty_reason = (
                    f"近邻审计 confirmed（{neighbor_count} 篇近邻，"
                    f"novelty_confidence={novelty_confidence}，存在 remaining delta）")
        elif audit.audit_result == "closed":
            novelty_status = "FAIL"
            novelty_reason = f"近邻审计显示核心 claim 已被覆盖（{neighbor_count} 篇近邻）"
        elif audit.audit_result == "partially_closed" and audit.remaining_delta:
            novelty_status = "UNKNOWN"
            novelty_reason = "Gap 需先收窄后再判断新颖性"

    # O1: relax the feasibility gate's keyword over-kill. A bare substring
    # match (e.g. the LLM mentions "train a small classifier" as an auxiliary
    # step) should NOT hard-FAIL the whole intervention. We distinguish:
    #   - core intervention text mentions training  -> WARN (needs confirmation,
    #     lands in B tier, still flows downstream)
    #   - implementation_cost/plan explicitly centers on training AND the
    #     contract forbids it AND no GPU -> FAIL (genuinely infeasible)
    feasibility_status = "PASS"
    feasibility_reason = "符合当前 Contract 约束"
    core_text = candidate.proposed_intervention.lower()
    cost_text = (candidate.implementation_cost or "").lower()
    text = f"{core_text} {cost_text}"
    training_terms = ("fine-tune", "finetune", "fine tune", "training", "train ", "训练", "微调")
    # A bare "benchmark" used to hard-FAIL interventions like a "benchmarking
    # protocol" / "micro-benchmark" that are ordinary single-GPU experiments,
    # not the large-scale benchmark construction the contract forbids. Split
    # the two notions apart so only an explicitly large-scale effort fails.
    large_benchmark_terms = ("large benchmark", "large-scale", "large scale",
                             "large dataset", "大型", "大规模", "超大")
    small_benchmark_terms = ("micro-benchmark", "micro benchmark", "microbenchmark",
                             "small benchmark", "small-scale", "small scale",
                             "benchmarking protocol", "controlled benchmark",
                             "微基准", "小规模", "受控", "单机", "single-gpu")
    mentions_benchmark = ("benchmark" in text or "基准" in text)

    mentions_training = any(token in text for token in training_terms)
    mentions_large_benchmark = any(token in text for token in large_benchmark_terms)
    mentions_small_benchmark = any(token in text for token in small_benchmark_terms)

    if contract.allow_model_training is False and mentions_training:
        # Only hard-FAIL when training is clearly central (appears in the core
        # proposed intervention) AND no GPU is available; otherwise WARN.
        core_training = any(token in core_text for token in training_terms)
        if core_training and contract.gpu_available is False:
            feasibility_status = "FAIL"
            feasibility_reason = "Contract 不允许模型训练且无 GPU，方案核心依赖训练"
        else:
            feasibility_status = "WARN"
            feasibility_reason = "方案提及训练/微调但可能仅为辅助步骤，需人工确认是否可在约束内替换"
    elif contract.allow_large_benchmark is False and mentions_benchmark:
        if mentions_large_benchmark and not mentions_small_benchmark:
            feasibility_status = "FAIL"
            feasibility_reason = "Contract 不允许构建大型 Benchmark"
        elif mentions_small_benchmark:
            feasibility_status = "PASS"
            feasibility_reason = "方案为受控小规模基准实验（micro-benchmark），符合 Contract 约束"
        else:
            feasibility_status = "WARN"
            feasibility_reason = "方案提及基准/数据集，需确认规模是否在约束内"
    elif contract.gpu_available is False and any(token in core_text for token in ("gpu", "train", "fine-tune", "训练", "微调")):
        feasibility_status = "WARN"
        feasibility_reason = "无 GPU 资源，需确认方案是否有免训练替代路径"

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

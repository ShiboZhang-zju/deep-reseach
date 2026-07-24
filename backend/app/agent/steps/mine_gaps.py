"""Step: Mine lightweight, evidence-backed research gap candidates."""

import json
import logging

from app.agent.state import ResearchState
from app.db.models import EvidenceUnit, QuestionEvidenceLink, ResearchContract, ResearchQuestion
from app.db.repositories import gap_repo, paper_repo
from app.schemas.schemas import GapCandidateList

logger = logging.getLogger(__name__)

_SUPPORTED_GAP_TYPES = {"boundary_gap", "missing_evaluation"}

_GAP_MINING_SYSTEM = """You identify candidate research gaps from supplied paper evidence.
Return only gaps that are supported by the supplied evidence. A gap is not merely a topic with little evidence.

Allowed gap types:
- boundary_gap: an existing method or evaluation has a documented boundary, failure condition, or untested setting.
- missing_evaluation: existing work does not evaluate a concrete, important capability, condition, cost, or risk.

For every gap:
- cite only supplied question IDs and evidence IDs;
- state what existing work already covers and the smallest missing capability;
- write a falsifiable condition that would close the gap;
- do not propose a solution or invent paper findings.
Return an empty list when the evidence is insufficient."""

_GAP_MINING_USER = """Research topic: {topic}
Target problem: {target_problem}
Target setting: {target_setting}

Research questions:
{questions}

Evidence (only these IDs may be cited):
{evidence}

Generate at most {max_gaps} evidence-backed candidate gaps."""


def _format_evidence(evidence: list[EvidenceUnit]) -> str:
    lines = []
    for item in evidence:
        conditions = json.loads(item.conditions_json or "{}")
        condition_text = json.dumps(conditions, ensure_ascii=False) if conditions else "(none)"
        lines.append(
            f"- Evidence ID: {item.id}\n"
            f"  Type: {item.evidence_type}\n"
            f"  Claim: {item.normalized_claim}\n"
            f"  Section: {item.section or '(unknown)'}\n"
            f"  Conditions: {condition_text}\n"
            f"  Verification: {item.verification_status or 'unverified'}"
        )
    return "\n".join(lines)


async def mine_gap_candidates(
    db,
    state: ResearchState,
    llm,
    task_id: str,
    max_gaps: int = 3,
) -> list:
    """Create evidence-backed boundary/evaluation gap candidates.

    This MVP step deliberately mines only gap types that can be grounded in
    limitations, negative results, evaluation, and comparison evidence. It does
    not infer that missing evidence alone is a research gap.
    """
    if not state.contract_id:
        logger.warning("Task %s: skip gap mining without an active contract", task_id[:8])
        return []

    contract = db.get(ResearchContract, state.contract_id)
    if not contract or contract.status != "active":
        logger.warning("Task %s: skip gap mining without an active contract record", task_id[:8])
        return []

    existing = gap_repo.list_gaps_for_contract(db, task_id, contract.id)
    if existing:
        state.active_gap_ids = [gap.id for gap in existing if gap.status != "rejected"]
        return existing

    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract.id,
        ResearchQuestion.status.in_(["open", "partially_covered", "covered"]),
    ).order_by(ResearchQuestion.importance.desc()).all()
    if not questions:
        return []

    question_ids = {question.id for question in questions}
    linked_evidence_ids = {
        evidence_id
        for (evidence_id,) in db.query(QuestionEvidenceLink.evidence_id).filter(
            QuestionEvidenceLink.question_id.in_(question_ids),
        ).distinct().all()
    }
    evidence_query = db.query(EvidenceUnit).filter(EvidenceUnit.task_id == task_id)
    if linked_evidence_ids:
        evidence_query = evidence_query.filter(EvidenceUnit.id.in_(linked_evidence_ids))
    evidence = evidence_query.filter(
        EvidenceUnit.evidence_type.in_(["limitation", "negative_result", "comparison", "result", "metric"]),
        ~EvidenceUnit.verification_status.in_(["rejected", "conflicted"]),
    ).order_by(EvidenceUnit.extraction_confidence.desc()).limit(20).all()
    if not evidence:
        logger.info("Task %s: no usable evidence for gap mining", task_id[:8])
        return []

    question_text = "\n".join(
        f"- Question ID: {question.id}\n  Type: {question.question_type}\n  Question: {question.question}"
        for question in questions
    )
    result = await llm.chat_json([
        {"role": "system", "content": _GAP_MINING_SYSTEM},
        {"role": "user", "content": _GAP_MINING_USER.format(
            topic=contract.topic,
            target_problem=contract.target_problem or "(not specified)",
            target_setting=contract.target_setting or "(not specified)",
            questions=question_text,
            evidence=_format_evidence(evidence),
            max_gaps=max_gaps,
        )},
    ], GapCandidateList)

    valid_evidence_ids = {item.id for item in evidence}
    created = []
    for candidate in result.gaps[:max_gaps]:
        if candidate.gap_type not in _SUPPORTED_GAP_TYPES:
            continue
        if not set(candidate.question_ids).issubset(question_ids):
            logger.warning("Task %s: skipped gap with invalid question IDs", task_id[:8])
            continue
        if not set(candidate.supporting_evidence_ids).issubset(valid_evidence_ids):
            logger.warning("Task %s: skipped gap with invalid evidence IDs", task_id[:8])
            continue
        if not set(candidate.contradicting_evidence_ids).issubset(valid_evidence_ids):
            logger.warning("Task %s: skipped gap with invalid contradicting evidence IDs", task_id[:8])
            continue

        gap = gap_repo.create_gap_candidate(
            db,
            task_id=task_id,
            contract_id=contract.id,
            gap_type=candidate.gap_type,
            description=candidate.description,
            target_setting=candidate.target_setting,
            observed_problem=candidate.observed_problem,
            existing_coverage=candidate.existing_coverage,
            missing_capability=candidate.missing_capability,
            claimed_delta=candidate.claimed_delta,
            testable_hypothesis=candidate.testable_hypothesis,
            falsification_condition=candidate.falsification_condition,
            provenance_status="complete",
            question_ids=candidate.question_ids,
            mining_round=state.current_round,
            novelty_score=candidate.novelty_score,
            feasibility_score=candidate.feasibility_score,
            significance_score=candidate.significance_score,
        )
        for evidence_id in candidate.supporting_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "suggests", 0.8)
        for evidence_id in candidate.contradicting_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "contradicts", 0.8)
        created.append(gap)

    state.active_gap_ids = [gap.id for gap in created]
    paper_repo.save_trace(db, task_id, "mine_gap_candidates", "action", output_data={
        "contract_id": contract.id,
        "candidate_count": len(created),
        "evidence_count": len(evidence),
        "supported_gap_types": sorted(_SUPPORTED_GAP_TYPES),
    })
    db.commit()
    return created

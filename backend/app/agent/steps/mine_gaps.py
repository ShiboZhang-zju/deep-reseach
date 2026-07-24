"""Step: Mine lightweight, evidence-backed research gap candidates."""

import json
import logging
from dataclasses import dataclass, field

from app.agent.state import ResearchState
from app.db.models import EvidenceUnit, QuestionEvidenceLink, ResearchContract, ResearchQuestion
from app.db.repositories import gap_repo, paper_repo
from app.schemas.schemas import GapCandidateList

logger = logging.getLogger(__name__)

_SUPPORTED_GAP_TYPES = {"boundary_gap", "missing_evaluation"}
_SUPPORTING_RELATIONS = {"supports", "partially_answers"}
_ADMISSIBLE_STATUSES = {"verified", "upgraded", "abstract_only"}
_LIMITATION_SIGNAL_TYPES = {"limitation", "negative_result"}
_MIN_RELATION_RELEVANCE = 0.5
GAP_MINING_POLICY_VERSION = "evidence-admission-v1"

@dataclass(frozen=True)
class QuestionEvidenceAdmission:
    question_id: str
    status: str
    reason_codes: list[str] = field(default_factory=list)
    admissible_evidence_ids: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    limiting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    supporting_paper_ids: list[str] = field(default_factory=list)
    verified_span_count: int = 0

def _is_fulltext_locatable(item: EvidenceUnit) -> bool:
    return (
        item.verification_status in {"verified", "upgraded"}
        and bool(item.original_span)
        and bool(item.source_chunk_hash)
        and (item.page_number is not None or item.page_start is not None)
        and item.span_start is not None
        and item.span_end is not None
    )

def evaluate_gap_mining_admission(question, links, evidence_by_id) -> QuestionEvidenceAdmission:
    if question.status not in {"open", "partially_covered"}:
        return QuestionEvidenceAdmission(question.id, "FAIL", ["QUESTION_NOT_ELIGIBLE"])
    if not links:
        return QuestionEvidenceAdmission(question.id, "UNKNOWN", ["NO_LINKED_EVIDENCE"])
    supporting, contradicting = [], []
    for link in links:
        item = evidence_by_id.get(link.evidence_id)
        if not item or (link.relevance_score or 0) < _MIN_RELATION_RELEVANCE:
            continue
        if link.relation_type in _SUPPORTING_RELATIONS and item.verification_status in _ADMISSIBLE_STATUSES:
            supporting.append(item)
        if link.relation_type == "contradicts" and item.verification_status in {"verified", "upgraded"}:
            contradicting.append(item)
    papers = sorted({item.paper_id for item in supporting})
    spans = sum(_is_fulltext_locatable(item) for item in supporting)
    limiting = [item for item in supporting if item.evidence_type in _LIMITATION_SIGNAL_TYPES]
    reasons = []
    if len(papers) < 2: reasons.append("INSUFFICIENT_INDEPENDENT_PAPERS")
    if spans < 1: reasons.append("NO_FULLTEXT_LOCATABLE_EVIDENCE")
    if not limiting: reasons.append("NO_LIMITATION_SIGNAL")
    if contradicting: reasons.append("UNRESOLVED_VERIFIED_CONTRADICTION")
    return QuestionEvidenceAdmission(question.id, "PASS" if not reasons else "UNKNOWN", reasons, [item.id for item in supporting], [item.id for item in supporting], [item.id for item in limiting], [item.id for item in contradicting], papers, spans)

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

    existing = [gap for gap in gap_repo.list_gaps_for_contract(db, task_id, contract.id)
                if gap.mining_policy_version == GAP_MINING_POLICY_VERSION]
    if existing:
        state.active_gap_ids = [gap.id for gap in existing if gap.status != "rejected"]
        return existing

    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract.id,
        ResearchQuestion.status.in_(["open", "partially_covered"]),
    ).order_by(ResearchQuestion.importance.desc()).all()
    if not questions:
        return []

    question_ids = [question.id for question in questions]
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_(question_ids),
    ).all()
    links_by_question = {question_id: [] for question_id in question_ids}
    for link in links:
        links_by_question.setdefault(link.question_id, []).append(link)
    linked_evidence_ids = {link.evidence_id for link in links}
    evidence_by_id = {
        item.id: item for item in db.query(EvidenceUnit).filter(
            EvidenceUnit.task_id == task_id,
            EvidenceUnit.id.in_(linked_evidence_ids),
        ).all()
    } if linked_evidence_ids else {}
    admissions = [
        evaluate_gap_mining_admission(question, links_by_question[question.id], evidence_by_id)
        for question in questions
    ]
    paper_repo.save_trace(db, task_id, "gap_mining_admission", "decision", output_data={
        "contract_id": contract.id,
        "passed_question_ids": [item.question_id for item in admissions if item.status == "PASS"],
        "question_results": [{
            "question_id": item.question_id, "status": item.status, "reason_codes": item.reason_codes,
            "supporting_paper_count": len(item.supporting_paper_ids),
            "verified_span_count": item.verified_span_count,
            "limitation_evidence_count": len(item.limiting_evidence_ids),
            "contradiction_count": len(item.contradicting_evidence_ids),
        } for item in admissions],
    })
    passed_admissions = {item.question_id: item for item in admissions if item.status == "PASS"}
    if not passed_admissions:
        db.commit()
        return []

    allowed_evidence_ids = {
        evidence_id for admission in passed_admissions.values()
        for evidence_id in admission.admissible_evidence_ids
    }
    evidence = [evidence_by_id[evidence_id] for evidence_id in allowed_evidence_ids]
    question_text = "\n".join(
        f"- Question ID: {question.id}\n  Type: {question.question_type}\n  Question: {question.question}"
        for question in questions if question.id in passed_admissions
    )
    result = await llm.chat_json([{
        "role": "system", "content": _GAP_MINING_SYSTEM,
    }, {
        "role": "user", "content": _GAP_MINING_USER.format(
            topic=contract.topic, target_problem=contract.target_problem or "(not specified)",
            target_setting=contract.target_setting or "(not specified)", questions=question_text,
            evidence=_format_evidence(evidence), max_gaps=max_gaps,
        ),
    }], GapCandidateList)

    created = []
    rejected_candidates = []
    for candidate in result.gaps[:max_gaps]:
        candidate_question_ids = set(candidate.question_ids)
        allowed_for_candidate = {
            evidence_id for question_id in candidate_question_ids
            for evidence_id in passed_admissions.get(question_id, QuestionEvidenceAdmission("", "FAIL")).admissible_evidence_ids
        }
        support = [evidence_by_id.get(evidence_id) for evidence_id in candidate.supporting_evidence_ids]
        reason = None
        if candidate.gap_type not in _SUPPORTED_GAP_TYPES: reason = "UNSUPPORTED_GAP_TYPE"
        elif not candidate_question_ids or not candidate_question_ids.issubset(passed_admissions): reason = "UNKNOWN_QUESTION_ID"
        elif not candidate.supporting_evidence_ids or not set(candidate.supporting_evidence_ids).issubset(allowed_for_candidate): reason = "UNKNOWN_EVIDENCE_ID"
        elif None in support or len({item.paper_id for item in support}) < 2: reason = "INSUFFICIENT_CANDIDATE_PAPER_SUPPORT"
        elif not any(_is_fulltext_locatable(item) for item in support): reason = "CANDIDATE_LACKS_FULLTEXT_EVIDENCE"
        elif not any(item.evidence_type in _LIMITATION_SIGNAL_TYPES for item in support): reason = "CANDIDATE_LACKS_LIMITATION_SIGNAL"
        if reason:
            rejected_candidates.append(reason)
            continue
        gap = gap_repo.create_gap_candidate(
            db, task_id=task_id, contract_id=contract.id, gap_type=candidate.gap_type,
            description=candidate.description, target_setting=candidate.target_setting,
            observed_problem=candidate.observed_problem, existing_coverage=candidate.existing_coverage,
            missing_capability=candidate.missing_capability, claimed_delta=candidate.claimed_delta,
            testable_hypothesis=candidate.testable_hypothesis,
            falsification_condition=candidate.falsification_condition, provenance_status="complete",
            question_ids=candidate.question_ids, mining_round=state.current_round,
            novelty_score=candidate.novelty_score, feasibility_score=candidate.feasibility_score,
            significance_score=candidate.significance_score,
            mining_policy_version=GAP_MINING_POLICY_VERSION,
        )
        for evidence_id in candidate.supporting_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "suggests", 0.8)
        for evidence_id in candidate.contradicting_evidence_ids:
            gap_repo.create_gap_evidence_link(db, gap.id, evidence_id, "contradicts", 0.8)
        created.append(gap)
    paper_repo.save_trace(db, task_id, "gap_mining_candidates", "decision", output_data={
        "llm_candidate_count": len(result.gaps), "accepted_candidate_count": len(created),
        "rejected_candidates": rejected_candidates,
    })
    state.active_gap_ids = [gap.id for gap in created]
    paper_repo.save_trace(db, task_id, "mine_gap_candidates", "action", output_data={
        "contract_id": contract.id,
        "candidate_count": len(created),
        "evidence_count": len(evidence),
        "supported_gap_types": sorted(_SUPPORTED_GAP_TYPES),
    })
    db.commit()
    return created

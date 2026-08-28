"""Read-only Gap API routes.

Phase 3A Closure: Only GET endpoints — no mining or audit write operations.
Evidence is returned from gap_evidence_links (authoritative source),
not from deprecated JSON snapshot fields.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.db.repositories import gap_repo
from app.schemas.schemas import (
    GapCandidateOut, GapAuditOut, NeighborComparisonOut, GapEvidenceLinkOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["gaps"])


def _gap_to_out(gap) -> GapCandidateOut:
    """Convert GapCandidate ORM to API response, reading evidence from links."""
    return GapCandidateOut(
        id=gap.id,
        task_id=gap.task_id,
        contract_id=gap.contract_id,
        gap_type=gap.gap_type,
        description=gap.description,
        target_setting=gap.target_setting,
        observed_problem=gap.observed_problem,
        existing_coverage=gap.existing_coverage,
        missing_capability=gap.missing_capability,
        claimed_delta=gap.claimed_delta,
        testable_hypothesis=gap.testable_hypothesis,
        falsification_condition=gap.falsification_condition,
        provenance_status=gap.provenance_status or "partial",
        question_ids=json.loads(gap.question_ids_json or "[]"),
        supporting_evidence_ids=json.loads(gap.supporting_evidence_ids_json or "[]"),
        contradicting_evidence_ids=json.loads(gap.contradicting_evidence_ids_json or "[]"),
        mining_round=gap.mining_round,
        novelty_score=gap.novelty_score,
        feasibility_score=gap.feasibility_score,
        significance_score=gap.significance_score,
        risk_score=gap.risk_score,
        status=gap.status,
        version=gap.version,
        created_at=gap.created_at.isoformat() + "+00:00" if gap.created_at else "",
        updated_at=gap.updated_at.isoformat() + "+00:00" if gap.updated_at else "",
    )


def _audit_to_out(audit) -> GapAuditOut:
    # P1.2: the epistemic contract — "confirmed" in storage reads as
    # "survived the current audit" in the API, and novelty_confidence is
    # surfaced under its true meaning (search-coverage ranking heuristic).
    verdict_map = {"confirmed": "survived_current_audit"}
    return GapAuditOut(
        id=audit.id,
        gap_id=audit.gap_id,
        task_id=audit.task_id,
        audit_result=audit.audit_result,
        nearest_neighbor_summary=audit.nearest_neighbor_summary,
        differentiation_summary=audit.differentiation_summary,
        neighbor_paper_ids=json.loads(audit.neighbor_paper_ids_json or "[]"),
        evidence_for_gap=json.loads(audit.evidence_for_gap_json or "[]"),
        evidence_against_gap=json.loads(audit.evidence_against_gap_json or "[]"),
        remaining_delta=audit.remaining_delta,
        novelty_confidence=audit.novelty_confidence,
        audit_confidence=audit.audit_confidence,
        recommended_action=audit.recommended_action or "continue",
        rejection_reason=audit.rejection_reason,
        failure_reason_codes=json.loads(audit.failure_reason_codes_json or "[]"),
        evidence_delta=json.loads(audit.evidence_delta_json or "{}"),
        audit_round=audit.audit_round,
        created_at=audit.created_at.isoformat() + "+00:00" if audit.created_at else "",
        audit_verdict=verdict_map.get(audit.audit_result, audit.audit_result),
        search_confidence=audit.novelty_confidence,
        closest_killer_work=json.loads(getattr(audit, "killer_work_json", None) or "{}"),
        search_coverage=json.loads(getattr(audit, "search_coverage_json", None) or "{}"),
    )


def _comparison_to_out(comp) -> NeighborComparisonOut:
    return NeighborComparisonOut(
        id=comp.id,
        gap_id=comp.gap_id,
        paper_id=comp.paper_id,
        task_id=comp.task_id,
        similarity_score=comp.similarity_score,
        shared_aspects=json.loads(comp.shared_aspects_json or "[]"),
        differentiating_aspects=json.loads(comp.differentiating_aspects_json or "[]"),
        overlap_risk=comp.overlap_risk,
        shared_problem=comp.shared_problem,
        shared_mechanism=comp.shared_mechanism,
        shared_evaluation=comp.shared_evaluation,
        covered_claims=json.loads(comp.covered_claims_json or "[]"),
        uncovered_claims=json.loads(comp.uncovered_claims_json or "[]"),
        overlap_ratio=comp.overlap_ratio,
        created_at=comp.created_at.isoformat() + "+00:00" if comp.created_at else "",
    )


@router.get("/tasks/{task_id}/gaps", response_model=list[GapCandidateOut])
def list_gaps(
    task_id: str,
    include_superseded: bool = Query(False, description="Include superseded gaps"),
    contract_id: str | None = Query(None, description="Filter by contract ID"),
    db: Session = Depends(get_db_session),
):
    """List gaps for a task. Defaults to current active contract, non-superseded."""
    if contract_id:
        gaps = gap_repo.list_gaps_for_contract(db, task_id, contract_id, include_superseded)
    else:
        gaps = gap_repo.list_active_gaps_for_task(db, task_id, include_superseded)
    return [_gap_to_out(g) for g in gaps]


@router.get("/gaps/{gap_id}", response_model=GapCandidateOut)
def get_gap(gap_id: str, db: Session = Depends(get_db_session)):
    """Get a single gap by ID."""
    gap = gap_repo.get_gap(db, gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Gap not found")
    return _gap_to_out(gap)


@router.get("/gaps/{gap_id}/evidence", response_model=list[GapEvidenceLinkOut])
def get_gap_evidence(gap_id: str, db: Session = Depends(get_db_session)):
    """Get evidence links for a gap. This is the authoritative evidence source."""
    gap = gap_repo.get_gap(db, gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Gap not found")
    links = gap_repo.list_gap_evidence(db, gap_id)
    return [
        GapEvidenceLinkOut(
            id=l.id, gap_id=l.gap_id, evidence_id=l.evidence_id,
            relation_type=l.relation_type, relevance_score=l.relevance_score,
            created_at=l.created_at.isoformat() + "+00:00" if l.created_at else "",
        )
        for l in links
    ]


@router.get("/gaps/{gap_id}/audits", response_model=list[GapAuditOut])
def get_gap_audits(gap_id: str, db: Session = Depends(get_db_session)):
    """Get audit records for a gap."""
    gap = gap_repo.get_gap(db, gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Gap not found")
    audits = gap_repo.list_gap_audits(db, gap_id)
    return [_audit_to_out(a) for a in audits]


@router.get("/gaps/{gap_id}/neighbors", response_model=list[NeighborComparisonOut])
def get_gap_neighbors(gap_id: str, db: Session = Depends(get_db_session)):
    """Get neighbor comparisons for a gap."""
    gap = gap_repo.get_gap(db, gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Gap not found")
    comparisons = gap_repo.list_neighbor_comparisons(db, gap_id)
    return [_comparison_to_out(c) for c in comparisons]

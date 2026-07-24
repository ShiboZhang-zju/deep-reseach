"""Gap repository — CRUD for GapCandidate, GapEvidenceLink, GapAudit, NeighborComparison.

Phase 3A Closure:
- All write operations validate task_id consistency across related entities.
- gap_evidence_links is the authoritative source for Gap↔Evidence.
- supporting/contradicting_evidence_ids_json are deprecated snapshots.
"""

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    GapCandidate, GapEvidenceLink, GapAudit, NeighborComparison,
    EvidenceUnit, ResearchContract,
)

logger = logging.getLogger(__name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CrossTaskValidationError(Exception):
    """Raised when trying to link entities from different tasks."""


# === GapCandidate ===

def create_gap_candidate(
    db: Session, task_id: str, gap_type: str, description: str,
    contract_id: str | None = None,
    target_setting: str | None = None,
    observed_problem: str | None = None,
    existing_coverage: str | None = None,
    missing_capability: str | None = None,
    claimed_delta: str | None = None,
    testable_hypothesis: str | None = None,
    falsification_condition: str | None = None,
    provenance_status: str = "partial",
    question_ids: list[str] | None = None,
    mining_round: int = 0,
    novelty_score: float | None = None,
    feasibility_score: float | None = None,
    significance_score: float | None = None,
    risk_score: float | None = None,
    status: str = "candidate",
) -> GapCandidate:
    """Create a new gap candidate."""
    # Validate contract belongs to this task
    if contract_id:
        contract = db.get(ResearchContract, contract_id)
        if not contract or contract.task_id != task_id:
            raise CrossTaskValidationError(
                f"Contract {contract_id} does not belong to task {task_id}"
            )

    gap = GapCandidate(
        task_id=task_id,
        contract_id=contract_id,
        gap_type=gap_type,
        description=description,
        target_setting=target_setting,
        observed_problem=observed_problem,
        existing_coverage=existing_coverage,
        missing_capability=missing_capability,
        claimed_delta=claimed_delta,
        testable_hypothesis=testable_hypothesis,
        falsification_condition=falsification_condition,
        provenance_status=provenance_status,
        question_ids_json=json.dumps(question_ids or []),
        mining_round=mining_round,
        novelty_score=novelty_score,
        feasibility_score=feasibility_score,
        significance_score=significance_score,
        risk_score=risk_score,
        status=status,
    )
    db.add(gap)
    db.flush()
    return gap


def get_gap(db: Session, gap_id: str) -> GapCandidate | None:
    return db.get(GapCandidate, gap_id)


def list_active_gaps_for_task(
    db: Session, task_id: str,
    include_superseded: bool = False,
) -> list[GapCandidate]:
    """List gaps for a task, excluding superseded by default."""
    query = db.query(GapCandidate).filter(GapCandidate.task_id == task_id)
    if not include_superseded:
        query = query.filter(GapCandidate.status != "superseded")
    return query.order_by(GapCandidate.created_at).all()


def list_gaps_for_contract(
    db: Session, task_id: str, contract_id: str,
    include_superseded: bool = False,
) -> list[GapCandidate]:
    """List gaps belonging to a specific contract."""
    query = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == contract_id,
    )
    if not include_superseded:
        query = query.filter(GapCandidate.status != "superseded")
    return query.order_by(GapCandidate.created_at).all()


def supersede_contract_gaps(db: Session, task_id: str, contract_id: str) -> int:
    """Mark all gaps for a contract as superseded (when contract changes)."""
    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == contract_id,
        GapCandidate.status != "superseded",
    ).all()
    now = _utcnow()
    for g in gaps:
        g.status = "superseded"
        g.superseded_at = now
    db.flush()
    return len(gaps)


# === GapEvidenceLink (authoritative source) ===

def create_gap_evidence_link(
    db: Session, gap_id: str, evidence_id: str,
    relation_type: str = "suggests", relevance_score: float = 0.5,
) -> GapEvidenceLink:
    """Create a gap-evidence link. Validates task consistency."""
    gap = db.get(GapCandidate, gap_id)
    if not gap:
        raise ValueError(f"Gap {gap_id} not found")

    evidence = db.get(EvidenceUnit, evidence_id)
    if not evidence:
        raise ValueError(f"Evidence {evidence_id} not found")

    if gap.task_id != evidence.task_id:
        raise CrossTaskValidationError(
            f"Gap task {gap.task_id} != Evidence task {evidence.task_id}"
        )

    # Check for existing (idempotent)
    existing = db.query(GapEvidenceLink).filter(
        GapEvidenceLink.gap_id == gap_id,
        GapEvidenceLink.evidence_id == evidence_id,
    ).first()
    if existing:
        return existing

    link = GapEvidenceLink(
        gap_id=gap_id,
        evidence_id=evidence_id,
        relation_type=relation_type,
        relevance_score=relevance_score,
    )
    db.add(link)
    db.flush()
    return link


def list_gap_evidence(db: Session, gap_id: str) -> list[GapEvidenceLink]:
    """List all evidence links for a gap. This is the authoritative source."""
    return db.query(GapEvidenceLink).filter(
        GapEvidenceLink.gap_id == gap_id,
    ).all()


def replace_gap_evidence_links(
    db: Session, gap_id: str,
    evidence_links: list[dict],
) -> int:
    """Replace all evidence links for a gap.

    evidence_links: [{"evidence_id": "...", "relation_type": "...", "relevance_score": 0.5}]
    """
    # Delete existing
    db.query(GapEvidenceLink).filter(
        GapEvidenceLink.gap_id == gap_id,
    ).delete()

    count = 0
    for link_data in evidence_links:
        create_gap_evidence_link(
            db, gap_id,
            link_data["evidence_id"],
            link_data.get("relation_type", "suggests"),
            link_data.get("relevance_score", 0.5),
        )
        count += 1
    db.flush()
    return count


# === GapAudit ===

def create_gap_audit(
    db: Session, gap_id: str, task_id: str,
    adversarial_queries: list[str] | None = None,
    audit_result: str = "pending",
    nearest_neighbor_summary: str | None = None,
    differentiation_summary: str | None = None,
    neighbor_paper_ids: list[str] | None = None,
    evidence_for_gap: list[str] | None = None,
    evidence_against_gap: list[str] | None = None,
    remaining_delta: str | None = None,
    novelty_confidence: float | None = None,
    audit_confidence: float | None = None,
    recommended_action: str = "continue",
    rejection_reason: str | None = None,
    audit_round: int = 0,
) -> GapAudit:
    """Create a gap audit record. Validates task consistency."""
    gap = db.get(GapCandidate, gap_id)
    if not gap:
        raise ValueError(f"Gap {gap_id} not found")
    if gap.task_id != task_id:
        raise CrossTaskValidationError(
            f"Gap task {gap.task_id} != Audit task {task_id}"
        )

    audit = GapAudit(
        gap_id=gap_id,
        task_id=task_id,
        adversarial_queries_json=json.dumps(adversarial_queries or []),
        audit_result=audit_result,
        nearest_neighbor_summary=nearest_neighbor_summary,
        differentiation_summary=differentiation_summary,
        neighbor_paper_ids_json=json.dumps(neighbor_paper_ids or []),
        evidence_for_gap_json=json.dumps(evidence_for_gap or []),
        evidence_against_gap_json=json.dumps(evidence_against_gap or []),
        remaining_delta=remaining_delta,
        novelty_confidence=novelty_confidence,
        audit_confidence=audit_confidence,
        recommended_action=recommended_action,
        rejection_reason=rejection_reason,
        audit_round=audit_round,
    )
    db.add(audit)
    db.flush()
    return audit


def list_gap_audits(db: Session, gap_id: str) -> list[GapAudit]:
    return db.query(GapAudit).filter(
        GapAudit.gap_id == gap_id,
    ).order_by(GapAudit.created_at).all()


# === NeighborComparison ===

def create_neighbor_comparison(
    db: Session, gap_id: str, paper_id: str, task_id: str,
    similarity_score: float = 0.0,
    shared_problem: str | None = None,
    shared_mechanism: str | None = None,
    shared_evaluation: str | None = None,
    covered_claims: list[str] | None = None,
    uncovered_claims: list[str] | None = None,
    overlap_ratio: float = 0.0,
    overlap_risk: float = 0.0,
) -> NeighborComparison:
    """Create a neighbor comparison. Validates task consistency."""
    gap = db.get(GapCandidate, gap_id)
    if not gap:
        raise ValueError(f"Gap {gap_id} not found")
    if gap.task_id != task_id:
        raise CrossTaskValidationError(
            f"Gap task {gap.task_id} != Comparison task {task_id}"
        )

    comp = NeighborComparison(
        gap_id=gap_id,
        paper_id=paper_id,
        task_id=task_id,
        similarity_score=similarity_score,
        shared_problem=shared_problem,
        shared_mechanism=shared_mechanism,
        shared_evaluation=shared_evaluation,
        covered_claims_json=json.dumps(covered_claims or []),
        uncovered_claims_json=json.dumps(uncovered_claims or []),
        overlap_ratio=overlap_ratio,
        overlap_risk=overlap_risk,
    )
    db.add(comp)
    db.flush()
    return comp


def list_neighbor_comparisons(db: Session, gap_id: str) -> list[NeighborComparison]:
    return db.query(NeighborComparison).filter(
        NeighborComparison.gap_id == gap_id,
    ).order_by(NeighborComparison.similarity_score.desc()).all()

"""Phase 2 Readiness Gate — formal evidence/coverage readiness check.

Phase 2.2A Final Closure (#4): Replaces the ad-hoc `evidence_count > 0`
check with a structured gate that verifies the entire control plane
(Evidence + Coverage) is complete and consistent before allowing
downstream report generation.
"""

import json
import logging
from dataclasses import dataclass, field
from sqlalchemy import func

from app.db.models import (
    ResearchContract,
    ResearchQuestion,
    EvidenceUnit,
    CoverageRecord,
)

logger = logging.getLogger(__name__)

# Importance threshold for "high importance" questions
HIGH_IMPORTANCE_THRESHOLD = 0.7

# O3: Minimum fraction of high-importance questions that must have a latest
# coverage snapshot for the pipeline to proceed. Below this, we do NOT hard-fail
# the whole task — we downgrade to more_research_required so the task can still
# surface a landscape brief and (optionally) trigger targeted follow-up search.
MIN_HIGH_IMPORTANCE_SNAPSHOT_RATIO = 0.6


@dataclass
class Phase2ReadinessResult:
    """Result of Phase 2 readiness evaluation."""
    ready: bool
    status: str  # "ready" / "failed" / "more_research_required"
    reason: str
    active_question_count: int
    questions_with_latest_snapshot: int
    high_importance_question_count: int
    high_importance_covered_count: int
    evidence_count: int
    valid_evidence_count: int
    latest_round: int
    missing_question_ids: list[str] = field(default_factory=list)
    unresolved_question_ids: list[str] = field(default_factory=list)


def _get_latest_coverage_per_question(db, task_id: str,
                                      active_question_ids: list[str]) -> dict:
    """Get the latest CoverageRecord for each question.

    Uses MAX(round_number) GROUP BY question_id — not a global max round.

    Returns: {question_id: CoverageRecord}
    """
    if not active_question_ids:
        return {}

    # Subquery: max round_number per question
    max_round_subq = db.query(
        CoverageRecord.question_id.label("q_id"),
        func.max(CoverageRecord.round_number).label("max_round"),
    ).filter(
        CoverageRecord.task_id == task_id,
        CoverageRecord.question_id.in_(active_question_ids),
    ).group_by(CoverageRecord.question_id).subquery()

    # Join back to get the actual records
    records = db.query(CoverageRecord).join(
        max_round_subq,
        (CoverageRecord.question_id == max_round_subq.c.q_id) &
        (CoverageRecord.round_number == max_round_subq.c.max_round) &
        (CoverageRecord.task_id == task_id),
    ).all()

    return {r.question_id: r for r in records}


def evaluate_phase2_readiness(db, task_id: str,
                              contract_id: str | None) -> Phase2ReadinessResult:
    """Evaluate whether the pipeline is ready for downstream (report/ideas).

    Gate data scope (ONLY these):
    - Current active ResearchContract
    - Active ResearchQuestions belonging to this Contract
    - Non-rejected, non-conflicted EvidenceUnits for this task
    - Latest CoverageRecord per active Question

    Returns:
    - status="failed": Control plane is incomplete (missing contract,
      questions, snapshots, or evidence). Blocks everything.
    - status="more_research_required": Control plane is complete but
      high-importance questions have insufficient coverage.
    - status="ready": Can proceed to report/ideas generation.
    """
    # --- 1. Active Contract exists ---
    if not contract_id:
        return Phase2ReadinessResult(
            ready=False, status="failed",
            reason="no_active_contract",
            active_question_count=0,
            questions_with_latest_snapshot=0,
            high_importance_question_count=0,
            high_importance_covered_count=0,
            evidence_count=0, valid_evidence_count=0, latest_round=0,
        )

    contract = db.get(ResearchContract, contract_id)
    if not contract or contract.status != "active":
        return Phase2ReadinessResult(
            ready=False, status="failed",
            reason=f"contract_not_active ({contract.status if contract else 'not_found'})",
            active_question_count=0,
            questions_with_latest_snapshot=0,
            high_importance_question_count=0,
            high_importance_covered_count=0,
            evidence_count=0, valid_evidence_count=0, latest_round=0,
        )

    # --- 2. Active Questions > 0 ---
    active_questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.contract_id == contract_id,
        ResearchQuestion.status.in_(["open", "partially_covered", "covered"]),
    ).all()

    active_question_ids = [q.id for q in active_questions]

    if not active_questions:
        return Phase2ReadinessResult(
            ready=False, status="failed",
            reason="no_active_questions",
            active_question_count=0,
            questions_with_latest_snapshot=0,
            high_importance_question_count=0,
            high_importance_covered_count=0,
            evidence_count=0, valid_evidence_count=0, latest_round=0,
        )

    # --- 3. Valid Evidence > 0 ---
    all_evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
        ~EvidenceUnit.verification_status.in_(["rejected", "conflicted"]),
    ).all()

    if not all_evidence:
        return Phase2ReadinessResult(
            ready=False, status="failed",
            reason="no_valid_evidence",
            active_question_count=len(active_questions),
            questions_with_latest_snapshot=0,
            high_importance_question_count=0,
            high_importance_covered_count=0,
            evidence_count=0, valid_evidence_count=0, latest_round=0,
        )

    # --- 4. Latest Coverage per Question ---
    latest_cov = _get_latest_coverage_per_question(db, task_id, active_question_ids)

    high_importance_qs = [q for q in active_questions
                         if (q.importance or 0) >= HIGH_IMPORTANCE_THRESHOLD]
    high_importance_ids = [q.id for q in high_importance_qs]

    # Questions missing latest snapshot
    missing_snapshot_ids = [q.id for q in high_importance_qs
                            if q.id not in latest_cov]

    # O3: Do NOT hard-fail on any single missing snapshot. Only fail when the
    # control plane is genuinely broken (no high-importance question has any
    # snapshot at all). When coverage is partial, downgrade to
    # more_research_required so the task can still emit a landscape brief and
    # optionally trigger targeted follow-up search — instead of dying.
    if high_importance_qs:
        snapshot_ratio = (len(high_importance_qs) - len(missing_snapshot_ids)) / len(high_importance_qs)
    else:
        snapshot_ratio = 1.0

    if missing_snapshot_ids and snapshot_ratio == 0.0:
        # Total control-plane failure: not a single high-importance question
        # produced a coverage snapshot.
        return Phase2ReadinessResult(
            ready=False, status="failed",
            reason=f"no_high_importance_question_has_coverage_snapshot ({len(missing_snapshot_ids)})",
            active_question_count=len(active_questions),
            questions_with_latest_snapshot=len(latest_cov),
            high_importance_question_count=len(high_importance_qs),
            high_importance_covered_count=0,
            evidence_count=len(all_evidence),
            valid_evidence_count=len(all_evidence),
            latest_round=max((r.round_number for r in latest_cov.values()), default=0),
            missing_question_ids=missing_snapshot_ids,
        )

    if missing_snapshot_ids and snapshot_ratio < MIN_HIGH_IMPORTANCE_SNAPSHOT_RATIO:
        # Partial coverage below the acceptable ratio: recoverable, not fatal.
        return Phase2ReadinessResult(
            ready=False, status="more_research_required",
            reason=(f"high_importance_coverage_below_ratio "
                    f"({snapshot_ratio:.2f} < {MIN_HIGH_IMPORTANCE_SNAPSHOT_RATIO})"),
            active_question_count=len(active_questions),
            questions_with_latest_snapshot=len(latest_cov),
            high_importance_question_count=len(high_importance_qs),
            high_importance_covered_count=len(high_importance_qs) - len(missing_snapshot_ids),
            evidence_count=len(all_evidence),
            valid_evidence_count=len(all_evidence),
            latest_round=max((r.round_number for r in latest_cov.values()), default=0),
            missing_question_ids=missing_snapshot_ids,
            unresolved_question_ids=missing_snapshot_ids,
        )
    # else: snapshot_ratio >= MIN ratio (or nothing missing) — proceed, the
    # remaining questions without snapshots are simply treated as uncovered
    # in the downstream coverage check below.

    # --- 5. Check if high-importance questions have any coverage ---
    high_covered = 0
    unresolved = []
    for q in high_importance_qs:
        cov = latest_cov.get(q.id)
        if cov and (cov.coverage_score > 0 or cov.unavailable_reason):
            high_covered += 1
        else:
            unresolved.append(q.id)

    # At least one high-importance question must have coverage > 0 or unavailable_reason
    if high_covered == 0 and high_importance_qs:
        return Phase2ReadinessResult(
            ready=False, status="more_research_required",
            reason="high_importance_questions_have_no_coverage",
            active_question_count=len(active_questions),
            questions_with_latest_snapshot=len(latest_cov),
            high_importance_question_count=len(high_importance_qs),
            high_importance_covered_count=0,
            evidence_count=len(all_evidence),
            valid_evidence_count=len(all_evidence),
            latest_round=max((r.round_number for r in latest_cov.values()), default=0),
            unresolved_question_ids=unresolved,
        )

    # All checks passed — ready for downstream
    latest_round = max((r.round_number for r in latest_cov.values()), default=0)
    return Phase2ReadinessResult(
        ready=True, status="ready",
        reason="all_checks_passed",
        active_question_count=len(active_questions),
        questions_with_latest_snapshot=len(latest_cov),
        high_importance_question_count=len(high_importance_qs),
        high_importance_covered_count=high_covered,
        evidence_count=len(all_evidence),
        valid_evidence_count=len(all_evidence),
        latest_round=latest_round,
    )

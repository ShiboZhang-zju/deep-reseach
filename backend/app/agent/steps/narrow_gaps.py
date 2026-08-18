"""Narrow gap candidates that a neighbour audit only partially closed.

The audit can return `partially_closed` with a concrete `remaining_delta`: the
neighbours cover part of the claimed contribution, and what is left is stated
explicitly. `generate_interventions` already refuses such a gap with "Gap 需先
收窄后再判断新颖性", and the audit sets the gap to `audited` expecting it to be
narrowed — but no narrowing step existed, so those gaps were dead ends. The
pipeline then burned full remediation rounds (roughly 19 minutes each) looking
for more papers, even though the audit had already told us precisely how to
shrink the claim.

Narrowing here is deliberately deterministic and LLM-free: adopt the audit's own
`remaining_delta` as the gap's new claimed delta, and fold what the neighbours
already cover into `existing_coverage`. That is exactly the information the audit
produced, so no new claim is invented; the narrowed gap is then re-audited
against the neighbours and must earn `confirmed` on its own.
"""

import json
import logging
from datetime import datetime, timezone

from app.agent.state import ResearchState
from app.db.models import GapAudit, GapCandidate
from app.db.repositories import gap_repo, paper_repo

logger = logging.getLogger(__name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

# A gap may be narrowed at most this many times. Without a cap, a gap whose
# audit keeps returning partially_closed would be narrowed indefinitely, and
# each pass makes the claim smaller until it is no longer worth proposing.
MAX_NARROW_ATTEMPTS = 2

_MIN_REMAINING_DELTA_CHARS = 20


def _latest_audit(db, gap_id: str) -> GapAudit | None:
    audits = gap_repo.list_gap_audits(db, gap_id)
    return audits[-1] if audits else None


def _narrow_attempts(db, gap_id: str) -> int:
    """Total narrowing passes across a gap's lineage.

    Each narrowed version carries only its own audits, so the cap must count
    every `narrow` action from the lineage root down — otherwise a gap could be
    narrowed indefinitely, one fresh version at a time.
    """
    total = 0
    current_id = gap_id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        gap = db.get(GapCandidate, current_id)
        if gap is None:
            break
        total += sum(1 for audit in gap_repo.list_gap_audits(db, current_id)
                     if audit.recommended_action == "narrow")
        current_id = gap.parent_gap_id
    return total


def narrow_audited_gaps(db, state: ResearchState, task_id: str) -> list[str]:
    """Rewrite partially closed gaps around their remaining delta.

    Returns the IDs of the gaps that were narrowed and are ready for re-audit.
    """
    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == state.contract_id,
        GapCandidate.status == "audited",
    ).all()

    narrowed = []
    skipped = []
    for gap in gaps:
        audit = _latest_audit(db, gap.id)
        if audit is None or audit.audit_result != "partially_closed":
            skipped.append({"gap_id": gap.id, "reason": "NO_PARTIALLY_CLOSED_AUDIT"})
            continue
        remaining = (audit.remaining_delta or "").strip()
        if len(remaining) < _MIN_REMAINING_DELTA_CHARS:
            # Without a concrete remaining delta there is nothing to narrow to,
            # and inventing one would be exactly the unsupported claim the audit
            # is meant to prevent.
            skipped.append({"gap_id": gap.id, "reason": "NO_CONCRETE_REMAINING_DELTA"})
            continue
        attempts = _narrow_attempts(db, gap.id)
        if attempts > MAX_NARROW_ATTEMPTS:
            skipped.append({"gap_id": gap.id, "reason": "NARROW_ATTEMPTS_EXHAUSTED",
                            "attempts": attempts})
            continue
        if (gap.claimed_delta or "").strip() == remaining:
            # Already narrowed to this delta and the audit still says partially
            # closed: narrowing again would not change the input.
            skipped.append({"gap_id": gap.id, "reason": "DELTA_UNCHANGED"})
            continue

        covered = (audit.nearest_neighbor_summary or "").strip()
        previous_coverage = (gap.existing_coverage or "").strip()
        merged_coverage = previous_coverage
        if covered and covered not in previous_coverage:
            merged_coverage = (
                f"{previous_coverage}\n[narrowed after audit] {covered}".strip())

        # Version the gap instead of overwriting it. The pre-narrow claim is
        # part of the lineage, and a narrowed claim is a *new version* of the
        # same canonical gap — never a sibling row that a later audit could
        # independently judge to a contradictory verdict.
        child = gap_repo.create_gap_candidate(
            db,
            task_id=task_id,
            contract_id=gap.contract_id,
            gap_type=gap.gap_type,
            description=gap.description,
            target_setting=gap.target_setting,
            observed_problem=gap.observed_problem,
            existing_coverage=merged_coverage,
            missing_capability=gap.missing_capability,
            claimed_delta=remaining,
            testable_hypothesis=gap.testable_hypothesis,
            falsification_condition=gap.falsification_condition,
            provenance_status=gap.provenance_status,
            question_ids=json.loads(gap.question_ids_json or "[]"),
            mining_round=gap.mining_round,
            mining_policy_version=gap.mining_policy_version,
            novelty_score=gap.novelty_score,
            feasibility_score=gap.feasibility_score,
            significance_score=gap.significance_score,
            risk_score=gap.risk_score,
            status="auditing",
            version=(gap.version or 1) + 1,
            canonical_gap_id=gap.canonical_gap_id or gap.id,
            parent_gap_id=gap.id,
        )
        # The evidence behind the gap is unchanged by narrowing; only the claim
        # is smaller, so the child inherits the same evidence links.
        for link in gap_repo.list_gap_evidence(db, gap.id):
            gap_repo.create_gap_evidence_link(
                db, child.id, link.evidence_id, link.relation_type,
                link.relevance_score or 0.5,
            )
        # Retire the parent so it cannot surface as an independent gap.
        gap.status = "superseded"
        gap.superseded_at = _utcnow()
        narrowed.append(child.id)
        logger.info("Gap %s: narrowed to v%d (child %s, attempt %d)",
                    gap.id[:8], child.version, child.id[:8], attempts)

    if narrowed or skipped:
        paper_repo.save_trace(db, task_id, "narrow_gaps", "decision", output_data={
            "narrowed_gap_ids": narrowed, "skipped": skipped,
        })
    db.commit()
    return narrowed

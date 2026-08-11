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

import logging

from app.agent.state import ResearchState
from app.db.models import GapAudit, GapCandidate
from app.db.repositories import gap_repo, paper_repo

logger = logging.getLogger(__name__)

# A gap may be narrowed at most this many times. Without a cap, a gap whose
# audit keeps returning partially_closed would be narrowed indefinitely, and
# each pass makes the claim smaller until it is no longer worth proposing.
MAX_NARROW_ATTEMPTS = 2

_MIN_REMAINING_DELTA_CHARS = 20


def _latest_audit(db, gap_id: str) -> GapAudit | None:
    audits = gap_repo.list_gap_audits(db, gap_id)
    return audits[-1] if audits else None


def _narrow_attempts(db, gap_id: str) -> int:
    return sum(1 for audit in gap_repo.list_gap_audits(db, gap_id)
               if audit.recommended_action == "narrow")


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
        if covered and covered not in previous_coverage:
            gap.existing_coverage = (
                f"{previous_coverage}\n[narrowed after audit] {covered}".strip())
        gap.claimed_delta = remaining
        # Back to the audit queue: the narrowed claim has not been audited yet.
        gap.status = "auditing"
        narrowed.append(gap.id)
        logger.info("Gap %s: narrowed to the audited remaining delta (attempt %d)",
                    gap.id[:8], attempts)

    if narrowed or skipped:
        paper_repo.save_trace(db, task_id, "narrow_gaps", "decision", output_data={
            "narrowed_gap_ids": narrowed, "skipped": skipped,
        })
    db.commit()
    return narrowed

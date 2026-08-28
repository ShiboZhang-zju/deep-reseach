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
import re
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

# P0-2 (2026-08-28, task d6f64087): a partially_closed audit's remaining_delta
# is sometimes a HEDGE — "The audit cannot confirm novelty because... The
# evidence is insufficient to rule out prior work" — an explanation of why
# novelty could not be confirmed, not a positive delta claim. Copying it
# verbatim as the narrowed gap's claimed_delta inverts the semantics: the
# audit's DOUBT became the gap's CLAIM (observed: surviving gap 475c8acd's
# claimed_delta was the audit's uncertainty text verbatim). This pattern
# mirrors the _AUDIT_TEXT_CONFLICT_RE family in audit_gaps.py (kept local:
# importing audit_gaps here would pull the whole audit stack into module load).
_NARROW_HEDGE_RE = re.compile(
    r"\bdecision is (?:uncertain|inconclusive)\b"
    r"|\b(?:cannot|unable to) (?:confirm|conclude)\b"
    r"|\binsufficient evidence to (?:confirm|rule out)\b"
    r"|\bnovelty (?:cannot|can not) be (?:confirmed|established)\b"
    r"|\bthe audit cannot\b",
    re.IGNORECASE,
)

_DISTILL_SYSTEM = """You rewrite an audit conclusion into a positive research delta claim.
The input text is an auditor explaining WHY it could not confirm a research gap's novelty.
Extract the concrete unverified research territory the text implies and state it as a
positive claim of what remains unaddressed (e.g. "The intersection of X and Y in setting Z
remains unverified in prior work"). Output ONE sentence of at most 60 words, in the same
language as the input. Never use hedging words like "cannot confirm", "unable to" or
"insufficient evidence"."""

_DISTILL_USER = """Audit conclusion to distill:
{text}

Positive delta claim (one sentence):"""


async def _distill_positive_delta(llm, hedge_text: str) -> str:
    """One bounded LLM call to turn a hedged audit conclusion into a positive
    delta claim. Returns "" on any failure so the caller can degrade to the
    raw hedge text (a polluted claim beats a dropped gap)."""
    if llm is None or not hasattr(llm, "chat"):
        return ""
    try:
        answer = await llm.chat(
            [{"role": "system", "content": _DISTILL_SYSTEM},
             {"role": "user", "content": _DISTILL_USER.format(text=hedge_text)}],
            temperature=0.2,
        )
        claim = (answer or "").strip().strip('"')
        # The distillation itself must not be a hedge — if the model echoed
        # the doubt, we gained nothing and fall back to the raw text.
        if not claim or _NARROW_HEDGE_RE.search(claim):
            return ""
        return claim
    except Exception as distill_err:
        logger.warning("Hedge distillation failed (falling back to raw "
                       "remaining_delta): %s", distill_err)
        return ""


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


async def narrow_audited_gaps(db, state: ResearchState, task_id: str, llm=None) -> list[str]:
    """Rewrite partially closed gaps around their remaining delta.

    Returns the IDs of the gaps that were narrowed and are ready for re-audit.
    P0-2: `llm` is optional and only used for hedge distillation — a missing or
    failing LLM degrades to the pre-P0-2 verbatim behaviour, never blocks
    narrowing.
    Spun-gap finalization (2026-08-28): a gap whose narrowing budget is
    exhausted (or whose claim already equals the audit's remaining delta) can
    neither be confirmed nor narrowed further, but its "audited" status keeps
    it in the runner's re-audit loop forever. Such gaps are finalized as
    rejected here, redirecting the remediation budget to fresh candidates.
    """
    gaps = db.query(GapCandidate).filter(
        GapCandidate.task_id == task_id,
        GapCandidate.contract_id == state.contract_id,
        GapCandidate.status == "audited",
    ).all()

    narrowed = []
    skipped = []
    hedge_traces = []
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
        # P0-2: a hedged remaining_delta ("The audit cannot confirm novelty
        # because...") is the auditor's doubt, not a delta claim. Distill it
        # into a positive claim before it becomes the narrowed gap's
        # claimed_delta; keep the raw text only when distillation is
        # unavailable (degradation > dropped gap) and trace the fallback.
        distillation = {"hedged": False}
        if _NARROW_HEDGE_RE.search(remaining):
            distilled = await _distill_positive_delta(llm, remaining)
            if distilled:
                remaining = distilled
                distillation = {"hedged": True, "distilled": True,
                                "original_hedge": (audit.remaining_delta or "")[:300]}
            else:
                distillation = {"hedged": True, "distilled": False,
                                "fallback": "raw_hedge_kept"}
        attempts = _narrow_attempts(db, gap.id)
        if attempts >= MAX_NARROW_ATTEMPTS:
            # 空转修复（2026-08-28，任务 94caca35）：额度耗尽而判决仍是
            # partially_closed 时，该 gap 既无法确认也无法进一步收窄，但
            # 留在 "audited" 状态会被 runner 的复审通道反复捞起重审——实测
            # 同一 gap 连续三轮完整审计（每轮 ~11 分钟对抗检索），判决
            # 0.9/0.85/0.85 零变化，25 分钟纯浪费。终态化为 rejected
            # （复用 novelty_undecidable 的"无法定论"先例状态），复审通道
            # 与 survivor 查询都不再捞它，remediation 预算转向挖掘新候选。
            gap.status = "rejected"
            gap.updated_at = _utcnow()
            db.commit()
            skipped.append({"gap_id": gap.id, "reason": "NARROW_ATTEMPTS_EXHAUSTED",
                            "attempts": attempts, "finalized_as": "rejected",
                            "final_reason": "narrow_budget_exhausted"})
            continue
        if (gap.claimed_delta or "").strip() == remaining:
            # Already narrowed to this delta and the audit still says partially
            # closed: narrowing again would not change the input. The claim is
            # self-covered at this granularity — same spin risk as an exhausted
            # budget, so finalize instead of leaving it in the re-audit pool.
            gap.status = "rejected"
            gap.updated_at = _utcnow()
            db.commit()
            skipped.append({"gap_id": gap.id, "reason": "DELTA_UNCHANGED",
                            "finalized_as": "rejected",
                            "final_reason": "claim_self_covered"})
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
        if distillation.get("hedged"):
            distillation["gap_id"] = gap.id
            distillation["child_gap_id"] = child.id
            hedge_traces.append(distillation)
        logger.info("Gap %s: narrowed to v%d (child %s, attempt %d)",
                    gap.id[:8], child.version, child.id[:8], attempts)

    if narrowed or skipped:
        paper_repo.save_trace(db, task_id, "narrow_gaps", "decision", output_data={
            "narrowed_gap_ids": narrowed, "skipped": skipped,
            "hedge_distillations": hedge_traces,
        })
    db.commit()
    return narrowed

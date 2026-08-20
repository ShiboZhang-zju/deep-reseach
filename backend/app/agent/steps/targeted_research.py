"""Step: O2 — Targeted remediation search.

When a pipeline gate fails (readiness / no gap / no surviving gap / no
intervention), instead of terminating immediately, run one directed search
round aimed at the *specific* failure reason, then let the caller retry the
gate.

Design principles:
- Deterministic reason -> intent/query mapping (no reliance on LLM to decide
  what to search for; the LLM only expands seed phrases into concrete queries).
- Reuses the existing per-round pipeline (search -> score -> evidence ->
  coverage) so a remediation round is indistinguishable from a normal round
  downstream.
- Bounded by settings.max_remediation_attempts (per reason) and
  settings.max_remediation_rounds_total (per task) to cap runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.config import settings
from app.agent.state import ResearchState
from app.db.repositories import paper_repo, task_repo
from app.db.repositories.search_query_repo import save_search_query
from app.agent.steps.generate_queries import SearchQueryExecution
from app.agent.steps.search_papers import search_and_save_papers
from app.agent.steps.score_papers import score_papers
from app.agent.steps.extract_evidence import extract_evidence_units
from app.agent.steps.update_coverage import update_coverage_matrix
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


# Map a failure reason (stop_reason / reason_code) to the search intents and
# seed query fragments most likely to unblock the corresponding gate.
# Each entry: (intent, list of seed phrase templates using {topic}).
_REASON_PLAYBOOK: dict[str, tuple[str, list[str]]] = {
    # Readiness / coverage shortfall — broaden evidence for under-covered questions.
    "readiness_more_research": (
        "question_answering",
        ["{topic} survey", "{topic} systematic review", "{topic} state of the art"],
    ),
    # No evidence-backed gap — we lack limitation / negative-result signals.
    "no_evidence_backed_gap_candidates": (
        "limitation",
        [
            "limitations of {topic}",
            "failure cases {topic}",
            "{topic} negative results",
            "challenges open problems {topic}",
        ],
    ),
    # Gap survived mining but audit could not confirm — we lack external neighbors.
    # Seed phrases target *method-level* comparisons (the direct neighbors the
    # audit needs), not a broad "recent advances" sweep that recalls mostly
    # low-priority, off-mechanism papers.
    "no_surviving_gap_after_audit": (
        "direct_neighbor",
        [
            "{topic} method comparison",
            "{topic} ablation study",
            "{topic} mechanism analysis",
        ],
    ),
    # Interventions failed hard gates — need stronger evidence / feasibility anchors.
    "no_intervention_passed_hard_gates": (
        "benchmark",
        [
            "{topic} evaluation methodology",
            "{topic} empirical study",
            "reproducible {topic}",
        ],
    ),
}

# Reasons that are eligible for remediation. Others (e.g. hard failures,
# identical_error_streak) are not retried by directed search.
REMEDIABLE_REASONS = set(_REASON_PLAYBOOK.keys())


@dataclass
class RemediationResult:
    attempted: bool
    reason: str
    new_paper_count: int
    new_evidence_count: int
    exhausted: bool  # True when budget/attempts are used up


def _reason_key(reason: str) -> str | None:
    """Normalize a possibly-prefixed reason string to a playbook key."""
    if not reason:
        return None
    for key in _REASON_PLAYBOOK:
        if reason == key or reason.startswith(key):
            return key
    return None


def can_remediate(state: ResearchState, reason: str,
                  service_families: list[str] | None = None) -> bool:
    """Return True if a directed remediation round is allowed for this reason.

    Two budgets apply:
    - Task-level hard cap (max_remediation_rounds_total): a global safety net
      over ALL remediation rounds so runtime cannot balloon. Applies to every
      reason.
    - Per-canonical-gap-family budget (max_remediation_attempts_per_gap): a
      remediation round answers "how much more retrieval does THIS gap's novelty
      need?", so it is charged to the canonical gap family that consumed it. A
      NEW family that appears after a mining revision starts at 0 and must NOT
      inherit an old family's spend (task 08005641: old gap A ate the budget,
      then gaps B/C appeared and were left with zero remediation chances).
      Narrowed versions (v1 -> v2) share the family budget.

    For gap-scoped reasons the round proceeds only if at least one served
    canonical family still has per-family budget; the task hard cap still
    bounds the total.
    """
    if settings.max_remediation_attempts <= 0:
        return False
    key = _reason_key(reason)
    if key is None:
        return False
    attempts = state.remediation_attempts or {}
    total = int(attempts.get("__total__", 0))
    if total >= settings.max_remediation_rounds_total:
        return False
    if key == "no_surviving_gap_after_audit":
        # Gap-scoped remediation: proceed only if at least one served canonical
        # family still has per-family budget.
        if not service_families:
            return False
        used = state.gap_remediation_used or {}
        remaining = [fam for fam in set(service_families)
                     if int(used.get(fam, 0)) < settings.max_remediation_attempts_per_gap]
        return bool(remaining)
    per_reason = int(attempts.get(key, 0))
    if per_reason >= settings.max_remediation_attempts:
        return False
    return True


def _build_seed_queries(reason_key: str, topic: str) -> list[str]:
    _intent, templates = _REASON_PLAYBOOK[reason_key]
    seeds = []
    for tpl in templates:
        try:
            seeds.append(tpl.format(topic=topic))
        except Exception:
            seeds.append(tpl)
    return seeds


class RemediationQuerySchema(BaseModel):
    queries: list[str] = Field(description="English search queries, one per line")


_REMEDIATION_QUERY_SYSTEM = """You are generating academic search queries for a targeted literature search.
A research gap could not be confirmed as novel because the audit could not find
direct method-level comparisons. Generate search queries that would retrieve the
most relevant prior work that could already close these specific gaps.

Rules:
- Output English only.
- Anchor each query on a gap's specific mechanism or claim, not the broad topic.
- Derive terms from the supplied gap descriptions; do not invent new concepts."""

_REMEDIATION_QUERY_USER = """Topic: {topic}
Gaps that need direct comparison material:
{gap_claims}

Generate {n} concise search queries (3-8 words each)."""


async def _generate_remediation_queries(
    llm, topic: str, reason_key: str, gap_claims: list[str], n: int
) -> list[str] | None:
    """Use the LLM to turn the gaps' actual claims into precise search queries.

    The deterministic seed templates only know the topic, so they recall broad
    "recent advances" style papers and pile up low-priority results. The gaps'
    claimed deltas carry the exact mechanism the audit failed to find comparisons
    for, so the LLM can anchor queries on it. Returns None on any failure so the
    caller keeps the template queries.
    """
    try:
        gen = await llm.chat_json([
            {"role": "system", "content": _REMEDIATION_QUERY_SYSTEM},
            {"role": "user", "content": _REMEDIATION_QUERY_USER.format(
                topic=topic,
                gap_claims="\n".join(f"- {c}" for c in gap_claims[:5]),
                n=n,
            )},
        ], RemediationQuerySchema)
        queries = [q.strip() for q in gen.queries if q and q.strip()]
        if queries:
            return queries[:n]
    except Exception as exc:
        logger.warning("Remediation query generation failed (%s); using templates", exc)
    return None


async def run_targeted_research_round(
    db, state: ResearchState, llm, task_id: str, reason: str,
    context: list[str] | None = None,
    service_families: list[str] | None = None,
) -> RemediationResult:
    """RunONE directed search round aimed at `reason`, then return outcome.

    `service_families` are the canonical gap family ids this round is answering
    "how much more retrieval does this gap's novelty need?" for. The round is
    charged against the task budget AND each served family's per-family budget,
    so a new family surfacing after a mining revision gets its own remediation
    chances instead of inheriting an old family's spend.

    The caller is responsible for re-running the failed gate afterwards.
    """
    key = _reason_key(reason)
    if key is None or not can_remediate(state, reason, service_families):
        return RemediationResult(False, reason, 0, 0, exhausted=True)

    intent, _templates = _REASON_PLAYBOOK[key]
    topic = state.normalized_topic or state.user_input or ""

    # Prefer LLM-generated queries that use the gaps' actual claims (context);
    # fall back to the deterministic templates when context is missing or the
    # LLM call fails.
    seeds = None
    if context and llm is not None:
        seeds = await _generate_remediation_queries(llm, topic, key, context, len(_templates))
    if not seeds:
        seeds = _build_seed_queries(key, topic)

    # A remediation round is a real search round for STORAGE purposes, but it
    # must NOT inflate `current_round` past `max_rounds` (which would surface as
    # a confusing "round 5/3" in the UI). Keep a separate counter and offset the
    # stored round_num above max_rounds so it stays unique against the primary
    # search rounds and can be filtered out of resume calibration.
    state.remediation_round += 1
    round_num = settings.max_rounds + state.remediation_round

    logger.info("Task %s: O2 targeted remediation round %d (remediation #%d) for reason='%s' (intent=%s)",
                task_id[:8], round_num, state.remediation_round, key, intent)
    task_repo.update_status(db, task_id, "searching")
    db.commit()
    emit_event(task_id, "status", {"status": "searching", "reason": f"remediation:{key}"})
    emit_event(task_id, "round_start", {"round": round_num, "remediation": True})

    # Bind the seed queries to the highest-importance active question so they
    # satisfy SearchQueryExecution's non-null target requirement.
    target_question_id = state.active_question_ids[0] if state.active_question_ids else None

    executions: list[SearchQueryExecution] = []
    for seed in seeds:
        try:
            record = save_search_query(
                db, task_id, seed, intent, target_question_id, intent, round_num
            )
            executions.append(SearchQueryExecution(
                query_id=record.id,
                query_text=seed,
                intent=intent,
                target_question_id=target_question_id or "legacy",
                expected_evidence_type=intent,
            ))
        except Exception as e:
            logger.warning("Task %s: failed to record remediation query '%s': %s",
                           task_id[:8], seed[:40], e)
    if not executions:
        state.remediation_round -= 1
        return RemediationResult(False, reason, 0, 0, exhausted=True)

    state.used_queries.extend(e.query_text for e in executions)

    new_evidence_count = 0
    try:
        papers_found, _deduped, new_paper_ids = await search_and_save_papers(
            db, state, executions, task_id, round_num
        )
        await score_papers(db, state, llm, task_id, round_num)

        if state.pipeline_version >= 2:
            try:
                ev = await extract_evidence_units(db, state, llm, task_id, round_num)
                new_evidence_count = ev if isinstance(ev, int) else 0
                await update_coverage_matrix(db, state, llm, task_id, round_num)
            except Exception as ec_err:
                logger.warning("Task %s: remediation evidence/coverage failed (non-fatal): %s",
                               task_id[:8], ec_err)

        paper_repo.save_round(
            db, task_id, round_num, [e.query_text for e in executions],
            papers_found, len(new_paper_ids),
            1.0 - (len(new_paper_ids) / max(papers_found, 1)) if papers_found else 0.0,
            f"Targeted remediation round for '{key}'", [],
        )
        new_paper_count = len(new_paper_ids)
    except Exception as e:
        logger.warning("Task %s: remediation search round failed (non-fatal): %s", task_id[:8], e)
        db.rollback()
        state = task_repo.get_state(db, task_id)
        new_paper_count = 0

    # Record the attempt (per-reason + task-global + per-family), then persist.
    attempts = dict(state.remediation_attempts or {})
    attempts[key] = int(attempts.get(key, 0)) + 1
    attempts["__total__"] = int(attempts.get("__total__", 0)) + 1
    state.remediation_attempts = attempts
    if service_families:
        family_used = dict(state.gap_remediation_used or {})
        for fam in set(service_families):
            # Charge only families that still have budget left; an already
            # exhausted family is not charged again (its gap should have been
            # finalized, not re-served by a later family's round).
            if int(family_used.get(fam, 0)) < settings.max_remediation_attempts_per_gap:
                family_used[fam] = int(family_used.get(fam, 0)) + 1
        state.gap_remediation_used = family_used
    task_repo.save_state(db, task_id, state)
    db.commit()

    emit_event(task_id, "round_done", {
        "round": round_num, "new_papers": new_paper_count,
        "remediation": True, "reason": key,
    })
    logger.info("Task %s: remediation round %d done — %d new papers, %d new evidence",
                task_id[:8], round_num, new_paper_count, new_evidence_count)

    exhausted = not can_remediate(state, reason)
    return RemediationResult(True, reason, new_paper_count, new_evidence_count, exhausted)

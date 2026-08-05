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
    "no_surviving_gap_after_audit": (
        "direct_neighbor",
        [
            "{topic} recent advances",
            "comparison methods {topic}",
            "{topic} benchmark evaluation",
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


def can_remediate(state: ResearchState, reason: str) -> bool:
    """Return True if a directed remediation round is allowed for this reason."""
    if settings.max_remediation_attempts <= 0:
        return False
    key = _reason_key(reason)
    if key is None:
        return False
    attempts = state.remediation_attempts or {}
    per_reason = int(attempts.get(key, 0))
    total = int(attempts.get("__total__", 0))
    if per_reason >= settings.max_remediation_attempts:
        return False
    if total >= settings.max_remediation_rounds_total:
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


async def run_targeted_research_round(
    db, state: ResearchState, llm, task_id: str, reason: str
) -> RemediationResult:
    """RunONE directed search round aimed at `reason`, then return outcome.

    The caller is responsible for re-running the failed gate afterwards.
    """
    key = _reason_key(reason)
    if key is None or not can_remediate(state, reason):
        return RemediationResult(False, reason, 0, 0, exhausted=True)

    intent, _templates = _REASON_PLAYBOOK[key]
    topic = state.normalized_topic or state.user_input or ""
    seeds = _build_seed_queries(key, topic)

    # Bump round counter — a remediation round is a real search round.
    state.current_round += 1
    round_num = state.current_round

    logger.info("Task %s: O2 targeted remediation round %d for reason='%s' (intent=%s)",
                task_id[:8], round_num, key, intent)
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
        state.current_round -= 1
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

    # Record the attempt (per-reason + global), then persist.
    attempts = dict(state.remediation_attempts or {})
    attempts[key] = int(attempts.get(key, 0)) + 1
    attempts["__total__"] = int(attempts.get("__total__", 0)) + 1
    state.remediation_attempts = attempts
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

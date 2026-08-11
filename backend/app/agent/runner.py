"""Agent Runner - main orchestration loop.

This module contains ONLY the orchestration logic:
- Task lifecycle management (start/stop/timeout)
- Main search loop
- Step sequencing (delegated to app.agent.steps.*)
- Idea retry loop

All step implementations live in app/agent/steps/ for testability.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass

from app.config import settings
from app.db.session import SessionLocal
from app.db.repositories import task_repo, paper_repo
from app.db.repositories import phase_repo
from app.services import phase_service
from app.agent.state import ResearchState
from app.agent.policy import should_stop, early_termination_check
from app.agent.steps import (
    clarify_topic,
    generate_queries,
    search_and_save_papers,
    score_papers,
    summarize_round,
    build_paper_clusters,
    generate_report,
    generate_and_score_ideas,
    generate_experiments,
    build_research_contract,
    decompose_research_space,
    extract_evidence_units,
    update_coverage_matrix,
    mine_gap_candidates,
    audit_gap_candidates,
    generate_interventions,
    generate_minimal_experiments,
    generate_landscape_brief,
    run_targeted_research_round,
    can_remediate,
)
from app.agent.steps.analyze_papers import analyze_papers
from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
from app.llm.factory import get_llm
from app.llm.base import LLMBudgetExceeded
from app.services.event_service import emit_event, emit_event_with_cleanup

logger = logging.getLogger(__name__)


class RoundEvidenceCoverageError(Exception):
    """Raised when evidence extraction or coverage update fails in a round."""


@dataclass
class SearchLoopResult:
    """Result of the search loop — determines if downstream can proceed."""
    status: str  # completed / stopped_normally / failed / more_research_required
    reason: str
    completed_rounds: int
    failed_attempts: int


@dataclass
class RoundSearchResult:
    """Formal result of a single search round — returned by _search_op()."""
    query_ids: list[str]
    query_texts: list[str]
    papers_found: int
    deduped_count: int
    new_paper_ids: list[str]
    duplicate_rate: float
    high_priority_before: int
    high_priority_after: int
    new_high_priority_count: int
    round_summary: str
    knowledge_gaps: list[str]

    def to_phase_payload(self) -> dict:
        """Serialize to a stable dict for PhaseRun.output_json."""
        return dataclasses.asdict(self)

    @classmethod
    def from_phase_payload(cls, payload: dict) -> "RoundSearchResult":
        """Deserialize from PhaseRun.output_json payload.

        Raises TypeError if required fields are missing or wrong type.
        """
        expected = {
            "query_ids", "query_texts", "papers_found", "deduped_count",
            "new_paper_ids", "duplicate_rate", "high_priority_before",
            "high_priority_after", "new_high_priority_count",
            "round_summary", "knowledge_gaps",
        }
        missing = expected - set(payload.keys())
        if missing:
            raise TypeError(
                f"RoundSearchResult.from_phase_payload: missing fields: {missing}"
            )
        return cls(
            query_ids=list(payload["query_ids"]),
            query_texts=list(payload["query_texts"]),
            papers_found=int(payload["papers_found"]),
            deduped_count=int(payload["deduped_count"]),
            new_paper_ids=list(payload["new_paper_ids"]),
            duplicate_rate=float(payload["duplicate_rate"]),
            high_priority_before=int(payload["high_priority_before"]),
            high_priority_after=int(payload["high_priority_after"]),
            new_high_priority_count=int(payload["new_high_priority_count"]),
            round_summary=str(payload["round_summary"]),
            knowledge_gaps=list(payload["knowledge_gaps"]),
        )


@dataclass
class RoundEvidenceCoverageResult:
    """Phase 2.2A Closure (#7): Per-round evidence/coverage result."""
    evidence_status: str   # completed / failed / partial
    coverage_status: str   # completed / failed
    new_evidence_count: int
    coverage_snapshot_count: int
    reason: str

# Task registry for asyncio tasks (P0-1: protected by lock)
_task_registry: dict[str, asyncio.Task] = {}
_registry_lock = asyncio.Lock()

# P0-1: global concurrency limiter — prevents SQLite write contention
# when multiple agents run simultaneously. Configurable via max_concurrent_agents.
_agent_semaphore: asyncio.Semaphore | None = None


def _get_agent_semaphore() -> asyncio.Semaphore:
    """Lazy-init the global agent concurrency semaphore."""
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(settings.max_concurrent_agents)
    return _agent_semaphore

# Statuses that indicate a task was interrupted (process crash / restart)
_INTERRUPTED_STATUSES = {
    "clarifying",
    "building_contract",
    "decomposing",
    "searching",
    "extracting_evidence",
    "updating_coverage",
    "summarizing",
    "analyzing_papers",
    "building_wiki",
    "building_clusters",
    "reporting",
    "generating_ideas",
    "judging_ideas",
    "generating_experiment",
    # Phase 1.5: New phases
    "mining_gaps",
    "auditing_gaps",
    "checking_feasibility",
    "synthesizing_ideas",
}


def start_agent(task_id: str) -> bool:
    """Start the agent loop as an asyncio background task with timeout protection.

    Returns True if the agent was started (or already running), False if
    rejected due to capacity limit.

    P0-1 / P1-11: Capacity check + registration are done atomically in a single
    synchronous block (no await between check and register), so concurrent
    start_agent() calls cannot both pass the capacity check. This works because
    we're in a single-threaded event loop — synchronous code runs to completion
    without yielding.

    Uses a global semaphore to limit concurrent agents (prevents SQLite write
    contention).
    """
    # Atomic check-and-register: no await between checking capacity and
    # inserting into the registry, so concurrent callers see consistent state.
    if task_id in _task_registry and not _task_registry[task_id].done():
        return True  # already running

    # Check capacity synchronously (no await → no interleaving)
    running = sum(1 for t in _task_registry.values() if not t.done())
    if running >= settings.max_concurrent_agents:
        logger.warning("Task %s: max concurrent agents (%d) reached, rejecting start",
                       task_id[:8], settings.max_concurrent_agents)
        return False  # rejected — caller should return 429

    AGENT_TIMEOUT = settings.agent_timeout_seconds

    async def _run():
        sem = _get_agent_semaphore()
        async with sem:
            try:
                await asyncio.wait_for(run_task(task_id), timeout=AGENT_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error("Agent task %s timed out after %d seconds", task_id, AGENT_TIMEOUT)
                await _safe_mark_failed(task_id, f"Agent timed out after {AGENT_TIMEOUT}s",
                                        f"Agent 超时（{AGENT_TIMEOUT}秒），可能因 API 限流或 PDF 下载阻塞")
            except asyncio.CancelledError:
                logger.info("Agent task %s was cancelled", task_id)
                await _safe_mark_failed(task_id, "Task cancelled by user", "用户手动停止", cleanup=True)
            except Exception as e:
                logger.exception("Agent task %s failed", task_id)
                await _safe_mark_failed(task_id, str(e)[:500], str(e))
            finally:
                async with _registry_lock:
                    _task_registry.pop(task_id, None)

    # Create the task and register it SYNCHRONOUSLY (before any await)
    # so that subsequent start_agent() calls see this task in the registry.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    task = loop.create_task(_run())
    # Synchronous registration — no await, no race window.
    # _registry_lock is only needed for async cleanup in _run()'s finally block;
    # here we're in sync context and won't be preempted by another start_agent()
    # call (single-threaded event loop, no await between create_task and assign).
    _task_registry[task_id] = task
    return True


async def _safe_mark_failed(task_id: str, reason: str, event_message: str, cleanup: bool = False):
    """Mark a task as failed with retry (3 attempts).

    Args:
        cleanup: If True, clean up the SSE event queue after marking failed.
    """
    from app.services.event_service import cleanup_task_events
    for attempt in range(3):
        try:
            db = SessionLocal()
            try:
                task_repo.update_status(db, task_id, "failed")
                task_repo.update_stop_reason(db, task_id, reason)
                db.commit()
                emit_event_with_cleanup(task_id, "error", {"message": event_message})
            finally:
                db.close()
            break
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
            else:
                logger.error("Failed to update task status after 3 retries")
    if cleanup:
        cleanup_task_events(task_id)


def stop_agent(task_id: str):
    """Stop a running agent task."""
    # P0-1: cancel synchronously (the lock is for async context, but
    # cancel() itself is safe to call from sync code)
    task = _task_registry.get(task_id)
    if task:
        task.cancel()
        # Removal will happen in the _run() finally block via _registry_lock


def recover_interrupted_tasks():
    """On startup, recover tasks with interrupted statuses (P0-2: smart resume).

    Phase 1.5: Also marks any running PhaseRun records as interrupted.
    """
    db = SessionLocal()
    try:
        from app.db.models import ResearchTask, ResearchRound
        tasks = db.query(ResearchTask).filter(
            ResearchTask.status.in_(list(_INTERRUPTED_STATUSES))
        ).all()
        for task in tasks:
            logger.warning("Task %s was interrupted (status=%s), resetting to pending for resume",
                          task.id[:8], task.status)
            task.status = "pending"
            task.stop_reason = "interrupted_by_restart"

            # Phase 1.5: Mark any running PhaseRuns as interrupted
            phase_repo.mark_interrupted_phases(db, task.id)

            # Calibrate state.current_round
            if task.state_json:
                try:
                    state = ResearchState.from_json(task.state_json)
                    max_completed = db.query(ResearchRound.round_number).filter(
                        ResearchRound.task_id == task.id
                    ).order_by(ResearchRound.round_number.desc()).first()
                    actual_rounds = max_completed[0] if max_completed else 0

                    if state.current_round != actual_rounds:
                        logger.info(
                            "Task %s: calibrating current_round %d -> %d (from research_rounds table)",
                            task.id[:8], state.current_round, actual_rounds,
                        )
                        state.current_round = actual_rounds
                        task.state_json = state.to_json()
                except Exception as state_err:
                    logger.warning("Task %s: state calibration failed (non-fatal): %s",
                                  task.id[:8], state_err)

        db.commit()
        if tasks:
            logger.info("Recovered %d interrupted tasks (reset to pending for resume)", len(tasks))
    except Exception as e:
        logger.error("Failed to recover interrupted tasks: %s", e)
        db.rollback()
    finally:
        db.close()


async def run_task(task_id: str):
    """Main agent loop.

    DB Session strategy: main session for search/ideas loop (needs state continuity),
    but separate sessions for RAG/Wiki phases (long-running, don't need main state).

    P0-2: Supports resume — if state.current_round > 0 and papers already collected,
    the search loop will skip already-completed rounds (should_stop checks high_priority
    count which is preserved in state). The clarify phase is skipped if normalized_topic
    is already set.
    """
    db = SessionLocal()
    try:
        state = task_repo.get_state(db, task_id)
        llm = get_llm()
        # A resumed task still carries the stop_reason of the run that ended
        # (e.g. "interrupted_by_restart", or a previous terminal reason).
        # Leaving it in place makes the API/UI report a stale terminal reason
        # while the task is in fact running again.
        task_repo.update_stop_reason(db, task_id, "")
        db.commit()
        # High-priority #3: enforce a per-task LLM budget so cost is bounded and
        # a runaway task degrades gracefully instead of hard-failing/timing out.
        if hasattr(llm, "set_budget"):
            llm.set_budget(
                max_calls=settings.max_llm_calls_per_task,
                max_total_tokens=settings.max_llm_tokens_per_task,
            )

        # === 1. Topic clarification (skip if already clarified) ===
        if state.normalized_topic:
            logger.info("Task %s: resuming — topic already clarified: %s",
                       task_id[:8], state.normalized_topic[:50])
        else:
            await _run_clarification_phase(db, state, llm, task_id)
            # Re-check: clarification may have set waiting_for_clarification
            # Phase 0 fix: removed `db.refresh = None` — must not override Session methods
            state = task_repo.get_state(db, task_id)
            if not state.normalized_topic:
                return  # waiting for user clarification

        # === 2. Build Research Contract (Phase 1, with PhaseRun) ===
        logger.info("Task %s: building research contract...", task_id[:8])
        task_repo.update_status(db, task_id, "building_contract")
        db.commit()
        emit_event(task_id, "status", {"status": "building_contract"})
        try:
            # Phase 2.2A: Use compute_contract_input_version for stable SHA-256
            from app.agent.steps.build_contract import compute_contract_input_version
            from app.db.models import ResearchTask as _RT
            task_obj = db.get(_RT, task_id)
            contract_input_version = compute_contract_input_version(db, task_obj, state) if task_obj else ""

            async def _build_contract_op(db):
                return await build_research_contract(db, state, llm, task_id)
            await phase_service.execute_phase(db, task_id, "build_contract", _build_contract_op,
                                               input_version=contract_input_version)
            state = task_repo.get_state(db, task_id)
        except Exception as e:
            logger.warning("Task %s: contract building failed (non-fatal, using fallback): %s",
                          task_id[:8], e)

        # === 2b. Decompose Research Space (Phase 1, with PhaseRun) ===
        logger.info("Task %s: decomposing research space...", task_id[:8])
        task_repo.update_status(db, task_id, "decomposing")
        db.commit()
        emit_event(task_id, "status", {"status": "decomposing"})
        try:
            async def _decompose_op(db):
                return await decompose_research_space(db, state, llm, task_id)
            await phase_service.execute_phase(db, task_id, "decompose", _decompose_op,
                                              input_version=state.contract_id or "")
            state = task_repo.get_state(db, task_id)
        except Exception as e:
            logger.warning("Task %s: decomposition failed (non-fatal, continuing): %s",
                          task_id[:8], e)

        # === 3. Search loop ===
        search_result = await _run_search_loop(db, state, llm, task_id)

        # Phase 1.5: Check search result before proceeding
        if search_result.status == "failed":
            logger.error("Task %s: search loop failed (%s), stopping", task_id[:8], search_result.reason)
            task_repo.update_status(db, task_id, "failed")
            task_repo.update_stop_reason(db, task_id, search_result.reason)
            emit_event(task_id, "error", {"message": f"Search failed: {search_result.reason}"})
            db.commit()
            return

        if search_result.status == "more_research_required":
            logger.warning("Task %s: search needs more research (%s)", task_id[:8], search_result.reason)
            task_repo.update_status(db, task_id, "more_research_required")
            task_repo.update_stop_reason(db, task_id, search_result.reason)
            emit_event(task_id, "status", {"status": "more_research_required", "reason": search_result.reason})
            db.commit()
            return

        # Only completed or stopped_normally can proceed
        logger.info("Task %s: search completed (%s, %d rounds), proceeding to analysis",
                    task_id[:8], search_result.status, search_result.completed_rounds)

        # Close main session before long-running RAG/Wiki phases
        db.close()
        db = SessionLocal()
        state = task_repo.get_state(db, task_id)

        # Phase 2.2A Final Closure (#4): Formal readiness gate replaces
        # the ad-hoc evidence_count > 0 check.
        # O2: wrapped in a loop so a'more_research_required' verdict can trigger
        # one directed remediation search round and re-evaluate, instead of
        # terminating immediately.
        from app.agent.steps.readiness_gate import evaluate_phase2_readiness
        while True:
            readiness = evaluate_phase2_readiness(db, task_id, state.contract_id)
            logger.info("Task %s: readiness gate — status=%s, reason=%s, "
                        "active_q=%d, high_imp_q=%d, high_imp_covered=%d, "
                        "evidence=%d, latest_round=%d",
                        task_id[:8], readiness.status, readiness.reason,
                        readiness.active_question_count,
                        readiness.high_importance_question_count,
                        readiness.high_importance_covered_count,
                        readiness.valid_evidence_count, readiness.latest_round)

            # Save readiness result to AgentTrace for auditability
            try:
                from app.db.models import AgentTrace
                import json as _trace_json
                trace = AgentTrace(
                    task_id=task_id,
                    step_name="phase2_readiness_gate",
                    step_type="decision",
                    input_json=_trace_json.dumps({"contract_id": state.contract_id or ""},
                                                 ensure_ascii=False),
                    output_json=_trace_json.dumps({
                        "ready": readiness.ready,
                        "status": readiness.status,
                        "reason": readiness.reason,
                        "active_question_count": readiness.active_question_count,
                        "questions_with_latest_snapshot": readiness.questions_with_latest_snapshot,
                        "high_importance_question_count": readiness.high_importance_question_count,
                        "high_importance_covered_count": readiness.high_importance_covered_count,
                        "evidence_count": readiness.evidence_count,
                        "valid_evidence_count": readiness.valid_evidence_count,
                        "latest_round": readiness.latest_round,
                        "missing_question_ids": readiness.missing_question_ids,
                        "unresolved_question_ids": readiness.unresolved_question_ids,
                    }, ensure_ascii=False),
                )
                db.add(trace)
                db.commit()
            except Exception as trace_err:
                logger.warning("Task %s: failed to save readiness trace (non-fatal): %s",
                              task_id[:8], trace_err)

            if readiness.status == "failed":
                logger.error("Task %s: readiness gate FAILED — %s", task_id[:8], readiness.reason)
                # Even a hard control-plane failure should hand the user
                # whatever was actually collected (papers, evidence, coverage)
                # instead of dying silently. Best-effort: never let brief
                # generation mask the real failure reason.
                try:
                    await generate_landscape_brief(
                        db, state, task_id, "failed", f"readiness_failed: {readiness.reason}")
                except Exception as brief_err:
                    logger.warning("Task %s: landscape brief on readiness failure "
                                   "could not be generated (non-fatal): %s",
                                   task_id[:8], brief_err)
                task_repo.update_status(db, task_id, "failed")
                task_repo.update_stop_reason(db, task_id, f"readiness_failed: {readiness.reason}")
                emit_event(task_id, "error", {
                    "message": f"Phase 2 readiness gate failed: {readiness.reason}",
                    "missing_questions": readiness.missing_question_ids,
                })
                db.commit()
                return

            if readiness.status == "more_research_required":
                logger.warning("Task %s: readiness gate — more research required — %s",
                              task_id[:8], readiness.reason)
                if await _try_remediate(db, state, llm, task_id, "readiness_more_research"):
                    state = task_repo.get_state(db, task_id)
                    continue  # re-evaluate readiness after directed search
                await _terminate_more_research(
                    db, state, task_id, "more_research_required",
                    f"readiness_more_research: {readiness.reason}")
                return

            break  # readiness ready — exit the remediation loop

        logger.info("Task %s: readiness gate PASSED — proceeding to analysis", task_id[:8])

        # === 2.5. RAG: Download PDFs and index high-priority papers ===
        # O5a: RAG indexing re-enabled via the pluggable embedding backend
        # (API embedding, no local PyTorch → stable on Windows). Falls back to
        # abstract-only if the embedding backend is unavailable.
        from app.services.embedding_service import embedding_enabled
        if settings.enable_rag_indexing and embedding_enabled():
            logger.info("Task %s: RAG indexing enabled (embedding backend=%s)",
                        task_id[:8], settings.embedding_backend)
            await _run_rag_indexing(db, task_id, llm)
        else:
            _skip_reason = ("disabled via enable_rag_indexing"
                            if not settings.enable_rag_indexing
                            else "embedding backend unavailable")
            logger.info("Task %s: skipping RAG indexing (%s, using abstract fallback)",
                        task_id[:8], _skip_reason)
            emit_event(task_id, "status", {"status": "indexing_skipped", "reason": _skip_reason})

        # === 2.5b. Paper Deep Analysis (新增：论文深度分析) ===
        # Download PDFs for high-priority papers, extract text, LLM structured analysis
        # Results stored in paper_analyses table, used by wiki/report/ideas.
        # Also gated by enable_rag_indexing because it does the same PDF parsing
        # that can native-segfault on Windows; gap mining relies on Evidence Units,
        # not on this deep analysis, so skipping it is safe for the V2 pipeline.
        if settings.enable_rag_indexing:
            logger.info("Task %s: starting paper deep analysis...", task_id[:8])
            task_repo.update_status(db, task_id, "analyzing_papers")
            db.commit()
            emit_event(task_id, "status", {"status": "analyzing_papers"})
            try:
                await analyze_papers(db, state, llm, task_id)
            except Exception as e:
                logger.warning("Task %s: paper analysis failed (non-fatal, continuing): %s", task_id[:8], e)
        else:
            logger.info("Task %s: skipping paper deep analysis (enable_rag_indexing=False)", task_id[:8])

        # Wiki, clustering, and reports are optional presentation features for Pipeline V2.
        # They must not determine whether an evidence-grounded opportunity is found.
        if state.pipeline_version < 2:
            await _run_wiki_ingest(db, task_id, llm)
            logger.info("Task %s: building clusters...", task_id[:8])
            cluster_list = await build_paper_clusters(db, state, llm, task_id)
            logger.info("Task %s: generating report...", task_id[:8])
            task_repo.update_status(db, task_id, "reporting")
            db.commit()
            emit_event(task_id, "status", {"status": "reporting"})
            await generate_report(db, state, llm, cluster_list)

        # === 5. Lightweight evidence-grounded opportunity discovery ===
        if state.pipeline_version >= 2:
            await _run_opportunity_pipeline(db, state, llm, task_id)
        else:
            # Legacy pipeline: generate ideas directly
            await _run_ideas_loop(db, state, llm, task_id, cluster_list)

    except LLMBudgetExceeded as budget_err:
        # High-priority #3: graceful degradation — emit a landscape brief with
        # whatever evidence exists rather than hard-failing on budget overrun.
        logger.warning("Task %s: LLM budget exceeded, degrading gracefully: %s",
                       task_id[:8], budget_err)
        try:
            db.rollback()
            state = task_repo.get_state(db, task_id)
            await _terminate_more_research(
                db, state, task_id, "more_research_required", "llm_budget_exceeded")
        except Exception as deg_err:
            logger.error("Task %s: degradation after budget overrun failed: %s",
                         task_id[:8], deg_err)
            task_repo.update_status(db, task_id, "failed")
            task_repo.update_stop_reason(db, task_id, "llm_budget_exceeded")
            db.commit()
    finally:
        db.close()


async def _terminate_more_research(db, state: ResearchState, task_id: str,
                                    status: str, reason: str):
    """Emit landscape brief + set a terminal 'more research' style status.

    Centralizes the 6 former inline termination exits so O2 remediation logic
    lives in one place.
    """
    await generate_landscape_brief(db, state, task_id, status, reason)
    task_repo.update_status(db, task_id, status)
    task_repo.update_stop_reason(db, task_id, reason)
    payload = {"status": status, "reason": reason}
    emit_event(task_id, "status", payload)
    db.commit()


async def _run_opportunity_pipeline(db, state: ResearchState, llm, task_id: str):
    """Pipeline V2 opportunity discovery with O2 targeted remediation.

    Sequence: mine_gaps -> audit_gaps -> interventions -> minimal experiments.
    On each blocking failure, if the reason is remediable and budget remains,
    runONE directed search round and retry the whole opportunity pipeline
    instead of terminating. Bounded by settings.max_remediation_rounds_total.
    """
    from app.db.repositories import gap_repo
    from app.db.models import GapCandidate, ResearchIdea

    # The opportunity pipeline is attempted; on a remediable stall we run a
    # directed search round and loop again. The loop count is bounded by the
    # global remediation budget (can_remediate checks the counters).
    max_pipeline_iters = 1 + settings.max_remediation_rounds_total
    for _iter in range(max_pipeline_iters):
        state = task_repo.get_state(db, task_id)

        # --- Gap mining ---
        logger.info("Task %s: mining evidence-backed gap candidates...", task_id[:8])
        task_repo.update_status(db, task_id, "mining_gaps")
        db.commit()
        emit_event(task_id, "status", {"status": "mining_gaps"})
        gap_input_version = hashlib.sha256(json.dumps({
            "contract_id": state.contract_id,
            "round": state.current_round,
            "pipeline_version": state.pipeline_version,
            "gap_mining_policy_version": GAP_MINING_POLICY_VERSION,
        }, sort_keys=True).encode()).hexdigest()

        async def _mine_gaps_op(db):
            return await mine_gap_candidates(db, state, llm, task_id)

        gaps = await phase_service.execute_phase(
            db, task_id, "mine_gaps", _mine_gaps_op, input_version=gap_input_version
        )
        if gaps is None:
            gaps = [gap for gap in gap_repo.list_gaps_for_contract(db, task_id, state.contract_id)
                    if gap.mining_policy_version == GAP_MINING_POLICY_VERSION]
        state = task_repo.get_state(db, task_id)
        if not gaps:
            if await _try_remediate(db, state, llm, task_id, "no_evidence_backed_gap_candidates"):
                continue
            await _terminate_more_research(db, state, task_id,
                                           "more_research_required", "no_evidence_backed_gap_candidates")
            return

        # --- Gap audit ---
        task_repo.update_status(db, task_id, "auditing_gaps")
        db.commit()
        emit_event(task_id, "status", {"status": "auditing_gaps", "gap_count": len(gaps)})

        async def _audit_gaps_op(db):
            return await audit_gap_candidates(db, state, llm, task_id, gap_ids=[gap.id for gap in gaps])

        audit_input_version = hashlib.sha256(json.dumps({
            "gap_ids": sorted(gap.id for gap in gaps),
            "round": state.current_round,
            "pipeline_version": state.pipeline_version,
            "gap_mining_policy_version": GAP_MINING_POLICY_VERSION,
        }, sort_keys=True).encode()).hexdigest()
        await phase_service.execute_phase(
            db, task_id, "audit_gaps", _audit_gaps_op, input_version=audit_input_version
        )
        current_gap_ids = [gap.id for gap in gaps]
        state.surviving_gap_ids = [gap.id for gap in db.query(GapCandidate).filter(
            GapCandidate.task_id == task_id,
            GapCandidate.contract_id == state.contract_id,
            GapCandidate.id.in_(current_gap_ids),
            GapCandidate.mining_policy_version == GAP_MINING_POLICY_VERSION,
            GapCandidate.status == "surviving",
        ).all()]
        task_repo.save_state(db, task_id, state)
        db.commit()
        state = task_repo.get_state(db, task_id)
        if not state.surviving_gap_ids:
            if await _try_remediate(db, state, llm, task_id, "no_surviving_gap_after_audit"):
                continue
            await _terminate_more_research(db, state, task_id,
                                           "more_research_required", "no_surviving_gap_after_audit")
            return

        # --- Interventions ---
        task_repo.update_status(db, task_id, "synthesizing_ideas")
        db.commit()
        emit_event(task_id, "status", {"status": "synthesizing_ideas"})

        async def _generate_interventions_op(db):
            return await generate_interventions(db, state, llm, task_id)

        intervention_input_version = hashlib.sha256(json.dumps({
            "surviving_gap_ids": sorted(state.surviving_gap_ids),
            "contract_id": state.contract_id,
            "pipeline_version": state.pipeline_version,
            "gap_mining_policy_version": GAP_MINING_POLICY_VERSION,
        }, sort_keys=True).encode()).hexdigest()
        interventions = await phase_service.execute_phase(
            db, task_id, "generate_interventions", _generate_interventions_op,
            input_version=intervention_input_version
        )
        if interventions is None:
            from app.db.repositories import intervention_repo
            recovered_interventions = intervention_repo.list_interventions_for_task(
                db, task_id, contract_id=state.contract_id, gap_ids=state.surviving_gap_ids
            )
            passed_intervention_ids = [item.id for item in recovered_interventions if item.status == "passed"]
        else:
            passed_intervention_ids = interventions.passed_intervention_ids
        if not passed_intervention_ids:
            if await _try_remediate(db, state, llm, task_id, "no_intervention_passed_hard_gates"):
                continue
            await _terminate_more_research(db, state, task_id,
                                           "more_research_required", "no_intervention_passed_hard_gates")
            return

        # --- Minimal experiments (final gate — no remediation past this point) ---
        task_repo.update_status(db, task_id, "generating_experiment")
        db.commit()
        emit_event(task_id, "status", {"status": "generating_experiment"})

        async def _minimal_experiments_op(db):
            return await generate_minimal_experiments(db, state, llm, task_id)

        experiment_input_version = hashlib.sha256(json.dumps({
            "intervention_ids": sorted(passed_intervention_ids),
            "contract_id": state.contract_id,
            "pipeline_version": state.pipeline_version,
            "gap_mining_policy_version": GAP_MINING_POLICY_VERSION,
        }, sort_keys=True).encode()).hexdigest()
        experiment_result = await phase_service.execute_phase(
            db, task_id, "generate_minimal_experiments", _minimal_experiments_op,
            input_version=experiment_input_version
        )
        if experiment_result is None:
            idea_ids = [idea.id for idea in db.query(ResearchIdea).filter(
                ResearchIdea.task_id == task_id,
                ResearchIdea.contract_id == state.contract_id,
                ResearchIdea.intervention_id.in_(passed_intervention_ids),
                ResearchIdea.pipeline_version == state.pipeline_version,
                ResearchIdea.decision == "conditional_go",
                ResearchIdea.idea_status == "active",
            ).all()]
        else:
            idea_ids = experiment_result.idea_ids
        if not idea_ids:
            await _terminate_more_research(db, state, task_id,
                                           "abstained", "no_minimal_experiment_generated")
            return

        # --- Success ---
        await generate_landscape_brief(db, state, task_id,
                                       "waiting_for_user_review", "evidence_grounded_ideas_ready")
        task_repo.update_status(db, task_id, "waiting_for_user_review")
        task_repo.update_stop_reason(db, task_id, "evidence_grounded_ideas_ready")
        emit_event(task_id, "status", {
            "status": "waiting_for_user_review",
            "reason": "evidence_grounded_ideas_ready",
            "idea_count": len(idea_ids),
        })
        db.commit()
        return

    # Budget exhausted without producing ideas — emit brief and stop.
    state = task_repo.get_state(db, task_id)
    await _terminate_more_research(db, state, task_id,
                                   "more_research_required", "remediation_budget_exhausted")


async def _try_remediate(db, state: ResearchState, llm, task_id: str, reason: str) -> bool:
    """Attempt one O2 directed remediation round for `reason`.

    Returns True if a remediation round ran (caller should retry the pipeline),
    False if remediation is not allowed/exhausted (caller should terminate).
    """
    if not can_remediate(state, reason):
        logger.info("Task %s: no remediation for '%s' (disabled/exhausted)", task_id[:8], reason)
        return False
    logger.info("Task %s: O2 remediation triggered for '%s'", task_id[:8], reason)
    result = await run_targeted_research_round(db, state, llm, task_id, reason)
    if not result.attempted:
        return False
    logger.info("Task %s: remediation added %d papers, %d evidence (exhausted=%s)",
                task_id[:8], result.new_paper_count, result.new_evidence_count, result.exhausted)
    return True


async def run_experiment_generation(task_id: str, idea_ids: list[str]):
    """Generate experiment plans for selected ideas."""
    db = SessionLocal()
    try:
        state = task_repo.get_state(db, task_id)
        llm = get_llm()

        task_repo.update_status(db, task_id, "judging_ideas")
        emit_event(task_id, "status", {"status": "judging_ideas"})

        result = await generate_experiments(db, state, llm, task_id, idea_ids)

        if result.get("status") == "need_more_research":
            task_repo.update_status(db, task_id, "waiting_for_user_review")
            emit_event(task_id, "status", {"status": "waiting_for_user_review", "reason": "no_idea_ready"})
            db.commit()
            return result

        task_repo.update_status(db, task_id, "done")
        emit_event_with_cleanup(task_id, "status", {"status": "done"})
        db.commit()
        return result

    finally:
        db.close()


# === Phase implementations ===

async def _run_clarification_phase(db, state: ResearchState, llm, task_id: str):
    """Phase 1: Topic clarification.

    Phase 2.2A Final Closure (#6): Adds clarify PhaseRun for control-plane
    consistency. Clarify input_version includes user_input hash + clarification
    feedback hash + pipeline_version.
    """
    task_repo.update_status(db, task_id, "clarifying")
    emit_event(task_id, "status", {"status": "clarifying"})

    # Phase 0 fix: No longer truncate user_input at "Clarifications:".
    # The clarification answers are already stored in state.user_input
    # (appended by the /clarify API endpoint) and will be processed by
    # build_research_contract (Phase 1). For now, use the full user_input
    # as normalized_topic if it contains clarifications.
    if "\nClarifications:" in state.user_input and not state.normalized_topic:
        # Use the full user_input (with clarifications) as the topic
        state.normalized_topic = state.user_input.split("\nClarifications:")[0].strip()
        state.keywords = []
        task_repo.update_normalized_topic(db, task_id, state.normalized_topic)
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "status", {"status": "clarified", "topic": state.normalized_topic})
        logger.info("Clarified (with answers): %s", state.normalized_topic[:80])
        return

    # Compute clarify input_version for PhaseRun
    clarify_input_version = hashlib.sha256(json.dumps({
        "user_input": state.user_input,
        "clarification_questions": state.clarification_questions,
        "pipeline_version": state.pipeline_version,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    async def _clarify_op(db):
        clarity = await clarify_topic(db, state, llm)
        if not clarity.is_clear:
            return {
                "status": "waiting_for_clarification",
                "questions": clarity.questions,
            }
        return {
            "status": "clarified",
            "normalized_topic": clarity.normalized_topic or state.user_input,
            "keywords": clarity.keywords,
        }

    try:
        clarity_result = await phase_service.execute_phase(
            db, task_id, "clarify", _clarify_op,
            input_version=clarify_input_version,
        )
    except Exception as e:
        logger.warning("Task %s: clarify phase failed (non-fatal, using fallback): %s",
                      task_id[:8], e)
        clarity_result = None

    if clarity_result and clarity_result.get("status") == "waiting_for_clarification":
        questions = clarity_result["questions"]
        state.clarification_questions = questions
        task_repo.save_state(db, task_id, state)
        task_repo.update_status(db, task_id, "waiting_for_clarification")
        emit_event(task_id, "clarification_needed", {"questions": questions})
        db.commit()
        return  # Wait for user to answer

    if clarity_result and clarity_result.get("status") == "clarified":
        state.normalized_topic = clarity_result.get("normalized_topic", state.user_input)
        state.keywords = clarity_result.get("keywords", [])
    else:
        # Fallback: call clarify_topic directly (no PhaseRun)
        clarity = await clarify_topic(db, state, llm)
        if not clarity.is_clear:
            state.clarification_questions = clarity.questions
            task_repo.save_state(db, task_id, state)
            task_repo.update_status(db, task_id, "waiting_for_clarification")
            emit_event(task_id, "clarification_needed", {"questions": clarity.questions})
            db.commit()
            return

        state.normalized_topic = clarity.normalized_topic or state.user_input
        state.keywords = clarity.keywords

    task_repo.update_normalized_topic(db, task_id, state.normalized_topic)
    task_repo.save_state(db, task_id, state)
    db.commit()


async def _run_search_loop(db, state: ResearchState, llm, task_id: str) -> SearchLoopResult:
    """Phase 2: Multi-round search loop.

    Phase 2.2A Final Runtime Closure:
    - RoundSearchResult as formal return from _search_op
    - State persisted after round increment and after search op
    - Stable round input version computed once before retry loop
    - Skipped phase recovery from DB
    - Separate PhaseRun for summarize_round_N
    """
    task_repo.update_status(db, task_id, "searching")
    db.commit()
    emit_event(task_id, "status", {"status": "searching", "topic": state.normalized_topic})

    no_new_high_priority_count = 0
    MAX_ATTEMPTS_PER_ROUND = 3
    TOTAL_FAILED_ROUND_BUDGET = 5
    total_failed_rounds = 0
    last_error_signature = None
    identical_error_streak = 0
    completed_rounds = 0

    while True:
        stop, reason = should_stop(state)
        if stop:
            state.stop_reason = reason
            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "stopping", {"reason": reason})
            logger.info("Stopping search: %s", reason)
            return SearchLoopResult(status="stopped_normally", reason=reason,
                                    completed_rounds=completed_rounds, failed_attempts=total_failed_rounds)

        # (#2) Persist round number immediately after increment
        state.current_round += 1
        round_num = state.current_round
        task_repo.save_state(db, task_id, state)
        db.commit()

        logger.info("=== Task %s: Round %d ===", task_id[:8], round_num)
        emit_event(task_id, "round_start", {"round": round_num})

        # (#4) Compute stable round input version ONCE before retry loop
        # Must NOT include dynamic outputs (used_queries, collected_paper_ids, evidence count)
        from app.db.models import ResearchContract as _RC_model
        active_contract = db.get(_RC_model, state.contract_id) if state.contract_id else None
        contract_input_hash = active_contract.input_hash if active_contract else ""

        # Get previous coverage version for stability
        from app.db.models import CoverageRecord as _CR_prev
        prev_cov_records = db.query(_CR_prev).filter(
            _CR_prev.task_id == task_id,
            _CR_prev.round_number < round_num,
        ).order_by(_CR_prev.round_number.desc()).limit(50).all()
        previous_coverage_version = hashlib.sha256(
            json.dumps(sorted([(cr.question_id, cr.round_number, round(cr.coverage_score, 4))
                               for cr in prev_cov_records]), ensure_ascii=False).encode()
        ).hexdigest() if prev_cov_records else "none"

        round_input_version = hashlib.sha256(json.dumps({
            "contract_input_hash": contract_input_hash,
            "active_question_ids": sorted(state.active_question_ids),
            "round_number": round_num,
            "previous_coverage_version": previous_coverage_version,
            "pipeline_version": state.pipeline_version,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

        round_attempts = 0
        round_succeeded = False

        while round_attempts < MAX_ATTEMPTS_PER_ROUND:
            round_attempts += 1
            try:
                # (#3) Record high_priority BEFORE scoring
                high_priority_before = len(state.high_priority_paper_ids)

                # (#5) PhaseRun for search_round_N
                search_phase_name = f"search_round_{round_num}"

                async def _search_op(db):
                    # Generate queries
                    query_executions = await generate_queries(db, state, llm)

                    query_payloads = [dataclasses.asdict(q) for q in query_executions]
                    emit_event(task_id, "queries_generated", {
                        "round": round_num,
                        "queries": [q.query_text for q in query_executions],
                        "structured_queries": query_payloads,
                    })

                    # Search + dedup + save
                    papers_found, deduped_count, new_paper_ids = await search_and_save_papers(
                        db, state, query_executions, task_id, round_num
                    )

                    # Score papers
                    scored = await score_papers(db, state, llm, task_id, round_num)

                    # (#3) Compute high_priority AFTER scoring
                    high_priority_after = len(state.high_priority_paper_ids)
                    new_high = high_priority_after - high_priority_before

                    # (#9) Separate PhaseRun for summarize_round_N
                    summarize_phase_name = f"summarize_round_{round_num}"
                    summarize_input_version = hashlib.sha256(json.dumps({
                        "round": round_num,
                        "papers_found": papers_found,
                        "new_papers": len(new_paper_ids),
                        "scored_count": len(scored),
                    }, sort_keys=True).encode()).hexdigest()

                    async def _summarize_op(db):
                        return await summarize_round(db, state, llm, round_num, scored)

                    round_summary = ""
                    gaps = []
                    try:
                        sum_result = await phase_service.execute_phase(
                            db, task_id, summarize_phase_name, _summarize_op,
                            input_version=summarize_input_version,
                            round_number=round_num,
                        )
                        if sum_result is not None:
                            round_summary, gaps = sum_result
                    except Exception as sum_err:
                        logger.warning("Task %s: summarize failed (non-fatal): %s", task_id[:8], sum_err)
                        round_summary = f"Round {round_num} summary (fallback)"
                        gaps = []

                    state.knowledge_gaps = gaps
                    state.round_summaries.append(round_summary)

                    duplicate_rate = 1.0 - (len(new_paper_ids) / max(papers_found, 1)) if papers_found > 0 else 0.0

                    # Save round record
                    query_texts = [q.query_text for q in query_executions]
                    query_ids = [q.query_id for q in query_executions]
                    paper_repo.save_round(
                        db, task_id, round_num,
                        query_texts,
                        papers_found, len(new_paper_ids), duplicate_rate,
                        round_summary, gaps
                    )

                    # (#2) Persist state before returning
                    task_repo.save_state(db, task_id, state)
                    db.commit()

                    # (#1) Return formal RoundSearchResult
                    return RoundSearchResult(
                        query_ids=query_ids,
                        query_texts=query_texts,
                        papers_found=papers_found,
                        deduped_count=deduped_count,
                        new_paper_ids=new_paper_ids,
                        duplicate_rate=duplicate_rate,
                        high_priority_before=high_priority_before,
                        high_priority_after=high_priority_after,
                        new_high_priority_count=new_high,
                        round_summary=round_summary,
                        knowledge_gaps=gaps,
                    )

                search_result = await phase_service.execute_phase(
                    db, task_id, search_phase_name, _search_op,
                    input_version=round_input_version,
                    round_number=round_num,
                )
                # Reload state after search phase
                state = task_repo.get_state(db, task_id)

                # (#5) Recover RoundSearchResult from PhaseRun.output_json
                if search_result is None:
                    search_result = _recover_round_search_result(
                        db, task_id, round_num, round_input_version
                    )
                    if search_result is None:
                        raise RuntimeError(
                            f"search_round_{round_num} was skipped but no previous output found in DB"
                        )

                # (#3) Use formal result — no local variable dependencies
                new_paper_ids = search_result.new_paper_ids
                papers_found = search_result.papers_found
                duplicate_rate = search_result.duplicate_rate
                round_new_high = search_result.new_high_priority_count

                logger.info("Round %d: %d high-priority (%d new), total high=%d",
                            round_num, round_new_high, round_new_high,
                            len(state.high_priority_paper_ids))

                # NOTE: no_new_high_priority_count is updated AFTER evidence/coverage
                # succeeds, not here — retry within the same round must not pollute
                # the cross-round termination counter.

                # Phase 2.2A: Extract evidence + update coverage PER ROUND
                if state.pipeline_version >= 2:
                    ec_result = RoundEvidenceCoverageResult(
                        evidence_status="pending", coverage_status="pending",
                        new_evidence_count=0, coverage_snapshot_count=0, reason=""
                    )

                    # (#6) Evidence input version with content hashes
                    round_paper_ids = _get_round_paper_ids(db, task_id, round_num)
                    ev_input_version = hashlib.sha256(json.dumps({
                        "round_number": round_num,
                        "round_paper_ids": sorted(round_paper_ids),
                        "active_question_ids": sorted(state.active_question_ids),
                        "pipeline_version": state.pipeline_version,
                    }, sort_keys=True).encode()).hexdigest()

                    ev_phase_name = f"extract_evidence_round_{round_num}"

                    async def _ev_op(db):
                        return await extract_evidence_units(db, state, llm, task_id, round_num)

                    try:
                        task_repo.update_status(db, task_id, "extracting_evidence")
                        state.current_phase = ev_phase_name
                        task_repo.save_state(db, task_id, state)
                        db.commit()
                        emit_event(task_id, "status", {"status": "extracting_evidence", "round": round_num})

                        ev_result = await phase_service.execute_phase(
                            db, task_id, ev_phase_name, _ev_op,
                            input_version=ev_input_version,
                            round_number=round_num,
                        )
                        state = task_repo.get_state(db, task_id)
                        ev_count = ev_result if isinstance(ev_result, int) else 0
                        ec_result.new_evidence_count = ev_count

                        # (#7) Check for valid evidence limited to THIS ROUND's papers
                        valid_evidence_count = _count_valid_round_evidence(db, task_id, round_num)

                        if valid_evidence_count == 0 and ev_count == 0:
                            ec_result.evidence_status = "failed"
                            ec_result.reason = "no valid evidence for round papers"
                        else:
                            ec_result.evidence_status = "completed"
                        logger.info("Round %d: extracted %d evidence units (%d valid for round)",
                                    round_num, ev_count, valid_evidence_count)
                    except Exception as ev_err:
                        logger.error("Task %s: Round %d evidence failed: %s", task_id[:8], round_num, ev_err)
                        ec_result.evidence_status = "failed"
                        ec_result.reason = f"evidence: {str(ev_err)[:200]}"
                        emit_event(task_id, "evidence_error", {"round": round_num, "error": str(ev_err)[:200]})

                    # Coverage update (only if evidence succeeded)
                    if ec_result.evidence_status != "failed":
                        # (#6) Coverage input version with content hashes
                        round_evidence_ids = _get_round_evidence_ids(db, task_id, round_num)
                        evidence_hashes = _get_round_evidence_hashes(db, task_id, round_num)
                        cov_input_version = hashlib.sha256(json.dumps({
                            "round_evidence_ids": sorted(round_evidence_ids),
                            "evidence_hashes": sorted(evidence_hashes),
                            "active_question_ids": sorted(state.active_question_ids),
                            "round_number": round_num,
                            "pipeline_version": state.pipeline_version,
                        }, sort_keys=True).encode()).hexdigest()

                        cov_phase_name = f"update_coverage_round_{round_num}"

                        async def _cov_op(db):
                            return await update_coverage_matrix(db, state, llm, task_id, round_num)

                        try:
                            task_repo.update_status(db, task_id, "updating_coverage")
                            state.current_phase = cov_phase_name
                            task_repo.save_state(db, task_id, state)
                            db.commit()
                            await phase_service.execute_phase(
                                db, task_id, cov_phase_name, _cov_op,
                                input_version=cov_input_version,
                                round_number=round_num,
                            )
                            state = task_repo.get_state(db, task_id)
                            from app.db.models import CoverageRecord as _CR
                            ec_result.coverage_snapshot_count = db.query(_CR).filter(
                                _CR.task_id == task_id,
                                _CR.round_number == round_num,
                            ).count()
                            if ec_result.coverage_snapshot_count > 0:
                                ec_result.coverage_status = "completed"
                            else:
                                ec_result.coverage_status = "failed"
                                ec_result.reason = "no coverage snapshots generated"
                            logger.info("Round %d: coverage updated (%d snapshots)",
                                        round_num, ec_result.coverage_snapshot_count)
                        except Exception as cov_err:
                            logger.error("Task %s: Round %d coverage failed: %s", task_id[:8], round_num, cov_err)
                            ec_result.coverage_status = "failed"
                            if not ec_result.reason:
                                ec_result.reason = f"coverage: {str(cov_err)[:200]}"
                            else:
                                ec_result.reason += f" | coverage: {str(cov_err)[:200]}"
                            emit_event(task_id, "coverage_error", {"round": round_num, "error": str(cov_err)[:200]})

                    # Failure propagation: raise to trigger round retry
                    if ec_result.evidence_status == "failed" or ec_result.coverage_status == "failed":
                        raise RoundEvidenceCoverageError(ec_result.reason or "evidence/coverage failed")

                # Phase 2.2A Final Closure (#1): no_new_high_priority_count is
                # only updated AFTER the entire round (search + evidence + coverage)
                # succeeds. Retry within the same round does NOT pollute this counter.
                if round_new_high == 0:
                    no_new_high_priority_count += 1
                else:
                    no_new_high_priority_count = 0

                # Check early termination (only after full round succeeds)
                et_stop, et_reason = early_termination_check(state, no_new_high_priority_count, duplicate_rate)
                if et_stop:
                    state.stop_reason = et_reason
                    task_repo.save_state(db, task_id, state)
                    db.commit()
                    emit_event(task_id, "stopping", {"reason": et_reason})
                    logger.info("Early stop: %s", et_reason)
                    return SearchLoopResult(status="stopped_normally", reason=et_reason,
                                            completed_rounds=completed_rounds + 1,
                                            failed_attempts=total_failed_rounds)

                # (#2) Final state persistence after round completes
                task_repo.save_state(db, task_id, state)
                db.commit()
                emit_event(task_id, "round_done", {"round": round_num, "new_papers": len(new_paper_ids)})
                logger.info("Round %d complete. Total papers: %d, high-priority: %d",
                            round_num, len(state.collected_paper_ids), len(state.high_priority_paper_ids))

                round_succeeded = True
                completed_rounds += 1
                break  # exit retry loop

            except Exception as round_err:
                logger.error("Task %s: Round %d attempt %d failed: %s",
                             task_id[:8], round_num, round_attempts, round_err)
                db.rollback()
                state = task_repo.get_state(db, task_id)

                error_sig = str(round_err)[:200]
                if error_sig == last_error_signature:
                    identical_error_streak += 1
                else:
                    identical_error_streak = 1
                    last_error_signature = error_sig

                total_failed_rounds += 1

                if identical_error_streak >= 3:
                    logger.error("Task %s: same error %d times, stopping", task_id[:8], identical_error_streak)
                    reason = f"identical_error_streak ({error_sig[:100]})"
                    state.stop_reason = reason
                    task_repo.save_state(db, task_id, state)
                    db.commit()
                    emit_event(task_id, "stopping", {"reason": reason})
                    return SearchLoopResult(status="failed", reason=reason,
                                            completed_rounds=completed_rounds,
                                            failed_attempts=total_failed_rounds)

                if total_failed_rounds >= TOTAL_FAILED_ROUND_BUDGET:
                    logger.error("Task %s: total failed rounds (%d) reached budget", task_id[:8], total_failed_rounds)
                    reason = f"total_failed_rounds ({total_failed_rounds})"
                    state.stop_reason = reason
                    task_repo.save_state(db, task_id, state)
                    db.commit()
                    emit_event(task_id, "stopping", {"reason": reason})
                    return SearchLoopResult(status="failed", reason=reason,
                                            completed_rounds=completed_rounds,
                                            failed_attempts=total_failed_rounds)

                if round_attempts < MAX_ATTEMPTS_PER_ROUND:
                    await asyncio.sleep(5)

        if not round_succeeded:
            logger.error("Task %s: Round %d failed after %d attempts", task_id[:8], round_num, round_attempts)
            state.current_round -= 1
            task_repo.save_state(db, task_id, state)
            db.commit()

            if total_failed_rounds >= TOTAL_FAILED_ROUND_BUDGET:
                return SearchLoopResult(status="failed", reason="failed_round_budget_exhausted",
                                        completed_rounds=completed_rounds,
                                        failed_attempts=total_failed_rounds)

    return SearchLoopResult(status="completed", reason="loop_exited",
                            completed_rounds=completed_rounds, failed_attempts=total_failed_rounds)


def _recover_round_search_result(db, task_id: str, round_num: int,
                                  round_input_version: str = "") -> RoundSearchResult | None:
    """Phase 2.2A Final Closure: Recover RoundSearchResult from PhaseRun.output_json.

    Primary path: Read complete output_json from the completed PhaseRun.
    Legacy fallback: Approximate reconstruction from secondary tables.

    If output_json is unavailable and legacy fallback cannot produce
    a precise result (new_high_priority_count is unknown), raises
    RuntimeError instead of returning a fake result with new_high=0.
    """
    # Primary path: exact restoration from PhaseRun.output_json
    if round_input_version:
        payload = phase_repo.get_completed_phase_output(
            db, task_id, f"search_round_{round_num}", round_input_version
        )
        if payload is not None:
            try:
                result = RoundSearchResult.from_phase_payload(payload)
                logger.info("Task %s: round %d recovered from PhaseRun.output_json (new_high=%d)",
                           task_id[:8], round_num, result.new_high_priority_count)
                return result
            except (TypeError, ValueError) as e:
                logger.warning("Task %s: round %d output_json deserialization failed: %s, "
                              "trying legacy fallback",
                              task_id[:8], round_num, e)

    # Legacy fallback: approximate reconstruction (no output_json)
    from app.db.models import (
        ResearchRound as _RR, SearchQueryRecord as _SQR,
        TaskPaper as _TP,
    )

    rr = db.query(_RR).filter(
        _RR.task_id == task_id, _RR.round_number == round_num
    ).first()
    if not rr:
        return None

    queries = db.query(_SQR).filter(
        _SQR.task_id == task_id, _SQR.round_number == round_num
    ).all()
    query_ids = [q.id for q in queries]
    query_texts = [q.query_text for q in queries]

    round_tps = db.query(_TP).filter(
        _TP.task_id == task_id, _TP.discovered_round == round_num
    ).all()
    new_paper_ids = [tp.paper_id for tp in round_tps]

    gaps = json.loads(rr.knowledge_gaps_json) if rr.knowledge_gaps_json else []

    papers_found = rr.papers_found or 0
    new_papers = rr.new_papers or 0
    duplicate_rate = rr.duplicate_rate or 0.0

    # Cannot determine high_priority_before/after/new_high from secondary tables
    raise RuntimeError(
        f"Cannot precisely recover RoundSearchResult for round {round_num}: "
        f"PhaseRun.output_json is NULL or not found, and secondary tables "
        f"do not contain high_priority_before/after/new_high_priority_count. "
        f"Refusing to return a fake result with new_high=0."
    )


def _get_round_paper_ids(db, task_id: str, round_num: int) -> list[str]:
    """Get paper IDs discovered in a specific round."""
    from app.db.models import TaskPaper as _TP
    return [tp.paper_id for tp in db.query(_TP).filter(
        _TP.task_id == task_id, _TP.discovered_round == round_num
    ).all()]


def _count_valid_round_evidence(db, task_id: str, round_num: int) -> int:
    """(#7) Count valid evidence for THIS ROUND's papers only."""
    from app.db.models import EvidenceUnit as _EU, TaskPaper as _TP
    round_paper_ids = _get_round_paper_ids(db, task_id, round_num)
    if not round_paper_ids:
        return 0
    return db.query(_EU).filter(
        _EU.task_id == task_id,
        _EU.paper_id.in_(round_paper_ids),
        _EU.verification_status.notin_(["rejected", "conflicted"]),
    ).count()


def _get_round_evidence_ids(db, task_id: str, round_num: int) -> list[str]:
    """Get evidence IDs for papers discovered in a specific round."""
    from app.db.models import EvidenceUnit as _EU, TaskPaper as _TP
    round_paper_ids = _get_round_paper_ids(db, task_id, round_num)
    if not round_paper_ids:
        return []
    return [eu.id for eu in db.query(_EU).filter(
        _EU.task_id == task_id,
        _EU.paper_id.in_(round_paper_ids),
    ).all()]


def _get_round_evidence_hashes(db, task_id: str, round_num: int) -> list[str]:
    """Get source_chunk_hashes for round evidence."""
    from app.db.models import EvidenceUnit as _EU
    round_paper_ids = _get_round_paper_ids(db, task_id, round_num)
    if not round_paper_ids:
        return []
    return [eu.source_chunk_hash or "" for eu in db.query(_EU).filter(
        _EU.task_id == task_id,
        _EU.paper_id.in_(round_paper_ids),
    ).all()]


def _check_early_termination(db, task_id, state, no_new_high_count, duplicate_rate) -> bool:
    """DEPRECATED: Phase 0 — moved to policy.early_termination_check().

    Kept for backward compatibility; delegates to the new function.
    """
    stop, reason = early_termination_check(state, no_new_high_count, duplicate_rate)
    if stop:
        state.stop_reason = reason
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "stopping", {"reason": reason})
        logger.info("Early stop (legacy): %s", reason)
    return stop


async def _run_rag_indexing(db, task_id: str, llm):
    """Phase 2.5: RAG indexing for high-priority papers.

    P1-15: Only index top-20 high-priority papers (by score) to control memory
    and processing time. Medium-priority papers use abstract-only fallback.
    Previously indexed ALL high+medium (up to 125 papers) → OOM crash on Windows.
    20 papers × ~10 chunks = 200 chunks, manageable memory footprint.

    P1-17: If RAG indexing crashes (PyTorch segfault on Windows), skip it
    entirely — report/ideas generation will use abstract-only fallback.
    """
    try:
        from app.services.rag_service import fetch_and_index_papers
        from app.db.models import Paper, TaskPaper as _TP

        high_papers_for_rag = db.query(Paper).join(_TP).filter(
            _TP.task_id == task_id,
            _TP.priority == "high",
        ).order_by(_TP.final_score.desc().nullslast()).limit(20).all()

        if high_papers_for_rag:
            logger.info("Task %s: RAG indexing %d high-priority papers...",
                       task_id[:8], len(high_papers_for_rag))
            emit_event(task_id, "status", {"status": "indexing_pdfs", "total": len(high_papers_for_rag)})
            rag_summary = await fetch_and_index_papers(high_papers_for_rag, llm, task_id)
            logger.info("Task %s: RAG indexing done: %s", task_id[:8], rag_summary)
    except Exception as e:
        logger.warning("Task %s: RAG indexing failed (non-fatal, continuing with abstracts): %s",
                      task_id[:8], e)
        emit_event(task_id, "status", {"status": "indexing_skipped", "reason": str(e)[:200]})


async def _run_wiki_ingest(db, task_id: str, llm):
    """Phase 2.6: LLM Wiki ingest."""
    try:
        from app.services.wiki_service import ingest_papers_to_wiki, lint_wiki
        from app.db.models import Paper as _WikiPaper, TaskPaper as _WikiTP

        wiki_papers = db.query(_WikiPaper).join(_WikiTP).filter(
            _WikiTP.task_id == task_id,
            _WikiTP.priority.in_(["high", "medium"]),
        ).order_by(_WikiTP.final_score.desc().nullslast()).limit(30).all()

        if wiki_papers:
            logger.info("Task %s: LLM Wiki ingesting %d papers...", task_id[:8], len(wiki_papers))
            emit_event(task_id, "status", {"status": "building_wiki", "total_papers": len(wiki_papers)})
            wiki_summary = await ingest_papers_to_wiki(db, wiki_papers, llm, task_id)
            logger.info("Task %s: LLM Wiki done: %s", task_id[:8], wiki_summary)

            # P0-4: Run wiki lint after ingest to detect contradictions
            try:
                lint_result = await lint_wiki(db, task_id, llm)
                logger.info("Task %s: wiki lint found %d issues",
                           task_id[:8], lint_result.get("total", 0))
            except Exception as lint_e:
                logger.warning("Task %s: wiki lint failed (non-fatal): %s", task_id[:8], lint_e)
    except Exception as e:
        logger.warning("Task %s: LLM Wiki ingest failed (non-fatal, falling back to LLM clustering): %s",
                      task_id[:8], e)


async def _run_ideas_loop(db, state: ResearchState, llm, task_id: str, cluster_list):
    """Phase 5: Ideas generation with retry loop.

    P1-5: Uses soft-delete (idea_status='superseded') instead of hard delete
    so user-visible history is preserved across retries.
    P0-4: Reduced max_idea_rounds from 3 to 2, and retry no longer triggers
    a full search round (which was the main cause of 30-min timeout crashes).
    Instead, retry just re-generates ideas with stronger feedback constraints.
    """
    from app.db.models import ResearchIdea
    max_idea_rounds = 2  # P0-4: reduced from 3 to 2

    for idea_round in range(max_idea_rounds):
        # Collect previous ideas as feedback for retry
        prev_ideas_feedback = ""
        if idea_round > 0:
            # P1-5: soft-delete previous ideas instead of hard delete
            old_ideas = db.query(ResearchIdea).filter(
                ResearchIdea.task_id == task_id,
                ResearchIdea.idea_status == "active",
            ).order_by(ResearchIdea.created_at.desc()).all()
            if old_ideas:
                feedback_lines = []
                for oi in old_ideas:
                    feedback_lines.append(
                        f"- 「{oi.title}」(分数: {oi.final_score or 'N/A'}, 决策: {oi.decision or 'N/A'})"
                    )
                prev_ideas_feedback = (
                    f"这是第 {idea_round + 1} 次生成创意。之前 {len(old_ideas)} 个创意质量不够高"
                    f"（均未达到 go 标准 0.70）：\n"
                    + "\n".join(feedback_lines)
                    + "\n\n请基于所有累积论文，生成比之前更有深度、更有创新性的创意。不要重复之前的创意方向。"
                    + "\n\n特别注意：\n"
                    + "- related_paper_ids 必须使用 [P1], [P2] 等编号，不能编造ID\n"
                    + "- 基线必须使用真实方法名（如 TOGA, GPT-4, BERT），不能编造\n"
                    + "- 数据集必须使用真实数据集名（如 Defects4J, MultiWOZ），不能编造\n"
                )
                # P1-5: mark as superseded (preserves history, excludes from active queries)
                for oi in old_ideas:
                    oi.idea_status = "superseded"
                db.commit()
                logger.info("Task %s: soft-deleted %d old ideas for retry", task_id[:8], len(old_ideas))

        logger.info("Task %s: generating ideas (idea round %d)...", task_id[:8], idea_round + 1)
        task_repo.update_status(db, task_id, "generating_ideas")
        db.commit()
        emit_event(task_id, "status", {"status": "generating_ideas"})
        await generate_and_score_ideas(db, state, llm, task_id, prev_ideas_feedback, cluster_list)

        # Check if any "go" or "revise" ideas exist (only among active ideas)
        # P0-4: Also accept "revise" ideas as acceptable (score >= 0.50) to avoid
        # unnecessary retries that cause timeout
        go_count = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task_id,
            ResearchIdea.decision.in_(["go"]),
            ResearchIdea.idea_status == "active",
        ).count()
        revise_count = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task_id,
            ResearchIdea.decision == "revise",
            ResearchIdea.idea_status == "active",
        ).count()
        active_count = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task_id,
            ResearchIdea.idea_status == "active",
        ).count()
        logger.info("Task %s: idea round %d done, %d go, %d revise, %d active",
                    task_id[:8], idea_round + 1, go_count, revise_count, active_count)

        # Phase 0 fix: Only accept when we have go OR revise ideas.
        # Previously `active_count > 0` meant even all-reject ideas would
        # pass, which is wrong — active_count includes reject ideas.
        if go_count > 0 or revise_count > 0:
            logger.info("Task %s: ideas ready (%d go, %d revise), waiting for user review",
                       task_id[:8], go_count, revise_count)
            task_repo.update_status(db, task_id, "waiting_for_user_review")
            emit_event(task_id, "status", {"status": "waiting_for_user_review"})
            db.commit()
            break

        if idea_round < max_idea_rounds - 1:
            # P0-4: No full search round — just retry idea generation with feedback
            # (previously _idea_retry_search_round was called here, causing 30-min timeout)
            logger.info("Task %s: no qualified ideas (all reject), retrying generation",
                       task_id[:8])
        else:
            # Phase 0 fix: No auto-promote. If all ideas are rejected after
            # max retries, the task ends with insufficient_evidence.
            _finish_with_insufficient_evidence(db, task_id, active_count)


async def _idea_retry_search_round(db, state: ResearchState, llm, task_id: str, cluster_list):
    """Run an extra search round during idea retry."""
    logger.info("Task %s: no qualified ideas, searching for more papers...", task_id[:8])
    emit_event(task_id, "status", {"status": "searching", "reason": "no_qualified_ideas"})
    task_repo.update_status(db, task_id, "searching")
    db.commit()

    state.current_round += 1
    round_num = state.current_round
    logger.info("=== Task %s: Round %d (idea retry) ===", task_id[:8], round_num)
    emit_event(task_id, "round_start", {"round": round_num})

    query_executions = await generate_queries(db, state, llm)
    # state.used_queries already updated inside generate_queries
    query_payloads = [dataclasses.asdict(q) for q in query_executions]
    emit_event(task_id, "queries_generated", {
        "round": round_num,
        "queries": [q.query_text for q in query_executions],
        "structured_queries": query_payloads,
    })

    papers_found, deduped_count, new_paper_ids = await search_and_save_papers(
        db, state, query_executions, task_id, round_num
    )

    await score_papers(db, state, llm, task_id, round_num)
    round_summary, gaps = await summarize_round(db, state, llm, round_num, [])
    state.knowledge_gaps = gaps
    state.round_summaries.append(round_summary)
    query_texts = [q.query_text for q in query_executions]
    paper_repo.save_round(db, task_id, round_num, query_texts, papers_found,
                          len(new_paper_ids), 1.0 - (len(new_paper_ids) / max(papers_found, 1)),
                          round_summary, gaps)
    task_repo.save_state(db, task_id, state)
    db.commit()
    emit_event(task_id, "round_done", {"round": round_num, "new_papers": len(new_paper_ids)})

    # Idea retry: RAG index + Wiki ingest newly found high-priority papers
    if new_paper_ids:
        await _idea_retry_rag_wiki(db, task_id, llm, new_paper_ids)


async def _idea_retry_rag_wiki(db, task_id: str, llm, new_paper_ids: list[str]):
    """RAG + Wiki incremental update for idea retry."""
    try:
        from app.services.rag_service import fetch_and_index_papers
        from app.services.wiki_service import ingest_papers_to_wiki
        from app.db.models import Paper as _RetryPaper, TaskPaper as _RetryTP

        new_high_papers = db.query(_RetryPaper).join(_RetryTP).filter(
            _RetryTP.task_id == task_id,
            _RetryTP.paper_id.in_(new_paper_ids),
            _RetryTP.priority.in_(["high", "medium"]),
        ).all()

        if new_high_papers:
            logger.info("Task %s: idea retry — RAG+Wiki indexing %d new papers...",
                       task_id[:8], len(new_high_papers))
            emit_event(task_id, "status", {"status": "indexing_pdfs", "total": len(new_high_papers)})
            await fetch_and_index_papers(new_high_papers, llm, task_id)
            await ingest_papers_to_wiki(db, new_high_papers, llm, task_id)
            logger.info("Task %s: idea retry — wiki updated with new papers", task_id[:8])
    except Exception as retry_e:
        logger.warning("Task %s: idea retry RAG/Wiki failed (non-fatal): %s",
                      task_id[:8], retry_e)


def _auto_promote_ideas(db, task_id: str):
    """DEPRECATED: Phase 0 fix — auto-promote is removed.
    
    Previously this function would promote reject ideas to 'conditional_go'
    to ensure the system always returns a result. This violates the
    principle that the system must allow 0 credible ideas.
    
    Now redirects to _finish_with_insufficient_evidence.
    """
    logger.warning("Task %s: _auto_promote_ideas is deprecated, using insufficient_evidence instead", task_id[:8])
    _finish_with_insufficient_evidence(db, task_id, 0)


def _finish_with_insufficient_evidence(db, task_id: str, active_idea_count: int):
    """Phase 0: When max idea rounds reached and no idea passed validation (>=0.50),
    the task ends with insufficient_evidence status.
    
    This is a legitimate outcome — the system honestly reports that it could
    not produce credible ideas from the available evidence, rather than
    auto-promoting low-quality ideas.
    """
    logger.info("Task %s: no credible ideas after max retries (%d active, all rejected), "
                "finishing with insufficient_evidence", task_id[:8], active_idea_count)
    task_repo.update_status(db, task_id, "insufficient_evidence")
    task_repo.update_stop_reason(db, task_id, "no_credible_ideas_after_retries")
    emit_event(task_id, "status", {
        "status": "insufficient_evidence",
        "reason": "no_credible_ideas",
        "active_idea_count": active_idea_count,
    })
    db.commit()

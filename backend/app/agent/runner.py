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
import logging
import time

from app.config import settings
from app.db.session import SessionLocal
from app.db.repositories import task_repo, paper_repo
from app.agent.state import ResearchState
from app.agent.policy import should_stop
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
)
from app.agent.steps.analyze_papers import analyze_papers
from app.llm.factory import get_llm
from app.services.event_service import emit_event, emit_event_with_cleanup

logger = logging.getLogger(__name__)

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
    "searching",
    "summarizing",
    "reporting",
    "generating_ideas",
    "judging_ideas",
    "generating_experiment",
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

    AGENT_TIMEOUT = 3600  # P4-1: increased from 1800 to 3600 (60 min) to avoid timeout on retry

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

    Instead of marking all interrupted tasks as 'failed', we now:
    - searching/summarizing → reset to 'pending' so user can restart from current_round
    - clarifying/waiting_for_clarification → reset to 'pending' (re-clarify)
    - reporting/generating_ideas/judging_ideas/generating_experiment → reset to 'pending'
      so runner can resume from the RAG/Wiki phase (already-collected papers preserved)

    State is preserved in state_json + research_rounds + papers, so a restart
    will skip already-completed rounds via should_stop() checks.

    P1-10: Calibrate state.current_round against research_rounds table to prevent
    inconsistency — e.g., if state_json says current_round=5 but only 2 rounds
    were actually saved to research_rounds, a restart would immediately hit
    max_rounds and stop without doing any work. We sync state.current_round to
    the actual max(round_number) from research_rounds.
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

            # Calibrate state.current_round against actually-completed rounds.
            # research_rounds table is the source of truth for what was persisted;
            # state_json.current_round may be ahead (incremented but round failed
            # before save_round committed) or behind (rare, but possible if
            # state save failed). Sync to max(round_number) to ensure should_stop
            # doesn't prematurely trigger max_rounds on resume.
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

        # === 1. Topic clarification (skip if already clarified) ===
        if state.normalized_topic:
            logger.info("Task %s: resuming — topic already clarified: %s",
                       task_id[:8], state.normalized_topic[:50])
        else:
            await _run_clarification_phase(db, state, llm, task_id)
            # Re-check: clarification may have set waiting_for_clarification
            db.refresh = None  # noop
            state = task_repo.get_state(db, task_id)
            if not state.normalized_topic:
                return  # waiting for user clarification

        # === 2. Search loop ===
        await _run_search_loop(db, state, llm, task_id)

        # Close main session before long-running RAG/Wiki phases
        # (they don't need 'state' and can take 5-10 minutes each)
        db.close()
        db = SessionLocal()
        state = task_repo.get_state(db, task_id)  # reload after search loop

        # === 2.5. RAG: Download PDFs and index high-priority papers ===
        # P1-17: RAG indexing disabled on Windows due to PyTorch segfault.
        # Report/ideas generation uses abstract-only fallback.
        # To re-enable: await _run_rag_indexing(db, task_id, llm)
        logger.info("Task %s: skipping RAG indexing (disabled on Windows, using abstract fallback)", task_id[:8])
        emit_event(task_id, "status", {"status": "indexing_skipped", "reason": "RAG disabled on Windows"})

        # === 2.5b. Paper Deep Analysis (新增：论文深度分析) ===
        # Download PDFs for high-priority papers, extract text, LLM structured analysis
        # Results stored in paper_analyses table, used by wiki/report/ideas
        logger.info("Task %s: starting paper deep analysis...", task_id[:8])
        task_repo.update_status(db, task_id, "analyzing_papers")
        db.commit()
        emit_event(task_id, "status", {"status": "analyzing_papers"})
        try:
            await analyze_papers(db, state, llm, task_id)
        except Exception as e:
            logger.warning("Task %s: paper analysis failed (non-fatal, continuing): %s", task_id[:8], e)

        # === 2.6. LLM Wiki: Ingest papers into wiki knowledge base ===
        await _run_wiki_ingest(db, task_id, llm)

        # === 3. Build clusters once (used by both report and ideas) ===
        logger.info("Task %s: building clusters...", task_id[:8])
        cluster_list = await build_paper_clusters(db, state, llm, task_id)

        # === 4. Generate report ONCE ===
        logger.info("Task %s: generating report...", task_id[:8])
        task_repo.update_status(db, task_id, "reporting")
        db.commit()
        emit_event(task_id, "status", {"status": "reporting"})
        await generate_report(db, state, llm, cluster_list)

        # === 5. Ideas generation loop (retry if no qualified ideas) ===
        await _run_ideas_loop(db, state, llm, task_id, cluster_list)

    finally:
        db.close()


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
    """Phase 1: Topic clarification."""
    task_repo.update_status(db, task_id, "clarifying")
    emit_event(task_id, "status", {"status": "clarifying"})

    # Skip clarify if user already submitted clarifications
    if "\nClarifications:" in state.user_input and not state.normalized_topic:
        state.normalized_topic = state.user_input.split("\nClarifications:")[0].strip()
        state.keywords = []
        task_repo.update_normalized_topic(db, task_id, state.normalized_topic)
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "status", {"status": "clarified", "topic": state.normalized_topic})
        logger.info("Skipping clarify (already clarified): %s", state.normalized_topic)
        return

    clarity = await clarify_topic(db, state, llm)

    if not clarity.is_clear:
        state.research_questions = clarity.questions
        task_repo.save_state(db, task_id, state)
        task_repo.update_status(db, task_id, "waiting_for_clarification")
        emit_event(task_id, "clarification_needed", {"questions": clarity.questions})
        db.commit()
        return  # Wait for user to answer

    state.normalized_topic = clarity.normalized_topic or state.user_input
    state.keywords = clarity.keywords
    task_repo.update_normalized_topic(db, task_id, state.normalized_topic)
    task_repo.save_state(db, task_id, state)
    db.commit()


async def _run_search_loop(db, state: ResearchState, llm, task_id: str):
    """Phase 2: Multi-round search loop."""
    task_repo.update_status(db, task_id, "searching")
    db.commit()
    emit_event(task_id, "status", {"status": "searching", "topic": state.normalized_topic})

    no_new_high_priority_count = 0

    while True:
        stop, reason = should_stop(state)
        if stop:
            state.stop_reason = reason
            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "stopping", {"reason": reason})
            logger.info("Stopping search: %s", reason)
            break

        state.current_round += 1
        round_num = state.current_round
        logger.info("=== Task %s: Round %d ===", task_id[:8], round_num)
        emit_event(task_id, "round_start", {"round": round_num})

        try:
            # Generate queries
            queries = await generate_queries(db, state, llm)
            state.used_queries.extend(queries)
            emit_event(task_id, "queries_generated", {"round": round_num, "queries": queries})

            # Search + dedup + save
            papers_found, deduped_count, new_paper_ids = await search_and_save_papers(
                db, state, queries, task_id, round_num
            )

            # Score papers
            high_priority_before = len(state.high_priority_paper_ids)
            scored_papers = await score_papers(db, state, llm, task_id, round_num)

            # Count new high-priority
            new_high = len(state.high_priority_paper_ids) - high_priority_before
            logger.info("Round %d: %d high-priority (%d new), total high=%d",
                        round_num, new_high, new_high, len(state.high_priority_paper_ids))
            if new_high == 0:
                no_new_high_priority_count += 1
            else:
                no_new_high_priority_count = 0

            # Round summary
            task_repo.update_status(db, task_id, "summarizing")
            db.commit()
            round_summary, gaps = await summarize_round(db, state, llm, round_num, scored_papers)
            state.knowledge_gaps = gaps
            state.round_summaries.append(round_summary)

            duplicate_rate = 1.0 - (len(new_paper_ids) / max(papers_found, 1))

            # Save round record
            paper_repo.save_round(
                db, task_id, round_num, queries,
                papers_found, len(new_paper_ids), duplicate_rate,
                round_summary, gaps
            )

            # Check early termination
            if _check_early_termination(db, task_id, state, no_new_high_priority_count, duplicate_rate):
                break

            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "round_done", {"round": round_num, "new_papers": len(new_paper_ids)})
            logger.info("Round %d complete. Total papers: %d, high-priority: %d",
                        round_num, len(state.collected_paper_ids), len(state.high_priority_paper_ids))

        except Exception as round_err:
            # Single round failure should NOT terminate the entire task.
            # Roll back uncommitted changes, log, and continue to next round.
            logger.error("Task %s: Round %d failed: %s — continuing to next round",
                         task_id[:8], round_num, round_err, exc_info=True)
            db.rollback()
            state.current_round -= 1  # don't consume a round on failure
            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "round_error", {"round": round_num, "error": str(round_err)[:200]})
            await asyncio.sleep(5)  # cooldown before retry


def _check_early_termination(db, task_id, state, no_new_high_count, duplicate_rate) -> bool:
    """Check early termination conditions. Returns True if should stop.

    P4-1: Lowered duplicate_rate threshold from 0.75 to 0.65 to stop sooner
    when search results are becoming redundant.
    """
    if no_new_high_count >= 2 and state.current_round >= 2:
        state.stop_reason = "no_new_high_priority_2_rounds"
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "stopping", {"reason": state.stop_reason})
        logger.info("Early stop: no new high-priority for 2 rounds")
        return True

    if duplicate_rate > 0.65 and state.current_round >= 2:  # P4-1: 0.75 -> 0.65
        state.stop_reason = "high_duplicate_rate"
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "stopping", {"reason": state.stop_reason})
        logger.info("Early stop: high duplicate rate %.2f", duplicate_rate)
        return True

    return False


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

        # P0-4: Accept if we have any go OR revise ideas, or any active ideas at all
        # (previously only "go" counted, causing excessive retries)
        if go_count > 0 or revise_count > 0 or active_count > 0:
            logger.info("Task %s: ideas ready (%d go, %d revise), waiting for user review",
                       task_id[:8], go_count, revise_count)
            task_repo.update_status(db, task_id, "waiting_for_user_review")
            emit_event(task_id, "status", {"status": "waiting_for_user_review"})
            db.commit()
            break

        if idea_round < max_idea_rounds - 1:
            # P0-4: No full search round — just retry idea generation with feedback
            # (previously _idea_retry_search_round was called here, causing 30-min timeout)
            logger.info("Task %s: no qualified ideas, retrying generation (no new search round)",
                       task_id[:8])
        else:
            # Last attempt: auto-promote top ideas
            _auto_promote_ideas(db, task_id)


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

    queries = await generate_queries(db, state, llm)
    state.used_queries.extend(queries)
    emit_event(task_id, "queries_generated", {"round": round_num, "queries": queries})

    papers_found, deduped_count, new_paper_ids = await search_and_save_papers(
        db, state, queries, task_id, round_num
    )

    await score_papers(db, state, llm, task_id, round_num)
    round_summary, gaps = await summarize_round(db, state, llm, round_num, [])
    state.knowledge_gaps = gaps
    state.round_summaries.append(round_summary)
    paper_repo.save_round(db, task_id, round_num, queries, papers_found,
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
    """Last resort: when max idea rounds reached and no idea passed validation (>=0.70),
    promote the top-scoring candidates as 'conditional_go' so the user can still review them.

    IMPORTANT: 'conditional_go' is NOT the same as 'go'. 'go' means the idea passed
    the 5-layer validation (novelty/baseline/dataset/metric/evidence). 'conditional_go'
    means the system could not produce a validated idea after max retries and is
    surfacing the best-available candidates for the user to decide manually.

    This preserves the semantic integrity of 'go' while still returning a result.
    """
    from app.db.models import ResearchIdea

    logger.info("Task %s: max idea rounds reached, auto-promoting top ideas as conditional_go",
                task_id[:8])
    # P1-5: only consider active ideas (not superseded)
    all_ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task_id,
        ResearchIdea.idea_status == "active",
    ).order_by(ResearchIdea.final_score.desc().nullslast()).all()
    promoted = 0
    promoted_titles = []
    for idea in all_ideas:
        if idea.final_score and idea.final_score >= 0.55:
            # Use 'conditional_go' instead of 'go' to preserve validation semantics.
            # Frontend treats both as selectable, but badge clearly distinguishes them.
            idea.decision = "conditional_go"
            promoted += 1
            promoted_titles.append(idea.title[:40] if idea.title else "")
        if promoted >= 3:
            break
    db.commit()

    # Record trace for auditability (previously this silent state change was unlogged)
    paper_repo.save_trace(
        db, task_id, "auto_promote_ideas", "decision",
        output_data={
            "promoted_count": promoted,
            "decision_used": "conditional_go",
            "promoted_titles": promoted_titles,
            "note": "ideas did not pass validation; surfaced as conditional_go for user review",
        },
    )
    db.commit()

    logger.info("Task %s: promoted %d ideas as conditional_go", task_id[:8], promoted)
    task_repo.update_status(db, task_id, "waiting_for_user_review")
    emit_event(task_id, "status", {"status": "waiting_for_user_review", "reason": "auto_promoted_conditional"})
    db.commit()

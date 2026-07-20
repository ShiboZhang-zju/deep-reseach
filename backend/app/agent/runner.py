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
from app.llm.factory import get_llm
from app.services.event_service import emit_event, emit_event_with_cleanup

logger = logging.getLogger(__name__)

# Task registry for asyncio tasks
_task_registry: dict[str, asyncio.Task] = {}

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


def start_agent(task_id: str):
    """Start the agent loop as an asyncio background task with timeout protection."""
    if task_id in _task_registry and not _task_registry[task_id].done():
        return

    AGENT_TIMEOUT = 1800  # 30 minutes max

    async def _run():
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

    task = asyncio.create_task(_run())
    _task_registry[task_id] = task


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
    if task_id in _task_registry:
        _task_registry[task_id].cancel()
        del _task_registry[task_id]


def recover_interrupted_tasks():
    """On startup, mark tasks with interrupted statuses as 'failed'.

    Call this once at application startup (see main.py) to recover from
    process crashes. Tasks stuck in running states are marked as failed
    so the user can restart them.
    """
    db = SessionLocal()
    try:
        from app.db.models import ResearchTask
        tasks = db.query(ResearchTask).filter(
            ResearchTask.status.in_(list(_INTERRUPTED_STATUSES))
        ).all()
        for task in tasks:
            logger.warning("Task %s was interrupted (status=%s), marking as failed",
                          task.id[:8], task.status)
            task.status = "failed"
            task.stop_reason = "interrupted_by_restart"
        db.commit()
        if tasks:
            logger.info("Recovered %d interrupted tasks", len(tasks))
    except Exception as e:
        logger.error("Failed to recover interrupted tasks: %s", e)
        db.rollback()
    finally:
        db.close()


async def run_task(task_id: str):
    """Main agent loop."""
    db = SessionLocal()
    try:
        state = task_repo.get_state(db, task_id)
        llm = get_llm()

        # === 1. Topic clarification ===
        await _run_clarification_phase(db, state, llm, task_id)

        # === 2. Search loop ===
        await _run_search_loop(db, state, llm, task_id)

        # === 2.5. RAG: Download PDFs and index high-priority papers ===
        await _run_rag_indexing(db, task_id, llm)

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


def _check_early_termination(db, task_id, state, no_new_high_count, duplicate_rate) -> bool:
    """Check early termination conditions. Returns True if should stop."""
    if no_new_high_count >= 2 and state.current_round >= 2:
        state.stop_reason = "no_new_high_priority_2_rounds"
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "stopping", {"reason": state.stop_reason})
        logger.info("Early stop: no new high-priority for 2 rounds")
        return True

    if duplicate_rate > 0.75 and state.current_round >= 2:
        state.stop_reason = "high_duplicate_rate"
        task_repo.save_state(db, task_id, state)
        db.commit()
        emit_event(task_id, "stopping", {"reason": state.stop_reason})
        logger.info("Early stop: high duplicate rate %.2f", duplicate_rate)
        return True

    return False


async def _run_rag_indexing(db, task_id: str, llm):
    """Phase 2.5: RAG indexing for high-priority papers."""
    try:
        from app.services.rag_service import fetch_and_index_papers
        from app.db.models import Paper, TaskPaper as _TP

        high_papers_for_rag = db.query(Paper).join(_TP).filter(
            _TP.task_id == task_id,
            _TP.priority.in_(["high", "medium"]),
        ).all()

        if high_papers_for_rag:
            logger.info("Task %s: RAG indexing %d high-priority papers...",
                       task_id[:8], len(high_papers_for_rag))
            emit_event(task_id, "status", {"status": "indexing_pdfs", "total": len(high_papers_for_rag)})
            rag_summary = await fetch_and_index_papers(high_papers_for_rag, llm, task_id)
            logger.info("Task %s: RAG indexing done: %s", task_id[:8], rag_summary)
    except Exception as e:
        logger.warning("Task %s: RAG indexing failed (non-fatal, continuing with abstracts): %s",
                      task_id[:8], e)


async def _run_wiki_ingest(db, task_id: str, llm):
    """Phase 2.6: LLM Wiki ingest."""
    try:
        from app.services.wiki_service import ingest_papers_to_wiki, lint_wiki
        from app.db.models import Paper as _WikiPaper, TaskPaper as _WikiTP

        wiki_papers = db.query(_WikiPaper).join(_WikiTP).filter(
            _WikiTP.task_id == task_id,
            _WikiTP.priority.in_(["high", "medium"]),
        ).all()

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
    """Phase 5: Ideas generation with retry loop."""
    from app.db.models import ResearchIdea
    max_idea_rounds = 3

    for idea_round in range(max_idea_rounds):
        # Collect previous ideas as feedback for retry
        prev_ideas_feedback = ""
        if idea_round > 0:
            old_ideas = db.query(ResearchIdea).filter(
                ResearchIdea.task_id == task_id,
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
                )
                db.query(ResearchIdea).filter(ResearchIdea.task_id == task_id).delete()
                db.commit()
                logger.info("Task %s: deleted %d old ideas for retry", task_id[:8], len(old_ideas))

        logger.info("Task %s: generating ideas (idea round %d)...", task_id[:8], idea_round + 1)
        task_repo.update_status(db, task_id, "generating_ideas")
        db.commit()
        emit_event(task_id, "status", {"status": "generating_ideas"})
        await generate_and_score_ideas(db, state, llm, task_id, prev_ideas_feedback, cluster_list)

        # Check if any "go" ideas exist
        go_count = db.query(ResearchIdea).filter(
            ResearchIdea.task_id == task_id,
            ResearchIdea.decision == "go",
        ).count()
        logger.info("Task %s: idea round %d done, %d go ideas",
                    task_id[:8], idea_round + 1, go_count)

        if go_count > 0:
            logger.info("Task %s: ideas ready, waiting for user review", task_id[:8])
            task_repo.update_status(db, task_id, "waiting_for_user_review")
            emit_event(task_id, "status", {"status": "waiting_for_user_review"})
            db.commit()
            break

        if idea_round < max_idea_rounds - 1:
            # No go ideas, do another search round
            await _idea_retry_search_round(db, state, llm, task_id, cluster_list)
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
    """Last resort: auto-promote top ideas to 'go' status."""
    from app.db.models import ResearchIdea

    logger.info("Task %s: max idea rounds reached, auto-promoting top ideas", task_id[:8])
    all_ideas = db.query(ResearchIdea).filter(
        ResearchIdea.task_id == task_id,
    ).order_by(ResearchIdea.final_score.desc()).all()
    promoted = 0
    for idea in all_ideas:
        if idea.final_score and idea.final_score >= 0.55:
            idea.decision = "go"
            promoted += 1
        if promoted >= 3:
            break
    db.commit()
    logger.info("Task %s: promoted %d ideas to go", task_id[:8], promoted)
    task_repo.update_status(db, task_id, "waiting_for_user_review")
    emit_event(task_id, "status", {"status": "waiting_for_user_review", "reason": "auto_promoted"})
    db.commit()

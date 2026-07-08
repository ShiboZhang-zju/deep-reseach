"""Agent Runner - main orchestration loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime

from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.db.repositories import task_repo, paper_repo
from app.agent.state import ResearchState
from app.agent.policy import should_stop
from app.agent.prompts import *
from app.llm.factory import get_llm
from app.services.search_service import search_service
from app.services.scoring_service import normalize_paper, deduplicate_papers
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

# Task registry for asyncio tasks
_task_registry: dict[str, asyncio.Task] = {}


def start_agent(task_id: str):
    """Start the agent loop as an asyncio background task."""
    if task_id in _task_registry and not _task_registry[task_id].done():
        return

    async def _run():
        try:
            await run_task(task_id)
        except Exception as e:
            logger.exception("Agent task %s failed", task_id)
            # Retry status update with backoff (SQLite lock may block)
            import time
            for attempt in range(3):
                try:
                    db = SessionLocal()
                    try:
                        task_repo.update_status(db, task_id, "failed")
                        task_repo.update_stop_reason(db, task_id, str(e)[:500])
                        db.commit()
                        emit_event(task_id, "error", {"message": str(e)})
                    finally:
                        db.close()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        logger.error("Failed to update task status after 3 retries")

    task = asyncio.create_task(_run())
    _task_registry[task_id] = task


def stop_agent(task_id: str):
    """Stop a running agent task."""
    if task_id in _task_registry:
        _task_registry[task_id].cancel()
        del _task_registry[task_id]


async def run_task(task_id: str):
    """Main agent loop."""
    db = SessionLocal()
    try:
        state = task_repo.get_state(db, task_id)
        llm = get_llm()

        # 1. Topic clarification
        task_repo.update_status(db, task_id, "clarifying")
        emit_event(task_id, "status", {"status": "clarifying"})

        # Skip clarify if user already submitted clarifications
        if "\nClarifications:" in state.user_input and not state.normalized_topic:
            # Use the clarified input directly as topic
            state.normalized_topic = state.user_input.split("\nClarifications:")[0].strip()
            state.keywords = []
            task_repo.update_normalized_topic(db, task_id, state.normalized_topic)
            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "status", {"status": "clarified", "topic": state.normalized_topic})
            logger.info("Skipping clarify (already clarified): %s", state.normalized_topic)
        else:
            clarity = await _clarify_topic(db, state, llm)

            if not clarity.is_clear:
                state.research_questions = clarity.questions
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

        # 2. Search loop
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
            queries = await _generate_queries(db, state, llm)
            state.used_queries.extend(queries)
            logger.info("Round %d: generated %d queries", round_num, len(queries))
            emit_event(task_id, "queries_generated", {"round": round_num, "queries": queries})

            # Search
            raw_papers = await search_service.search_multiple_queries(
                queries, settings.papers_per_source_per_query
            )
            papers_found = len(raw_papers)
            logger.info("Round %d: found %d raw papers", round_num, papers_found)
            emit_event(task_id, "search_done", {"round": round_num, "found": papers_found})

            # Deduplicate within batch
            deduped = deduplicate_papers(raw_papers)
            logger.info("Round %d: %d papers after dedup", round_num, len(deduped))

            # Save to DB and track new vs existing
            new_paper_ids = []
            for raw in deduped:
                normalized = normalize_paper(raw)
                paper, is_new = paper_repo.upsert_paper(db, normalized)
                tp = paper_repo.create_task_paper(db, task_id, paper.id, round_num)
                if is_new:
                    new_paper_ids.append(paper.id)
                state.collected_paper_ids.append(paper.id) if paper.id not in state.collected_paper_ids else None

            db.commit()
            logger.info("Round %d: %d new, %d dup", round_num, len(new_paper_ids), len(deduped) - len(new_paper_ids))

            # Score papers
            high_priority_before = len(state.high_priority_paper_ids)
            scored_papers = await _score_papers(db, state, llm, task_id, round_num)

            # Count new high-priority
            new_high = len(state.high_priority_paper_ids) - high_priority_before
            logger.info("Round %d: %d high-priority (%d new), total high=%d", round_num, new_high, new_high, len(state.high_priority_paper_ids))
            if new_high == 0:
                no_new_high_priority_count += 1
            else:
                no_new_high_priority_count = 0

            # Round summary
            task_repo.update_status(db, task_id, "summarizing")
            db.commit()
            round_summary, gaps = await _summarize_round(db, state, llm, round_num, scored_papers)
            state.knowledge_gaps = gaps
            state.round_summaries.append(round_summary)
            logger.info("Round %d: summary done, %d gaps", round_num, len(gaps))

            duplicate_rate = 1.0 - (len(new_paper_ids) / max(papers_found, 1))

            # Save round record
            paper_repo.save_round(
                db, task_id, round_num, queries,
                papers_found, len(new_paper_ids), duplicate_rate,
                round_summary, gaps
            )

            # Check early termination
            if no_new_high_priority_count >= 2 and state.current_round >= 2:
                state.stop_reason = "no_new_high_priority_2_rounds"
                task_repo.save_state(db, task_id, state)
                db.commit()
                emit_event(task_id, "stopping", {"reason": state.stop_reason})
                logger.info("Early stop: no new high-priority for 2 rounds")
                break

            if duplicate_rate > 0.75 and state.current_round >= 2:
                state.stop_reason = "high_duplicate_rate"
                task_repo.save_state(db, task_id, state)
                db.commit()
                emit_event(task_id, "stopping", {"reason": state.stop_reason})
                logger.info("Early stop: high duplicate rate %.2f", duplicate_rate)
                break

            task_repo.save_state(db, task_id, state)
            db.commit()
            emit_event(task_id, "round_done", {"round": round_num, "new_papers": len(new_paper_ids)})
            logger.info("Round %d complete. Total papers: %d, high-priority: %d", round_num, len(state.collected_paper_ids), len(state.high_priority_paper_ids))

        # 2.5. RAG: Download PDFs and index high-priority papers for full-text retrieval
        try:
            from app.services.rag_service import fetch_and_index_papers
            from app.db.models import Paper, TaskPaper as _TP

            high_papers_for_rag = db.query(Paper).join(_TP).filter(
                _TP.task_id == task_id,
                _TP.priority == "high",
            ).all()

            if high_papers_for_rag:
                logger.info("Task %s: RAG indexing %d high-priority papers...", task_id[:8], len(high_papers_for_rag))
                emit_event(task_id, "status", {"status": "indexing_pdfs", "total": len(high_papers_for_rag)})
                rag_summary = await fetch_and_index_papers(high_papers_for_rag, llm, task_id)
                logger.info("Task %s: RAG indexing done: %s", task_id[:8], rag_summary)
        except Exception as e:
            logger.warning("Task %s: RAG indexing failed (non-fatal, continuing with abstracts): %s", task_id[:8], e)

        # 3-4. Report + Ideas generation loop (retry if no qualified ideas)
        from app.db.models import ResearchIdea
        max_idea_rounds = 3  # Max attempts to generate qualified ideas

        for idea_round in range(max_idea_rounds):
            logger.info("Task %s: generating report (idea round %d)...", task_id[:8], idea_round + 1)
            task_repo.update_status(db, task_id, "reporting")
            db.commit()
            emit_event(task_id, "status", {"status": "reporting"})
            report_markdown = await _generate_report(db, state, llm)
            logger.info("Task %s: report generated (%d chars)", task_id[:8], len(report_markdown))

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
                        f"这是第 {idea_round + 1} 次生成创意。之前 {len(old_ideas)} 个创意质量不够高（均未达到 go 标准 0.70）：\n"
                        + "\n".join(feedback_lines)
                        + "\n\n请基于所有累积论文，生成比之前更有深度、更有创新性的创意。不要重复之前的创意方向。"
                    )
                    # Delete old ideas to avoid clutter
                    db.query(ResearchIdea).filter(ResearchIdea.task_id == task_id).delete()
                    db.commit()
                    logger.info("Task %s: deleted %d old ideas for retry", task_id[:8], len(old_ideas))

            logger.info("Task %s: generating ideas (idea round %d)...", task_id[:8], idea_round + 1)
            task_repo.update_status(db, task_id, "generating_ideas")
            db.commit()
            emit_event(task_id, "status", {"status": "generating_ideas"})
            await _generate_and_score_ideas(db, state, llm, task_id, prev_ideas_feedback)

            # Check if any "go" ideas exist
            go_count = db.query(ResearchIdea).filter(
                ResearchIdea.task_id == task_id,
                ResearchIdea.decision == "go",
            ).count()
            logger.info("Task %s: idea round %d done, %d go ideas", task_id[:8], idea_round + 1, go_count)

            if go_count > 0:
                # Has qualified ideas, wait for user review
                logger.info("Task %s: ideas ready, waiting for user review", task_id[:8])
                task_repo.update_status(db, task_id, "waiting_for_user_review")
                emit_event(task_id, "status", {"status": "waiting_for_user_review"})
                db.commit()
                break

            if idea_round < max_idea_rounds - 1:
                # No go ideas, do another search round to find more papers
                logger.info("Task %s: no qualified ideas, searching for more papers...", task_id[:8])
                emit_event(task_id, "status", {"status": "searching", "reason": "no_qualified_ideas"})
                task_repo.update_status(db, task_id, "searching")
                db.commit()

                state.current_round += 1
                round_num = state.current_round
                logger.info("=== Task %s: Round %d (idea retry) ===", task_id[:8], round_num)
                emit_event(task_id, "round_start", {"round": round_num})

                queries = await _generate_queries(db, state, llm)
                state.used_queries.extend(queries)
                emit_event(task_id, "queries_generated", {"round": round_num, "queries": queries})

                raw_papers = await search_service.search_multiple_queries(
                    queries, settings.papers_per_source_per_query
                )
                emit_event(task_id, "search_done", {"round": round_num, "found": len(raw_papers)})

                deduped = deduplicate_papers(raw_papers)
                new_paper_ids = []
                for raw in deduped:
                    normalized = normalize_paper(raw)
                    paper, is_new = paper_repo.upsert_paper(db, normalized)
                    paper_repo.create_task_paper(db, task_id, paper.id, round_num)
                    if is_new:
                        new_paper_ids.append(paper.id)
                    state.collected_paper_ids.append(paper.id) if paper.id not in state.collected_paper_ids else None
                db.commit()

                await _score_papers(db, state, llm, task_id, round_num)
                round_summary, gaps = await _summarize_round(db, state, llm, round_num, [])
                state.knowledge_gaps = gaps
                state.round_summaries.append(round_summary)
                paper_repo.save_round(db, task_id, round_num, queries, len(raw_papers),
                                      len(new_paper_ids), 1.0 - (len(new_paper_ids) / max(len(raw_papers), 1)),
                                      round_summary, gaps)
                task_repo.save_state(db, task_id, state)
                db.commit()
                emit_event(task_id, "round_done", {"round": round_num, "new_papers": len(new_paper_ids)})
            else:
                # Last attempt, still no go ideas — auto-promote top ideas
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

        # Deep evaluate selected ideas
        from app.db.models import ResearchIdea
        ideas = db.query(ResearchIdea).filter(ResearchIdea.id.in_(idea_ids)).all()

        good_ideas = []
        for idea in ideas:
            try:
                scores = await _score_idea(db, state, llm, idea)
            except Exception as e:
                logger.error("Failed to deep-score idea %s: %s", idea.id, e)
                # Use existing scores from initial evaluation
                if idea.final_score and idea.final_score >= 0.75:
                    good_ideas.append(idea)
                continue

            idea_score = (
                0.20 * scores.novelty + 0.20 * scores.feasibility + 0.20 * scores.significance +
                0.20 * scores.evidence_support + 0.10 * scores.differentiation +
                0.10 * scores.experimentability
            )
            final_score = idea_score - 0.08 * scores.risk

            if final_score >= 0.70:
                decision = "go"
            elif final_score >= 0.50:
                decision = "revise"
            else:
                decision = "reject"

            paper_repo.update_idea_scores(db, idea.id, scores.model_dump(), final_score, decision)
            idea.user_selected = True
            db.flush()
            db.commit()  # Commit after each idea to avoid losing progress

            if decision == "go":
                good_ideas.append(idea)
            logger.info("Idea '%s' deep-scored: %.3f -> %s", idea.title[:40], final_score, decision)

        emit_event(task_id, "ideas_judged", {
            "total": len(ideas),
            "go": len(good_ideas),
        })

        if not good_ideas:
            task_repo.update_status(db, task_id, "waiting_for_user_review")
            emit_event(task_id, "status", {"status": "waiting_for_user_review", "reason": "no_idea_ready"})
            db.commit()
            return {"status": "need_more_research", "reason": "No selected idea is ready for experiment."}

        # Generate experiment plans
        logger.info("Task %s: generating experiments for %d ideas...", task_id[:8], len(good_ideas))
        task_repo.update_status(db, task_id, "generating_experiment")
        db.commit()
        emit_event(task_id, "status", {"status": "generating_experiment"})

        from app.agent.prompts import EXPERIMENT_SYSTEM, EXPERIMENT_USER
        from app.schemas.schemas import ExperimentPlanSchema

        for idea in good_ideas:
            messages = [
                {"role": "system", "content": EXPERIMENT_SYSTEM},
                {"role": "user", "content": EXPERIMENT_USER.format(
                    topic=state.normalized_topic,
                    title=idea.title or "",
                    description=idea.description or "",
                    method=idea.method_sketch or "",
                    contribution=idea.expected_contribution or "",
                    related_papers=idea.related_paper_ids_json or "",
                )},
            ]
            try:
                plan = await llm.chat_json(messages, ExperimentPlanSchema)
            except Exception as e:
                logger.error("Failed to generate experiment for idea %s: %s", idea.id, e)
                continue

            plan_data = {
                "hypothesis": plan.hypothesis,
                "dataset": plan.dataset,
                "baselines": plan.baselines,
                "metrics": plan.metrics,
                "steps_markdown": "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan.steps)),
                "steps_json": json.dumps(plan.steps, ensure_ascii=False),
                "risks": plan.risks,
            }
            paper_repo.save_experiment(db, task_id, idea.id, plan_data)
            db.commit()
            emit_event(task_id, "experiment_generated", {"idea_id": idea.id, "title": idea.title})
            logger.info("Experiment generated for idea '%s'", idea.title[:40])

        task_repo.update_status(db, task_id, "done")
        emit_event(task_id, "status", {"status": "done"})
        db.commit()
        return {"status": "done", "experiments_generated": len(good_ideas)}

    finally:
        db.close()


# === Step implementations ===

async def _clarify_topic(db, state: ResearchState, llm):
    from app.schemas.schemas import ClarityResult
    messages = [
        {"role": "system", "content": CLARIFY_SYSTEM},
        {"role": "user", "content": CLARIFY_USER.format(user_input=state.user_input)},
    ]
    result = await llm.chat_json(messages, ClarityResult)
    paper_repo.save_trace(db, state.task_id, "clarify_topic", "action",
                          output_data=result.model_dump())
    db.commit()
    return result


async def _generate_queries(db, state: ResearchState, llm) -> list[str]:
    from app.schemas.schemas import QueryList
    messages = [
        {"role": "system", "content": QUERIES_SYSTEM.format(num_queries=settings.queries_per_round)},
        {"role": "user", "content": QUERIES_USER.format(
            topic=state.normalized_topic,
            keywords=", ".join(state.keywords),
            used_queries="\n".join(state.used_queries[-20:]) if state.used_queries else "(none)",
            gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
            feedback=state.user_feedback or "(none)",
            num_queries=settings.queries_per_round,
        )},
    ]
    result = await llm.chat_json(messages, QueryList)
    paper_repo.save_trace(db, state.task_id, "generate_queries", "action",
                          round_number=state.current_round,
                          output_data={"queries": result.queries})
    db.commit()
    return result.queries


async def _score_papers(db, state: ResearchState, llm, task_id: str, round_num: int):
    """Score new papers from this round (concurrent with semaphore)."""
    from app.db.models import TaskPaper, Paper
    from app.schemas.schemas import PaperScore

    # Get unscored task papers from this round
    unscored = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.discovered_round == round_num,
        TaskPaper.final_score.is_(None),
    ).all()

    # Pre-fetch all papers (avoid DB access during concurrent LLM calls)
    paper_map = {}
    for tp in unscored:
        paper = db.get(Paper, tp.paper_id)
        if paper:
            paper_map[tp.id] = (tp, paper)

    if not paper_map:
        return []

    logger.info("Scoring %d papers in round %d (concurrent, max 5)...", len(paper_map), round_num)

    # Concurrent scoring with semaphore
    semaphore = asyncio.Semaphore(5)

    async def score_one(tp_id: str):
        tp, paper = paper_map[tp_id]
        messages = [
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user", "content": SCORE_USER.format(
                topic=state.normalized_topic,
                title=paper.title,
                abstract=(paper.abstract or "")[:1000],
                authors=paper.authors_json or "",
                year=paper.year or "",
                venue=paper.venue or "",
                citations=paper.citation_count or 0,
            )},
        ]
        async with semaphore:
            try:
                score = await llm.chat_json(messages, PaperScore)
                return tp_id, score, None
            except Exception as e:
                logger.error("Failed to score paper %s: %s", paper.id, e)
                return tp_id, None, str(e)

    results = await asyncio.gather(*[score_one(tp_id) for tp_id in paper_map])

    # Process results sequentially (DB writes)
    scored = []
    for tp_id, score, error in results:
        if score is None:
            continue

        tp, paper = paper_map[tp_id]
        # P1-03: Adjusted weights — authority 0.20→0.25, relevance 0.35→0.30
        authority_adj = score.authority
        # Penalize papers with missing metadata (no citations + no year)
        if (paper.citation_count or 0) == 0 and paper.year is None:
            authority_adj = score.authority * 0.7
        # Boost papers from top venues
        TOP_VENUE_KEYWORDS = ["ICML", "NeurIPS", "ICLR", "CVPR", "ACL", "EMNLP",
                              "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "TSE", "TACL",
                              "Nature", "Science", "PAMI", "JMLR", "ICSE", "FSE",
                              "ASE", "ISSTA", "OOPSLA", "PLDI"]
        venue_str = (paper.venue or "").upper()
        if any(kv in venue_str for kv in TOP_VENUE_KEYWORDS):
            authority_adj = min(1.0, authority_adj + 0.1)

        final_score = (
            0.30 * score.relevance + 0.25 * authority_adj + 0.15 * score.recency +
            0.15 * score.novelty + 0.15 * score.idea_potential
        )
        priority = "high" if final_score >= 0.75 else ("medium" if final_score >= 0.5 else "low")

        paper_repo.update_task_paper_scores(
            db, tp.id, score.model_dump(), final_score, priority,
            score.reason, f"{score.summary} | 方法: {score.method_extract}" if score.method_extract else score.summary
        )

        if priority == "high":
            state.high_priority_paper_ids.append(paper.id)
        elif priority == "medium":
            state.medium_priority_paper_ids.append(paper.id)
        else:
            state.low_priority_paper_ids.append(paper.id)

        scored.append({
            "title": paper.title,
            "score": final_score,
            "priority": priority,
            "summary": score.summary,
        })

    logger.info("Scored %d/%d papers in round %d", len(scored), len(paper_map), round_num)

    paper_repo.save_trace(db, state.task_id, "score_papers", "action",
                          round_number=round_num,
                          output_data={"scored_count": len(scored)})
    db.commit()
    return scored


async def _summarize_round(db, state: ResearchState, llm, round_num: int, scored_papers: list):
    from app.schemas.schemas import RoundSummary
    papers_text = "\n".join(
        f"- {p['title']} (score: {p['score']:.2f}, {p['priority']}): {p['summary']}"
        for p in scored_papers[:30]
    )
    messages = [
        {"role": "system", "content": ROUND_SUMMARY_SYSTEM},
        {"role": "user", "content": ROUND_SUMMARY_USER.format(
            topic=state.normalized_topic,
            round_num=round_num,
            papers_summary=papers_text or "(no papers)",
            previous_gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
        )},
    ]
    result = await llm.chat_json(messages, RoundSummary)
    paper_repo.save_trace(db, state.task_id, "summarize_round", "action",
                          round_number=round_num,
                          output_data=result.model_dump())
    db.commit()
    return result.summary, result.knowledge_gaps


async def _generate_report(db, state: ResearchState, llm) -> str:
    from app.db.models import Paper, TaskPaper

    # Gather high-priority papers with numbering and DOI for citation traceability
    high_papers = db.query(Paper).join(TaskPaper).filter(
        TaskPaper.task_id == state.task_id,
        TaskPaper.priority == "high",
    ).limit(20).all()

    high_papers_text = "\n".join(
        f"[P{i+1}] {p.title} ({p.year}) [citations: {p.citation_count}] DOI: {p.doi or 'N/A'}"
        for i, p in enumerate(high_papers)
    )

    # RAG: Retrieve full-text passages as evidence for report
    rag_evidence_text = ""
    try:
        from app.services.rag_service import rag_retrieve
        high_paper_ids = [p.id for p in high_papers]
        rag_results = rag_retrieve(
            query=state.normalized_topic,
            top_k=20,
            paper_ids=high_paper_ids,
        )
        if rag_results:
            evidence_lines = []
            for r in rag_results[:20]:
                clean = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:200].strip()
                evidence_lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean}")
            rag_evidence_text = "\n\n## 论文全文证据段落（RAG检索）\n" + "\n".join(evidence_lines)
            logger.info("Task %s: RAG retrieved %d passages for report", state.task_id[:8], len(rag_results))
    except Exception as e:
        logger.warning("Task %s: RAG retrieval for report failed (non-fatal): %s", state.task_id[:8], e)

    messages = [
        {"role": "system", "content": REPORT_SYSTEM},
        {"role": "user", "content": REPORT_USER.format(
            topic=state.normalized_topic,
            keywords=", ".join(state.keywords),
            round_summaries="\n\n".join(state.round_summaries),
            high_papers=high_papers_text or "(none)",
            gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
        ) + (rag_evidence_text or "")},
    ]
    report_text = await llm.chat(messages, temperature=0.5)

    # P0-A: Self-feedback iteration (max 3 rounds)
    from app.schemas.schemas import ReportFeedback
    from app.agent.prompts import REPORT_FEEDBACK_SYSTEM, REPORT_FEEDBACK_USER, REPORT_REFINE_SYSTEM, REPORT_REFINE_USER

    for refine_round in range(3):
        logger.info("Task %s: report self-feedback round %d", state.task_id[:8], refine_round + 1)
        try:
            feedback = await llm.chat_json([
                {"role": "system", "content": REPORT_FEEDBACK_SYSTEM},
                {"role": "user", "content": REPORT_FEEDBACK_USER.format(
                    topic=state.normalized_topic,
                    report=report_text[:4000],
                    papers=high_papers_text or "(none)",
                )},
            ], ReportFeedback)
        except Exception as e:
            logger.warning("Report feedback failed: %s", e)
            break

        if not feedback.needs_improvement:
            logger.info("Task %s: report passed self-review", state.task_id[:8])
            break

        logger.info("Task %s: report needs improvement: %s", state.task_id[:8], feedback.suggestions[:2])
        try:
            report_text = await llm.chat([
                {"role": "system", "content": REPORT_REFINE_SYSTEM},
                {"role": "user", "content": REPORT_REFINE_USER.format(
                    report=report_text[:4000],
                    feedback="\n".join(feedback.suggestions),
                    papers=high_papers_text or "(none)",
                )},
            ], temperature=0.3)
        except Exception as e:
            logger.warning("Report refine failed: %s", e)
            break

    paper_repo.save_report(db, state.task_id, report_text)
    paper_repo.save_trace(db, state.task_id, "generate_report", "action",
                          output_data={"length": len(report_text)})
    db.commit()
    emit_event(state.task_id, "report_ready", {"length": len(report_text)})
    return report_text


async def _build_paper_clusters(db, state: ResearchState, llm, task_id: str):
    """Cluster ALL papers (not just high-priority) into thematic groups.
    Reference: Idea2Paper's knowledge graph approach."""
    from app.db.models import Paper, TaskPaper
    from app.schemas.schemas import ClusterList
    from app.agent.prompts import CLUSTER_SYSTEM, CLUSTER_USER

    # Get ALL papers (high + medium), not just high — with method extracts
    all_tps = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc()).limit(60).all()

    all_papers = []
    for tp in all_tps:
        p = db.query(Paper).filter(Paper.id == tp.paper_id).first()
        if p:
            all_papers.append((p, tp))

    if len(all_papers) < 5:
        logger.info("Task %s: too few papers (%d) for clustering, skipping", task_id[:8], len(all_papers))
        return None

    # Build paper list with title + abstract + method extract
    papers_text = "\n".join(
        f"- [{p.id}] {p.title}: {(p.abstract or '')[:150]}\n  方法: {tp.summary or 'N/A'}"
        for p, tp in all_papers
    )

    messages = [
        {"role": "system", "content": CLUSTER_SYSTEM},
        {"role": "user", "content": CLUSTER_USER.format(
            topic=state.normalized_topic,
            papers=papers_text,
        )},
    ]

    try:
        cluster_list = await llm.chat_json(messages, ClusterList)
        paper_repo.save_trace(db, task_id, "build_clusters", "action",
                              output_data={"cluster_count": len(cluster_list.clusters),
                                           "cross_gaps": len(cluster_list.cross_cluster_gaps)})
        db.commit()
        logger.info("Task %s: built %d clusters, %d cross-cluster gaps",
                    task_id[:8], len(cluster_list.clusters), len(cluster_list.cross_cluster_gaps))
        return cluster_list
    except Exception as e:
        logger.error("Task %s: clustering failed: %s", task_id[:8], e)
        return None


async def _generate_and_score_ideas(db, state: ResearchState, llm, task_id: str, prev_feedback: str = ""):
    from app.db.models import Paper, TaskPaper
    from app.schemas.schemas import IdeaList, IdeaScore

    # P1: Build paper clusters first — use ALL papers, not just high-priority
    cluster_list = await _build_paper_clusters(db, state, llm, task_id)

    # Get high-priority papers WITH method extracts from TaskPaper
    high_tps = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority == "high",
    ).order_by(TaskPaper.final_score.desc()).limit(30).all()

    high_papers = []
    for tp in high_tps:
        p = db.query(Paper).filter(Paper.id == tp.paper_id).first()
        if p:
            high_papers.append((p, tp))

    # Build paper list with ID + title + abstract + method_extract (from summary)
    valid_paper_ids = {p.id for p, _ in high_papers}
    high_papers_text = "\n".join(
        f"[{p.id}] {p.title}: {(p.abstract or '')[:200]}\n  方法摘要: {tp.summary or 'N/A'}"
        for p, tp in high_papers
    )

    # RAG: Retrieve full-text passages for idea grounding
    rag_evidence = ""
    try:
        from app.services.rag_service import rag_retrieve
        rag_results = rag_retrieve(
            query=state.normalized_topic,
            top_k=15,
            paper_ids=list(valid_paper_ids),
            section_filter=["method", "experiment"],
        )
        if rag_results:
            rag_lines = []
            for r in rag_results[:15]:
                # Truncate text and strip inline markers for prompt context
                clean_text = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:300].strip()
                rag_lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean_text}")
            rag_evidence = "\n\n## 论文全文段落（RAG检索，用于支撑创意技术细节）\n" + "\n".join(rag_lines)
            logger.info("Task %s: RAG retrieved %d passages for idea generation", task_id[:8], len(rag_results))
    except Exception as e:
        logger.warning("Task %s: RAG retrieval failed (non-fatal): %s", task_id[:8], e)

    # Build cluster context (key innovation: structured view of ALL papers)
    cluster_context = ""
    if cluster_list and cluster_list.clusters:
        cluster_lines = []
        for i, c in enumerate(cluster_list.clusters):
            cluster_lines.append(
                f"### 聚类 {i+1}: {c.cluster_name}\n"
                f"- 核心方法: {c.core_method}\n"
                f"- 技术细节: {c.technique_details}\n"
                f"- 解决问题: {c.problem_addressed}\n"
                f"- 关键发现: {c.key_findings}\n"
                f"- 局限性: {c.limitations}\n"
                f"- 代表论文: {', '.join(c.representative_papers[:3])}"
            )
        cluster_context = (
            "## 论文聚类分析（基于全部论文，非仅高优先级）\n\n"
            + "\n\n".join(cluster_lines)
            + "\n\n## 跨聚类机会\n"
            + "\n".join(f"- {g}" for g in cluster_list.cross_cluster_gaps)
        )

    # Get latest report
    from app.db.models import Report
    report = db.query(Report).filter(Report.task_id == task_id).order_by(Report.created_at.desc()).first()
    report_text = report.content_markdown if report else ""

    # Build user prompt with cluster context + papers + optional retry feedback
    user_content = IDEAS_USER.format(
        topic=state.normalized_topic,
        report=report_text[:3000],
        papers=high_papers_text or "(none)",
        gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
    )
    # Add cluster context — this gives LLM a structured view of ALL papers
    if cluster_context:
        user_content += "\n\n" + cluster_context
    if rag_evidence:
        user_content += rag_evidence
    if prev_feedback:
        user_content += "\n\n--- 之前创意反馈 ---\n" + prev_feedback

    messages = [
        {"role": "system", "content": IDEAS_SYSTEM.format(num_ideas=5)},
        {"role": "user", "content": user_content},
    ]
    idea_list = await llm.chat_json(messages, IdeaList)

    # P0-B: Novelty check for each idea
    from app.schemas.schemas import NoveltyCheck
    from app.agent.prompts import NOVELTY_CHECK_SYSTEM, NOVELTY_CHECK_USER

    for item in idea_list.ideas:
        # Validate related_paper_ids — only keep IDs that exist in our paper list
        valid_ids = [pid for pid in item.related_paper_ids if pid in valid_paper_ids]
        invalid_count = len(item.related_paper_ids) - len(valid_ids)
        if invalid_count > 0:
            logger.warning("Idea '%s': %d invalid paper IDs filtered out", item.title[:40], invalid_count)

        # Build titles for display from valid IDs
        id_to_title = {p.id: p.title for p, _ in high_papers}
        valid_titles = [id_to_title.get(pid, "") for pid in valid_ids if id_to_title.get(pid)]

        idea_data = {
            "title": item.title,
            "description": item.description,
            "motivation": item.motivation,
            "method_sketch": item.method_sketch,
            "expected_contribution": item.expected_contribution,
            "related_paper_ids_json": json.dumps(valid_ids, ensure_ascii=False),
        }
        idea = paper_repo.save_idea(db, task_id, idea_data)

        # P0-B: Novelty check — search for similar existing work
        novelty_penalty = 0.0
        try:
            novelty_query = f"{item.title} {item.description[:100]}"
            existing_papers = await search_service.search_all_sources(novelty_query, limit=5)
            existing_text = "\n".join(
                f"- {p.title}: {(p.abstract or '')[:150]}"
                for p in existing_papers[:5]
            ) or "(none found)"

            novelty_result = await llm.chat_json([
                {"role": "system", "content": NOVELTY_CHECK_SYSTEM},
                {"role": "user", "content": NOVELTY_CHECK_USER.format(
                    title=item.title,
                    description=item.description,
                    method=item.method_sketch,
                    existing_papers=existing_text,
                )},
            ], NoveltyCheck)

            if not novelty_result.is_novel:
                novelty_penalty = 0.1  # Penalize non-novel ideas (reduced from 0.2)
                logger.info("Idea '%s' novelty check: NOT NOVEL (similar: %s)",
                           item.title[:40], novelty_result.similar_papers[:2])
            else:
                logger.info("Idea '%s' novelty check: NOVEL", item.title[:40])
        except Exception as e:
            logger.warning("Novelty check failed for idea '%s': %s", item.title[:40], e)

        # Initial scoring
        try:
            score = await _score_idea(db, state, llm, idea)
            # Apply novelty penalty
            adjusted_novelty = max(0, score.novelty - novelty_penalty)
            idea_score_val = (
                0.20 * adjusted_novelty + 0.20 * score.feasibility + 0.20 * score.significance +
                0.20 * score.evidence_support + 0.10 * score.differentiation +
                0.10 * score.experimentability
            )
            final_score = idea_score_val - 0.08 * score.risk

            if final_score >= 0.70:
                decision = "go"
            elif final_score >= 0.50:
                decision = "revise"
            else:
                decision = "reject"

            paper_repo.update_idea_scores(db, idea.id, score.model_dump(), final_score, decision)
        except Exception as e:
            logger.error("Failed to score idea %s: %s", idea.id, e)

        emit_event(task_id, "idea_generated", {"id": idea.id, "title": idea.title})

    paper_repo.save_trace(db, task_id, "generate_ideas", "action",
                          output_data={"count": len(idea_list.ideas)})
    db.commit()


async def _score_idea(db, state: ResearchState, llm, idea) -> IdeaScore:
    from app.schemas.schemas import IdeaScore
    messages = [
        {"role": "system", "content": IDEA_SCORE_SYSTEM},
        {"role": "user", "content": IDEA_SCORE_USER.format(
            topic=state.normalized_topic,
            title=idea.title or "",
            description=idea.description or "",
            motivation=idea.motivation or "",
            method=idea.method_sketch or "",
            contribution=idea.expected_contribution or "",
            related_papers=idea.related_paper_ids_json or "(none)",
        )},
    ]
    return await llm.chat_json(messages, IdeaScore)

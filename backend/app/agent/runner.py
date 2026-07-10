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
                _TP.priority.in_(["high", "medium"]),
            ).all()

            if high_papers_for_rag:
                logger.info("Task %s: RAG indexing %d high-priority papers...", task_id[:8], len(high_papers_for_rag))
                emit_event(task_id, "status", {"status": "indexing_pdfs", "total": len(high_papers_for_rag)})
                rag_summary = await fetch_and_index_papers(high_papers_for_rag, llm, task_id)
                logger.info("Task %s: RAG indexing done: %s", task_id[:8], rag_summary)
        except Exception as e:
            logger.warning("Task %s: RAG indexing failed (non-fatal, continuing with abstracts): %s", task_id[:8], e)

        # 2.6. LLM Wiki: Ingest papers into wiki knowledge base (replaces GraphRAG)
        try:
            from app.services.wiki_service import ingest_papers_to_wiki
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
                    from app.services.wiki_service import lint_wiki
                    lint_result = await lint_wiki(db, task_id, llm)
                    logger.info("Task %s: wiki lint found %d issues", task_id[:8], lint_result.get("total", 0))
                except Exception as lint_e:
                    logger.warning("Task %s: wiki lint failed (non-fatal): %s", task_id[:8], lint_e)
        except Exception as e:
            logger.warning("Task %s: LLM Wiki ingest failed (non-fatal, falling back to LLM clustering): %s", task_id[:8], e)

        # 3. Build clusters once (used by both report and ideas)
        logger.info("Task %s: building clusters...", task_id[:8])
        cluster_list = await _build_paper_clusters(db, state, llm, task_id)

        # 4. Generate report ONCE (not in retry loop)
        logger.info("Task %s: generating report...", task_id[:8])
        task_repo.update_status(db, task_id, "reporting")
        db.commit()
        emit_event(task_id, "status", {"status": "reporting"})
        report_markdown = await _generate_report(db, state, llm, cluster_list)
        logger.info("Task %s: report generated (%d chars)", task_id[:8], len(report_markdown))

        # 5. Ideas generation loop (retry if no qualified ideas — report is NOT regenerated)
        from app.db.models import ResearchIdea
        max_idea_rounds = 3  # Max attempts to generate qualified ideas

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
            await _generate_and_score_ideas(db, state, llm, task_id, prev_ideas_feedback, cluster_list)

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

                # Idea retry: RAG index + Wiki ingest newly found high-priority papers
                if new_paper_ids:
                    try:
                        from app.services.rag_service import fetch_and_index_papers
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

                            # Wiki ingest new papers (incremental update)
                            from app.services.wiki_service import ingest_papers_to_wiki
                            await ingest_papers_to_wiki(db, new_high_papers, llm, task_id)
                            logger.info("Task %s: idea retry — wiki updated with new papers", task_id[:8])
                    except Exception as retry_e:
                        logger.warning("Task %s: idea retry RAG/Wiki failed (non-fatal): %s",
                                      task_id[:8], retry_e)
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

        # Get wiki context for experiment generation (verified methods/datasets)
        wiki_ctx = ""
        try:
            from app.services.wiki_service import get_wiki_context
            wiki_ctx = get_wiki_context(db, task_id, page_types=["method", "dataset", "model"], max_chars=8000)
            if wiki_ctx:
                logger.info("Task %s: wiki context loaded for experiments (%d chars)", task_id[:8], len(wiki_ctx))
        except Exception as e:
            logger.warning("Task %s: wiki context for experiments failed (non-fatal): %s", task_id[:8], e)

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
                    wiki_context=wiki_ctx or "(no wiki available)",
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


async def _generate_report(db, state: ResearchState, llm, cluster_list=None) -> str:
    """Two-step report generation (STORM-style: outline → fill section by section).

    Step 1: Generate a structured outline based on clusters + papers.
    Step 2: For each section, select relevant papers + RAG evidence, generate content.
    Step 3: Assemble all sections + references.
    """
    from app.db.models import Paper, TaskPaper
    from app.schemas.schemas import ReportOutline
    from app.agent.prompts import (REPORT_OUTLINE_SYSTEM, REPORT_OUTLINE_USER,
                                   REPORT_SECTION_SYSTEM, REPORT_SECTION_USER)

    # Gather high + medium priority papers for comprehensive coverage
    all_papers = db.query(Paper).join(TaskPaper).filter(
        TaskPaper.task_id == state.task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(50).all()

    if not all_papers:
        logger.warning("Task %s: no papers for report", state.task_id[:8])
        paper_repo.save_report(db, state.task_id, "无可用论文，无法生成报告。")
        db.commit()
        return "无可用论文，无法生成报告。"

    # Build papers text with [P1], [P2] numbering
    papers_text = "\n".join(
        f"[P{i+1}] {p.title} ({p.year}) [{p.venue or 'N/A'}] [citations: {p.citation_count}] DOI: {p.doi or 'N/A'}\n"
        f"  摘要: {(p.abstract or 'N/A')[:600]}"
        for i, p in enumerate(all_papers)
    )
    paper_id_by_index = {i + 1: p.id for i, p in enumerate(all_papers)}
    paper_by_index = {i + 1: p for i, p in enumerate(all_papers)}

    # Build clusters text from in-memory cluster_list (passed from caller)
    clusters_text = ""
    if cluster_list and cluster_list.clusters:
        cluster_lines = []
        for i, c in enumerate(cluster_list.clusters):
            # Match representative_papers (titles) to [Px] numbering
            paper_nums = []
            for title in c.representative_papers:
                for idx, p in enumerate(all_papers):
                    if title and p.title and title.lower() in p.title.lower():
                        paper_nums.append(f"[P{idx+1}]")
                        break
            cluster_lines.append(
                f"聚类{i+1}: {c.cluster_name or '未命名'}\n"
                f"  核心方法: {c.core_method or 'N/A'}\n"
                f"  论文: {', '.join(paper_nums) if paper_nums else '(未分配)'}\n"
                f"  局限: {c.limitations or 'N/A'}"
            )
        clusters_text = "\n".join(cluster_lines)

    # Get wiki context (pre-compiled knowledge from LLM Wiki)
    wiki_context_text = ""
    try:
        from app.services.wiki_service import get_wiki_context
        wiki_context_text = get_wiki_context(db, state.task_id)
        if wiki_context_text:
            logger.info("Task %s: wiki context loaded for report (%d chars)",
                       state.task_id[:8], len(wiki_context_text))
    except Exception as e:
        logger.warning("Task %s: wiki context retrieval failed (non-fatal): %s", state.task_id[:8], e)

    # === Step 1: Generate outline ===
    logger.info("Task %s: generating report outline (step 1/2)...", state.task_id[:8])
    try:
        outline = await llm.chat_json([
            {"role": "system", "content": REPORT_OUTLINE_SYSTEM},
            {"role": "user", "content": REPORT_OUTLINE_USER.format(
                topic=state.normalized_topic,
                clusters_text=clusters_text or "(无聚类信息)",
                papers_text=papers_text,
                round_summaries="\n\n".join(state.round_summaries[-3:]),
                gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
            ) + ("\n\n## 知识库 Wiki（预编译论文合成）\n" + wiki_context_text if wiki_context_text else "")},
        ], ReportOutline)
        logger.info("Task %s: outline generated with %d sections", state.task_id[:8], len(outline.sections))
    except Exception as e:
        logger.warning("Task %s: outline generation failed, falling back to one-shot: %s", state.task_id[:8], e)
        # Fallback: one-shot generation
        outline = None

    if outline and outline.sections:
        # === Step 2: Fill each section ===
        logger.info("Task %s: filling %d sections (step 2/2)...", state.task_id[:8], len(outline.sections))

        # RAG: Pre-retrieve evidence for all papers (one call, reused per section)
        all_paper_ids = [p.id for p in all_papers]
        rag_evidence_global = ""
        try:
            from app.services.rag_service import rag_retrieve
            rag_results = rag_retrieve(
                query=state.normalized_topic,
                top_k=50,
                paper_ids=all_paper_ids,
            )
            if rag_results:
                evidence_lines = []
                for r in rag_results[:50]:
                    clean = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:400].strip()
                    evidence_lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean}")
                rag_evidence_global = "\n".join(evidence_lines)
                logger.info("Task %s: RAG retrieved %d passages for report", state.task_id[:8], len(rag_results))
        except Exception as e:
            logger.warning("Task %s: RAG retrieval for report failed (non-fatal): %s", state.task_id[:8], e)

        # Generate each section (with concurrency limit)
        semaphore = asyncio.Semaphore(3)

        async def generate_one_section(section, section_idx):
            async with semaphore:
                # Select papers for this section
                section_paper_indices = section.paper_indices or []
                # If no indices specified, use all papers (fallback)
                if not section_paper_indices:
                    section_paper_indices = list(range(1, len(all_papers) + 1))

                # Build section-specific paper text
                section_papers = []
                section_paper_ids = []
                for idx in section_paper_indices:
                    p = paper_by_index.get(idx)
                    if p:
                        section_papers.append(
                            f"[P{idx}] {p.title} ({p.year}) [{p.venue or 'N/A'}] [citations: {p.citation_count}]\n"
                            f"  摘要: {(p.abstract or 'N/A')[:600]}"
                        )
                        section_paper_ids.append(p.id)

                # RAG evidence specific to this section's papers
                section_rag = ""
                try:
                    from app.services.rag_service import rag_retrieve
                    if section_paper_ids:
                        section_rag_results = rag_retrieve(
                            query=f"{state.normalized_topic} {section.title}",
                            top_k=15,
                            paper_ids=section_paper_ids,
                        )
                        if section_rag_results:
                            _fig_pat = re.compile(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]')
                            section_rag = "\n".join(
                                f"[{r['paper_id'][:8]}] ({r['section']}) "
                                f"{_fig_pat.sub('', r['text'])[:400].strip()}"
                                for r in section_rag_results[:15]
                            )
                except Exception:
                    pass  # Non-fatal, use global evidence

                try:
                    content = await llm.chat([
                        {"role": "system", "content": REPORT_SECTION_SYSTEM},
                        {"role": "user", "content": REPORT_SECTION_USER.format(
                            topic=state.normalized_topic,
                            section_title=section.title,
                            section_description=section.description or "",
                            section_papers="\n".join(section_papers) or "(none)",
                            rag_evidence=section_rag or rag_evidence_global or "(none)",
                            round_summaries="\n\n".join(state.round_summaries[-2:]),
                        ) + ("\n\n## 知识库 Wiki（预编译论文合成）\n" + wiki_context_text if wiki_context_text else "")},
                    ], temperature=0.4)
                    logger.info("Task %s: section %d/%d '%s' generated (%d chars)",
                               state.task_id[:8], section_idx + 1, len(outline.sections),
                               section.title[:30], len(content))
                    return content
                except Exception as e:
                    logger.warning("Task %s: section '%s' generation failed: %s",
                                  state.task_id[:8], section.title[:30], e)
                    return f"## {section.title}\n\n(本节生成失败)"

        # Run sections in parallel
        section_tasks = [
            generate_one_section(section, i)
            for i, section in enumerate(outline.sections)
        ]
        section_contents = await asyncio.gather(*section_tasks)

        # === Step 3: Assemble + add references ===
        report_text = "\n\n".join(section_contents)

        # Add references section
        ref_lines = ["## 参考文献\n"]
        for i, p in enumerate(all_papers):
            ref_lines.append(f"[P{i+1}] {p.title} ({p.year}). DOI: {p.doi or 'N/A'}")
        report_text += "\n\n" + "\n".join(ref_lines)

    else:
        # Fallback: one-shot generation (old approach)
        logger.info("Task %s: using one-shot report generation (fallback)", state.task_id[:8])
        from app.agent.prompts import REPORT_SYSTEM, REPORT_USER
        rag_evidence_text = ""
        try:
            from app.services.rag_service import rag_retrieve
            rag_results = rag_retrieve(
                query=state.normalized_topic,
                top_k=30,
                paper_ids=[p.id for p in all_papers],
            )
            if rag_results:
                evidence_lines = []
                for r in rag_results[:30]:
                    clean = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:500].strip()
                    evidence_lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean}")
                rag_evidence_text = "\n\n## 论文全文证据段落（RAG检索）\n" + "\n".join(evidence_lines)
        except Exception as e:
            logger.warning("Task %s: RAG retrieval for report failed: %s", state.task_id[:8], e)

        messages = [
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": REPORT_USER.format(
                topic=state.normalized_topic,
                keywords=", ".join(state.keywords),
                round_summaries="\n\n".join(state.round_summaries),
                high_papers=papers_text or "(none)",
                gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
            ) + (rag_evidence_text or "") + ("\n\n## 知识库 Wiki（预编译论文合成）\n" + wiki_context_text if wiki_context_text else "")},
        ]
        report_text = await llm.chat(messages, temperature=0.5)

    # Post-check: detect placeholder text
    import re as _re
    placeholder_patterns = [
        r'（.*?保持不变.*?）', r'（.*?保持原.*?）', r'（.*?新增内容.*?）',
        r'（.*?此部分.*?）', r'（.*?其余.*?不变.*?）', r'（.*?省略.*?）',
        r'（.*?详见.*?）', r'（.*?同上.*?）', r'（.*?参见.*?）',
        r'（.*?不?变.*?）', r'（.*?未变.*?）',
        r'\(.*?remain.*?unchanged.*?\)', r'\(.*?see above.*?\)', r'\(.*?omitted.*?\)',
        r'^（.*?补充.*?）$', r'^（.*?完善.*?）$',
    ]
    has_placeholders = any(_re.search(p, report_text) for p in placeholder_patterns)
    if has_placeholders:
        logger.warning("Task %s: report contains placeholder text", state.task_id[:8])

    paper_repo.save_report(db, state.task_id, report_text)
    paper_repo.save_trace(db, state.task_id, "generate_report", "action",
                          output_data={"length": len(report_text),
                                       "method": "two_step" if outline else "one_shot",
                                       "sections": len(outline.sections) if outline else 0})
    db.commit()
    emit_event(state.task_id, "report_ready", {"length": len(report_text)})
    return report_text


async def _build_paper_clusters(db, state: ResearchState, llm, task_id: str):
    """Cluster ALL papers into thematic groups.
    
    Primary: Use pre-compiled LLM Wiki concept pages (replaces GraphRAG community detection).
    Fallback: LLM-based clustering if wiki is not yet built.
    """
    from app.db.models import Paper, TaskPaper
    from app.schemas.schemas import ClusterList

    # === Primary: Get clusters from wiki ===
    try:
        from app.services.wiki_service import get_wiki_clusters
        wiki_clusters = get_wiki_clusters(db, task_id)
        if wiki_clusters and wiki_clusters.clusters:
            logger.info("Task %s: using wiki clusters (%d concept pages, %d cross-gaps)",
                       task_id[:8], len(wiki_clusters.clusters), len(wiki_clusters.cross_cluster_gaps))
            paper_repo.save_trace(db, task_id, "build_clusters", "action",
                                  output_data={"source": "wiki",
                                               "cluster_count": len(wiki_clusters.clusters),
                                               "cross_gaps": len(wiki_clusters.cross_cluster_gaps)})
            db.commit()
            return wiki_clusters
    except Exception as e:
        logger.warning("Task %s: wiki cluster retrieval failed, falling back to LLM: %s", task_id[:8], e)

    # === Fallback: LLM-based clustering ===
    from app.agent.prompts import CLUSTER_SYSTEM, CLUSTER_USER

    all_tps = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(80).all()

    all_papers = []
    for tp in all_tps:
        p = db.query(Paper).filter(Paper.id == tp.paper_id).first()
        if p:
            all_papers.append((p, tp))

    if len(all_papers) < 5:
        logger.info("Task %s: too few papers (%d) for clustering, skipping", task_id[:8], len(all_papers))
        return None

    papers_text = "\n".join(
        f"- [{p.id}] {p.title} ({p.year}) [{p.venue or 'N/A'}]: {(p.abstract or '')[:300]}\n  方法: {tp.summary or 'N/A'}"
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
                              output_data={"source": "llm_fallback",
                                           "cluster_count": len(cluster_list.clusters),
                                           "cross_gaps": len(cluster_list.cross_cluster_gaps)})
        db.commit()
        logger.info("Task %s: built %d clusters (LLM fallback), %d cross-cluster gaps",
                    task_id[:8], len(cluster_list.clusters), len(cluster_list.cross_cluster_gaps))
        return cluster_list
    except Exception as e:
        logger.error("Task %s: clustering failed: %s", task_id[:8], e)
        return None


async def _generate_and_score_ideas(db, state: ResearchState, llm, task_id: str, prev_feedback: str = "", cluster_list=None):
    from app.db.models import Paper, TaskPaper
    from app.schemas.schemas import IdeaList, IdeaScore

    # P1: Build paper clusters first — use ALL papers, not just high-priority
    if cluster_list is None:
        cluster_list = await _build_paper_clusters(db, state, llm, task_id)

    # Get high + medium priority papers for idea generation (not just high)
    all_tps = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(50).all()

    high_papers = []
    for tp in all_tps:
        p = db.query(Paper).filter(Paper.id == tp.paper_id).first()
        if p:
            high_papers.append((p, tp))

    # Build paper list with ID + title + abstract + method_extract (from summary)
    valid_paper_ids = {p.id for p, _ in high_papers}
    high_papers_text = "\n".join(
        f"[{p.id}] {p.title} ({p.year}) [{p.venue or 'N/A'}]: {(p.abstract or '')[:500]}\n"
        f"  方法摘要: {tp.summary or 'N/A'}"
        for p, tp in high_papers
    )

    # RAG: Retrieve full-text passages for idea grounding
    rag_evidence = ""
    try:
        from app.services.rag_service import rag_retrieve
        rag_results = rag_retrieve(
            query=state.normalized_topic,
            top_k=25,
            paper_ids=list(valid_paper_ids),
            section_filter=["method", "experiment"],
        )
        if rag_results:
            rag_lines = []
            for r in rag_results[:25]:
                # Truncate text and strip inline markers for prompt context
                clean_text = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:600].strip()
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
    # Add wiki context (pre-compiled knowledge: methods, datasets, concepts, synthesis)
    # P1-3: For idea generation, prioritize method/dataset/synthesis pages (anti-hallucination)
    #       while still including concept pages for theme awareness
    try:
        from app.services.wiki_service import get_wiki_context
        wiki_ctx = get_wiki_context(
            db, task_id,
            page_types=["method", "dataset", "model", "synthesis", "concept"],
            max_chars=12000,
        )
        if wiki_ctx:
            user_content += "\n\n" + wiki_ctx
            logger.info("Task %s: wiki context added to idea generation (%d chars)",
                       task_id[:8], len(wiki_ctx))
    except Exception as e:
        logger.warning("Task %s: wiki context for ideas failed (non-fatal): %s", task_id[:8], e)
    if prev_feedback:
        user_content += "\n\n--- 之前创意反馈 ---\n" + prev_feedback

    messages = [
        {"role": "system", "content": IDEAS_SYSTEM.format(num_ideas=5)},
        {"role": "user", "content": user_content},
    ]
    idea_list = await llm.chat_json(messages, IdeaList)

    # === Idea validation: dedup + baseline check + metric-hypothesis check ===
    validation_penalties: dict[int, float] = {}  # idea index -> penalty
    ideas_to_skip: set[int] = set()  # duplicate ideas to skip

    # MARKER: This trace proves the validation code is running
    paper_repo.save_trace(db, task_id, "idea_validation_start", "action",
                          output_data={"ideas_count": len(idea_list.ideas)})
    db.commit()

    # Step 1: LLM-based validation (dedup + metric-hypothesis check)
    try:
        from app.schemas.schemas import IdeaValidationList
        from app.agent.prompts import IDEA_VALIDATION_SYSTEM, IDEA_VALIDATION_USER

        paper_titles = [p.title for p, _ in high_papers][:50]

        ideas_text = "\n\n".join(
            f"### Idea {i+1}: {item.title}\n"
            f"- 描述: {item.description}\n"
            f"- 方法: {item.method_sketch}\n"
            for i, item in enumerate(idea_list.ideas)
        )

        validation_result = await llm.chat_json([
            {"role": "system", "content": IDEA_VALIDATION_SYSTEM},
            {"role": "user", "content": IDEA_VALIDATION_USER.format(
                topic=state.normalized_topic,
                paper_titles="\n".join(f"- {t}" for t in paper_titles),
                ideas_text=ideas_text,
            )},
        ], IdeaValidationList)

        for i, v in enumerate(validation_result.validations):
            if v.is_duplicate:
                ideas_to_skip.add(i)
                logger.info("Task %s: idea %d '%s' skipped as duplicate of '%s'",
                           task_id[:8], i+1, idea_list.ideas[i].title[:40], v.duplicate_of)
            if v.has_issues and v.metric_issues:
                penalty = min(0.15, 0.05 * len(v.metric_issues))
                validation_penalties[i] = validation_penalties.get(i, 0) + penalty
                logger.warning("Task %s: idea %d metric issues: %s",
                             task_id[:8], i+1, v.metric_issues)
        paper_repo.save_trace(db, task_id, "idea_validation_llm", "observation",
                              output_data={"duplicates": len(ideas_to_skip),
                                           "metric_issues": sum(len(v.metric_issues) for v in validation_result.validations if v.has_issues)})
        db.commit()
    except Exception as e:
        logger.warning("Task %s: LLM validation failed (non-fatal): %s", task_id[:8], e)
        paper_repo.save_trace(db, task_id, "idea_validation_llm", "observation",
                              output_data={"status": "failed", "error": str(e)[:200]})
        db.commit()

    # Step 2: Search-based baseline verification (external truth source)
    # Extract baseline names from each idea's method_sketch and verify via S2 search
    verified_baselines: set[str] = set()  # Available for post-enrichment check
    KNOWN_REAL_BASELINES = {
        "memgpt", "rag", "bert", "gpt-4", "gpt-3", "gpt-3.5", "llama", "llama-2",
        "llama-3", "t5", "transformer", "fine-tuned llm", "standard llm",
        "chain-of-thought", "few-shot", "zero-shot", "lora", "rlhf", "dpo",
        "bleu", "rouge", "f1", "accuracy", "precision", "recall",
        "memorybank", "standard memory", "no memory", "standard dialogue system",
        "standard multimodal", "standard multi-agent",
        # Chinese generic terms that are real but not specific method names
        "标准对话系统", "标准多智能体", "标准记忆机制", "标准记忆", "标准方法",
        "标准多模态", "标准LLM", "标准LLM无记忆", "无记忆", "标准基线",
        "标准对话", "标准多智能体协作", "标准框架",
    }
    try:

        # Collect all baselines to verify — use list to handle same name across multiple ideas
        baselines_to_check: list[tuple[str, int]] = []  # (baseline_name, idea_index)
        for idx, item in enumerate(idea_list.ideas):
            if idx in ideas_to_skip:
                continue
            # Extract baseline names from "具体基线:" line
            method = item.method_sketch or ""
            baseline_match = re.search(r'基线[：:]\s*(.+?)(?:\n|$)', method)
            if baseline_match:
                baseline_text = baseline_match.group(1)
                # Split by common delimiters including 和/与
                raw_names = re.split(r'[、,，;；和与]\s*', baseline_text)
                for name in raw_names:
                    name = name.strip().rstrip('。.')
                    # Remove leading comparison phrases
                    name = re.sub(r'^(比较|对比|与|和)\s*', '', name).strip()
                    # Remove trailing phrases: 进行对比/对比/的性能/等/框架/系统/基线/等基线
                    name = re.sub(r'(进行对比|对比|的性能|等基线|等|基线|框架|系统)$', '', name).strip()
                    # Remove trailing parenthetical descriptions: "EMA2（基于...）" → "EMA2"
                    name = re.sub(r'[（(].*?[)）]\s*$', '', name).strip()
                    # Skip generic terms and known real baselines
                    if len(name) > 2 and name.lower() not in KNOWN_REAL_BASELINES:
                        # Also skip Chinese generic terms starting with 标准/普通/基础/传统
                        if not re.match(r'^(标准|普通|基础|传统|常规|一般)', name):
                            baselines_to_check.append((name, idx))

        # Deduplicate for search, but keep all (name, idea_idx) pairs for penalty
        unique_names = set(name for name, _ in baselines_to_check)
        logger.info("Task %s: verifying %d unique baseline names via search",
                    task_id[:8], len(unique_names))

        # Verify each baseline via paper database + S2 search
        verified_baselines: set[str] = set()
        fabricated_baselines: list[tuple[str, int]] = []  # (name, idea_index)

        # First check: does the baseline name appear in our paper database?
        # Check both paper titles and abstracts for method name mentions
        all_paper_abstracts_lower = [(p.title.lower(), (p.abstract or "").lower()) for p, _ in high_papers]

        for name, idea_idx in baselines_to_check:
            name_lower = name.lower()
            found_in_db = any(
                name_lower in title or name_lower in abstract
                for title, abstract in all_paper_abstracts_lower
            )
            if found_in_db:
                verified_baselines.add(name)
            # else: needs search verification (below)

        # Collect names that need search verification
        names_to_search = {name for name, _ in baselines_to_check if name not in verified_baselines}

        # Second check: search for unverified baselines (S2 + Crossref fallback)
        semaphore = asyncio.Semaphore(3)
        async def verify_one_baseline(name: str) -> tuple[str, bool]:
            if name in verified_baselines:
                return name, True
            async with semaphore:
                name_lower = name.lower()
                try:
                    results = await search_service.search_all_sources(name, limit=5)
                    if not results or len(results) == 0:
                        # S2 may be rate-limited — try Crossref directly
                        try:
                            from app.paper_sources.crossref_source import CrossrefSource
                            cr = CrossrefSource()
                            cr_results = await cr.search(name, limit=5)
                            if cr_results:
                                for r in cr_results[:5]:
                                    if name_lower in (r.title or "").lower():
                                        return name, True
                        except Exception:
                            pass
                        return name, False  # No results at all → likely fabricated
                    # STRICT check: baseline name must appear in at least one result title
                    check_limit = 3 if len(name) < 5 else 5
                    for r in results[:check_limit]:
                        title_lower = (r.title or "").lower()
                        if name_lower in title_lower:
                            return name, True
                    # Results exist but name NOT in any title → suspicious
                    return name, False
                except Exception:
                    # Search API failed (rate limit, timeout, etc.) — mark as unverified
                    # Do NOT give benefit of doubt (previous bug: returned True)
                    return name, False

        if names_to_search:
            tasks_verify = [verify_one_baseline(name) for name in names_to_search]
            verify_results = await asyncio.gather(*tasks_verify)
            for name, is_real in verify_results:
                if is_real:
                    verified_baselines.add(name)

        # Build fabricated baselines list (name not verified for this idea)
        for name, idea_idx in baselines_to_check:
            if name not in verified_baselines:
                fabricated_baselines.append((name, idea_idx))

        # Apply penalties for fabricated baselines
        idea_fabricated_count: dict[int, list[str]] = {}
        for name, idea_idx in fabricated_baselines:
            idea_fabricated_count.setdefault(idea_idx, []).append(name)

        for idea_idx, fake_names in idea_fabricated_count.items():
            penalty = min(0.3, 0.15 * len(fake_names))
            validation_penalties[idea_idx] = validation_penalties.get(idea_idx, 0) + penalty
            logger.warning("Task %s: idea %d '%s' has %d FABRICATED baselines: %s (penalty: +%.2f)",
                          task_id[:8], idea_idx+1, idea_list.ideas[idea_idx].title[:40],
                          len(fake_names), fake_names, penalty)

        logger.info("Task %s: baseline verification done — %d verified, %d fabricated",
                    task_id[:8], len(verified_baselines), len(fabricated_baselines))
        paper_repo.save_trace(db, task_id, "idea_baseline_verification", "observation",
                              output_data={"verified": len(verified_baselines),
                                           "fabricated": len(fabricated_baselines),
                                           "fabricated_names": [n for n, _ in fabricated_baselines]})
        db.commit()

        # === P1-2: Dataset verification ===
        KNOWN_DATASETS = {
            "multwoz", "mmlu", "glue", "ms coco", "mscoco", "wikitext-103", "wikitext",
            "clevrer", "squad", "squad 2.0", "natural questions", "trivia qa", "triviaqa",
            "hotpotqa", "wikipedia", "cnn/dailymail", "cnn dailymail", "xsum", "wmt",
            "multiwoz", "babi", "dialogue", "persona-chat", "personachat",
            "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "fashion-mnist",
            "openbookqa", "arc", "hellaswag", "winogrande", "gsm8k", "math",
            "human eval", "humaneval", "mbpp", "codecontests",
        }
        datasets_to_check: list[tuple[str, int]] = []
        for idx, item in enumerate(idea_list.ideas):
            if idx in ideas_to_skip:
                continue
            method = item.method_sketch or ""
            ds_match = re.search(r'数据集[：:]\s*(.+?)(?:\n|$)', method)
            if ds_match:
                ds_text = ds_match.group(1)
                raw_names = re.split(r'[、,，;；和与]\s*', ds_text)
                for name in raw_names:
                    name = name.strip().rstrip('。.（）()')
                    # Remove descriptors like "格式的发票数据集"
                    name = re.sub(r'格式的.*$', '', name).strip()
                    if len(name) > 2:
                        datasets_to_check.append((name, idx))

        fabricated_datasets: list[tuple[str, int]] = []
        for ds_name, idea_idx in datasets_to_check:
            ds_lower = ds_name.lower()
            # Check if in known datasets
            if ds_lower in KNOWN_DATASETS:
                continue
            # Check if mentioned in our paper database
            found_in_db = any(
                ds_lower in title or ds_lower in abstract
                for title, abstract in all_paper_abstracts_lower
            )
            if not found_in_db:
                fabricated_datasets.append((ds_name, idea_idx))

        for ds_name, idea_idx in fabricated_datasets:
            penalty = 0.05  # Smaller penalty than baselines
            validation_penalties[idea_idx] = validation_penalties.get(idea_idx, 0) + penalty
            logger.warning("Task %s: idea %d has SUSPICIOUS dataset: %s (penalty: +%.2f)",
                          task_id[:8], idea_idx+1, ds_name, penalty)

        if fabricated_datasets:
            paper_repo.save_trace(db, task_id, "idea_dataset_verification", "observation",
                                  output_data={"suspicious_datasets": [n for n, _ in fabricated_datasets]})
            db.commit()

    except Exception as e:
        logger.warning("Task %s: baseline search verification failed (non-fatal): %s", task_id[:8], e)
        paper_repo.save_trace(db, task_id, "idea_baseline_verification", "observation",
                              output_data={"status": "failed", "error": str(e)[:200]})
        db.commit()

    # P0-B: Novelty check for each idea
    from app.schemas.schemas import NoveltyCheck
    from app.agent.prompts import NOVELTY_CHECK_SYSTEM, NOVELTY_CHECK_USER

    for idx, item in enumerate(idea_list.ideas):
        # Skip duplicate ideas
        if idx in ideas_to_skip:
            logger.info("Task %s: skipping duplicate idea '%s'", task_id[:8], item.title[:40])
            continue

        # Validate related_paper_ids — only keep IDs that exist in our paper list
        valid_ids = [pid for pid in item.related_paper_ids if pid in valid_paper_ids]
        invalid_count = len(item.related_paper_ids) - len(valid_ids)
        if invalid_count > 0:
            logger.warning("Idea '%s': %d invalid paper IDs filtered out", item.title[:40], invalid_count)

        # Build titles for display from valid IDs
        id_to_title = {p.id: p.title for p, _ in high_papers}
        valid_titles = [id_to_title.get(pid, "") for pid in valid_ids if id_to_title.get(pid)]

        # Two-step: Enrich method_sketch with per-idea RAG retrieval
        enriched_method = item.method_sketch
        try:
            from app.services.rag_service import rag_retrieve
            from app.agent.prompts import IDEA_METHOD_ENRICH_SYSTEM, IDEA_METHOD_ENRICH_USER

            # RAG retrieve specifically for this idea's topic
            idea_query = f"{item.title} {item.description[:200]}"
            idea_rag_results = rag_retrieve(
                query=idea_query,
                top_k=10,
                paper_ids=list(valid_paper_ids),
                section_filter=["method", "experiment"],
            )
            if idea_rag_results:
                # Pre-clean text to avoid backslash in f-string
                _figure_pattern = re.compile(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]')
                rag_passages = "\n\n".join(
                    f"[{r['paper_id'][:8]}] ({r['section']}) "
                    f"{_figure_pattern.sub('', r['text'])[:600].strip()}"
                    for r in idea_rag_results[:10]
                )
                # Build papers summary for related papers
                id_to_abstract = {p.id: (p.abstract or '')[:200] for p, _ in high_papers}
                related_papers_summary = "\n".join(
                    f"- {id_to_title.get(pid, 'unknown')}: {id_to_abstract.get(pid, '')}"
                    for pid in valid_ids[:5]
                )

                enriched_method = await llm.chat([
                    {"role": "system", "content": IDEA_METHOD_ENRICH_SYSTEM},
                    {"role": "user", "content": IDEA_METHOD_ENRICH_USER.format(
                        topic=state.normalized_topic,
                        title=item.title,
                        description=item.description,
                        motivation=item.motivation,
                        rag_passages=rag_passages,
                        papers_summary=related_papers_summary or "(none)",
                    )},
                ], temperature=0.4)
                logger.info("Task %s: enriched method_sketch for idea '%s' (%d -> %d chars)",
                           task_id[:8], item.title[:40], len(item.method_sketch), len(enriched_method))

            # Post-enrichment baseline check: verify baselines in the enriched method
            enriched_baseline_match = re.search(r'基线[：:]\s*(.+?)(?:\n|$)', enriched_method)
            if enriched_baseline_match:
                enriched_baselines = re.split(r'[、,，;；和与]\s*', enriched_baseline_match.group(1))
                new_fabricated = []
                for bname in enriched_baselines:
                    bname = re.sub(r'^(比较|对比|与|和)\s*', '', bname.strip().rstrip('。.'))
                    bname = re.sub(r'(进行对比|对比|的性能|等基线|等|基线|框架|系统)$', '', bname).strip()
                    bname = re.sub(r'[（(].*?[)）]\s*$', '', bname).strip()
                    if len(bname) > 2 and bname.lower() not in KNOWN_REAL_BASELINES and bname not in verified_baselines:
                        if not re.match(r'^(标准|普通|基础|传统|常规|一般)', bname):
                            new_fabricated.append(bname)

                if new_fabricated:
                    # Quick search for new baselines
                    async def verify_new(name):
                        try:
                            results = await search_service.search_all_sources(name, limit=3)
                            if not results:
                                return name, False
                            for r in results[:3]:
                                if name.lower() in (r.title or "").lower():
                                    return name, True
                            return name, False
                        except Exception:
                            return name, True

                    new_results = await asyncio.gather(*[verify_new(n) for n in new_fabricated])
                    confirmed_fake = [n for n, real in new_results if not real]
                    if confirmed_fake:
                        extra_penalty = min(0.2, 0.1 * len(confirmed_fake))
                        validation_penalties[idx] = validation_penalties.get(idx, 0) + extra_penalty
                        logger.warning("Task %s: idea %d enriched method has %d FABRICATED baselines: %s (penalty: +%.2f)",
                                      task_id[:8], idx+1, len(confirmed_fake), confirmed_fake, extra_penalty)

        except Exception as e:
            logger.warning("Task %s: method enrichment failed for idea '%s' (using original): %s",
                          task_id[:8], item.title[:40], e)

        idea_data = {
            "title": item.title,
            "description": item.description,
            "motivation": item.motivation,
            "method_sketch": enriched_method,
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
                    method=enriched_method,
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
            # Apply validation penalty (baseline issues + metric issues)
            val_penalty = validation_penalties.get(idx, 0.0)
            if val_penalty > 0:
                final_score = max(0, final_score - val_penalty)
                logger.info("Idea '%s': applied validation penalty %.2f, final_score=%.3f",
                           item.title[:40], val_penalty, final_score)

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

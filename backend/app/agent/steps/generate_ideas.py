"""Step: Generate and score research ideas with 5-layer validation."""

import asyncio
import json
import logging
import re

from app.agent.state import ResearchState
from app.agent.prompts import (
    IDEAS_SYSTEM, IDEAS_USER,
    IDEA_SCORE_SYSTEM, IDEA_SCORE_USER,
    NOVELTY_CHECK_SYSTEM, NOVELTY_CHECK_USER,
    IDEA_VALIDATION_SYSTEM, IDEA_VALIDATION_USER,
    IDEA_METHOD_ENRICH_SYSTEM, IDEA_METHOD_ENRICH_USER,
)
from app.db.models import Paper, TaskPaper, Report
from app.db.repositories import paper_repo
from app.schemas.schemas import (
    IdeaList, IdeaScore, NoveltyCheck,
    IdeaValidationList,
)
from app.services.event_service import emit_event
from app.services.search_service import search_service

logger = logging.getLogger(__name__)


# Known real baselines (generic terms that are real but not specific method names)
KNOWN_REAL_BASELINES = {
    "memgpt", "rag", "bert", "gpt-4", "gpt-3", "gpt-3.5", "llama", "llama-2",
    "llama-3", "t5", "transformer", "fine-tuned llm", "standard llm",
    "chain-of-thought", "few-shot", "zero-shot", "lora", "rlhf", "dpo",
    "bleu", "rouge", "f1", "accuracy", "precision", "recall",
    "memorybank", "standard memory", "no memory", "standard dialogue system",
    "standard multimodal", "standard multi-agent",
    "标准对话系统", "标准多智能体", "标准记忆机制", "标准记忆", "标准方法",
    "标准多模态", "标准LLM", "标准LLM无记忆", "无记忆", "标准基线",
    "标准对话", "标准多智能体协作", "标准框架",
}

KNOWN_DATASETS = {
    "multwoz", "mmlu", "glue", "ms coco", "mscoco", "wikitext-103", "wikitext",
    "clevrer", "squad", "squad 2.0", "natural questions", "trivia qa", "triviaqa",
    "hotpotqa", "wikipedia", "cnn/dailymail", "cnn dailymail", "xsum", "wmt",
    "multiwoz", "babi", "dialogue", "persona-chat", "personachat",
    "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "fashion-mnist",
    "openbookqa", "arc", "hellaswag", "winogrande", "gsm8k", "math",
    "human eval", "humaneval", "mbpp", "codecontests",
}


async def generate_and_score_ideas(db, state: ResearchState, llm, task_id: str,
                                    prev_feedback: str = "", cluster_list=None):
    """Generate ideas, validate (5-layer), enrich, and score them."""
    from app.agent.steps.build_clusters import build_paper_clusters

    # Build clusters first if not provided
    if cluster_list is None:
        cluster_list = await build_paper_clusters(db, state, llm, task_id)

    # Get high + medium priority papers for idea generation (eager load to avoid N+1)
    from sqlalchemy.orm import joinedload
    all_tps = db.query(TaskPaper).options(
        joinedload(TaskPaper.paper)
    ).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(50).all()

    high_papers = [(tp.paper, tp) for tp in all_tps if tp.paper]

    valid_paper_ids = {p.id for p, _ in high_papers}
    high_papers_text = "\n".join(
        f"[{p.id}] {p.title} ({p.year}) [{p.venue or 'N/A'}]: {(p.abstract or '')[:500]}\n"
        f"  方法摘要: {tp.summary or 'N/A'}"
        for p, tp in high_papers
    )

    # RAG: Retrieve full-text passages for idea grounding
    rag_evidence = _retrieve_idea_rag(state, valid_paper_ids, task_id)

    # Build cluster context
    cluster_context = _build_cluster_context(cluster_list)

    # Get latest report
    report = db.query(Report).filter(Report.task_id == task_id).order_by(Report.created_at.desc()).first()
    report_text = report.content_markdown if report else ""

    # Build user prompt
    user_content = IDEAS_USER.format(
        topic=state.normalized_topic,
        report=report_text[:3000],
        papers=high_papers_text or "(none)",
        gaps="\n".join(state.knowledge_gaps) if state.knowledge_gaps else "(none)",
    )
    if cluster_context:
        user_content += "\n\n" + cluster_context
    if rag_evidence:
        user_content += rag_evidence
    # Add wiki context
    user_content = _add_wiki_context(db, task_id, user_content)
    if prev_feedback:
        user_content += "\n\n--- 之前创意反馈 ---\n" + prev_feedback

    messages = [
        {"role": "system", "content": IDEAS_SYSTEM.format(num_ideas=5)},
        {"role": "user", "content": user_content},
    ]
    idea_list = await llm.chat_json(messages, IdeaList)

    # === Idea validation: dedup + baseline check + metric-hypothesis check ===
    validation_penalties: dict[int, float] = {}
    ideas_to_skip: set[int] = set()

    paper_repo.save_trace(db, task_id, "idea_validation_start", "action",
                          output_data={"ideas_count": len(idea_list.ideas)})
    db.commit()

    all_paper_abstracts_lower = [(p.title.lower(), (p.abstract or "").lower()) for p, _ in high_papers]

    # Step 1: LLM-based validation
    ideas_to_skip, validation_penalties = await _llm_validate_ideas(
        db, llm, state, task_id, idea_list, high_papers, validation_penalties, ideas_to_skip
    )

    # Step 2: Search-based baseline + dataset verification
    validation_penalties = await _verify_baselines_and_datasets(
        db, state, task_id, idea_list, high_papers, all_paper_abstracts_lower,
        validation_penalties, ideas_to_skip
    )

    # Process each idea: enrich + novelty check + score
    await _process_each_idea(
        db, state, llm, task_id, idea_list, high_papers,
        valid_paper_ids, validation_penalties, ideas_to_skip
    )

    paper_repo.save_trace(db, task_id, "generate_ideas", "action",
                          output_data={"count": len(idea_list.ideas)})
    db.commit()


def _retrieve_idea_rag(state, valid_paper_ids, task_id) -> str:
    """Retrieve RAG passages for idea grounding."""
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
                clean_text = re.sub(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]', '', r["text"])[:600].strip()
                rag_lines.append(f"[{r['paper_id'][:8]}] ({r['section']}) {clean_text}")
            logger.info("Task %s: RAG retrieved %d passages for idea generation", task_id[:8], len(rag_results))
            return "\n\n## 论文全文段落（RAG检索，用于支撑创意技术细节）\n" + "\n".join(rag_lines)
    except Exception as e:
        logger.warning("Task %s: RAG retrieval failed (non-fatal): %s", task_id[:8], e)
    return ""


def _build_cluster_context(cluster_list) -> str:
    """Build cluster context text from cluster_list."""
    if not cluster_list or not cluster_list.clusters:
        return ""
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
    return (
        "## 论文聚类分析（基于全部论文，非仅高优先级）\n\n"
        + "\n\n".join(cluster_lines)
        + "\n\n## 跨聚类机会\n"
        + "\n".join(f"- {g}" for g in cluster_list.cross_cluster_gaps)
    )


def _add_wiki_context(db, task_id, user_content) -> str:
    """Add wiki context to user content."""
    try:
        from app.services.wiki_service import get_wiki_context
        wiki_ctx = get_wiki_context(
            db, task_id,
            page_types=["method", "dataset", "model", "synthesis", "concept"],
            max_chars=12000,
        )
        if wiki_ctx:
            logger.info("Task %s: wiki context added to idea generation (%d chars)",
                       task_id[:8], len(wiki_ctx))
            return user_content + "\n\n" + wiki_ctx
    except Exception as e:
        logger.warning("Task %s: wiki context for ideas failed (non-fatal): %s", task_id[:8], e)
    return user_content


async def _llm_validate_ideas(db, llm, state, task_id, idea_list, high_papers,
                               validation_penalties, ideas_to_skip):
    """LLM-based validation: dedup + metric-hypothesis check."""
    try:
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
    return ideas_to_skip, validation_penalties


async def _verify_baselines_and_datasets(db, state, task_id, idea_list, high_papers,
                                          all_paper_abstracts_lower, validation_penalties, ideas_to_skip):
    """Search-based baseline + dataset verification."""
    try:
        baselines_to_check: list[tuple[str, int]] = []
        for idx, item in enumerate(idea_list.ideas):
            if idx in ideas_to_skip:
                continue
            method = item.method_sketch or ""
            baseline_match = re.search(r'基线[：:]\s*(.+?)(?:\n|$)', method)
            if baseline_match:
                baseline_text = baseline_match.group(1)
                raw_names = re.split(r'[、,，;；和与]\s*', baseline_text)
                for name in raw_names:
                    name = name.strip().rstrip('。.')
                    name = re.sub(r'^(比较|对比|与|和)\s*', '', name).strip()
                    name = re.sub(r'(进行对比|对比|的性能|等基线|等|基线|框架|系统)$', '', name).strip()
                    name = re.sub(r'[（(].*?[)）]\s*$', '', name).strip()
                    if len(name) > 2 and name.lower() not in KNOWN_REAL_BASELINES:
                        if not re.match(r'^(标准|普通|基础|传统|常规|一般)', name):
                            baselines_to_check.append((name, idx))

        unique_names = set(name for name, _ in baselines_to_check)
        logger.info("Task %s: verifying %d unique baseline names via search",
                    task_id[:8], len(unique_names))

        verified_baselines: set[str] = set()
        fabricated_baselines: list[tuple[str, int]] = []

        # First check: paper database
        for name, idea_idx in baselines_to_check:
            name_lower = name.lower()
            found_in_db = any(
                name_lower in title or name_lower in abstract
                for title, abstract in all_paper_abstracts_lower
            )
            if found_in_db:
                verified_baselines.add(name)

        # Second check: search verification
        names_to_search = {name for name, _ in baselines_to_check if name not in verified_baselines}
        semaphore = asyncio.Semaphore(3)

        async def verify_one_baseline(name: str) -> tuple[str, bool]:
            if name in verified_baselines:
                return name, True
            async with semaphore:
                name_lower = name.lower()
                try:
                    results = await search_service.search_all_sources(name, limit=5)
                    if not results or len(results) == 0:
                        try:
                            from app.paper_sources.crossref import CrossrefSource
                            cr = CrossrefSource()
                            cr_results = await cr.search(name, limit=5)
                            if cr_results:
                                for r in cr_results[:5]:
                                    if name_lower in (r.title or "").lower():
                                        return name, True
                        except Exception:
                            pass
                        return name, False
                    check_limit = 3 if len(name) < 5 else 5
                    for r in results[:check_limit]:
                        title_lower = (r.title or "").lower()
                        if name_lower in title_lower:
                            return name, True
                    return name, False
                except Exception:
                    return name, False

        if names_to_search:
            tasks_verify = [verify_one_baseline(name) for name in names_to_search]
            verify_results = await asyncio.gather(*tasks_verify)
            for name, is_real in verify_results:
                if is_real:
                    verified_baselines.add(name)

        # Build fabricated baselines list
        for name, idea_idx in baselines_to_check:
            if name not in verified_baselines:
                fabricated_baselines.append((name, idea_idx))

        # Apply penalties
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

        # === Dataset verification ===
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
                    name = re.sub(r'格式的.*$', '', name).strip()
                    if len(name) > 2:
                        datasets_to_check.append((name, idx))

        fabricated_datasets: list[tuple[str, int]] = []
        for ds_name, idea_idx in datasets_to_check:
            ds_lower = ds_name.lower()
            if ds_lower in KNOWN_DATASETS:
                continue
            found_in_db = any(
                ds_lower in title or ds_lower in abstract
                for title, abstract in all_paper_abstracts_lower
            )
            if not found_in_db:
                fabricated_datasets.append((ds_name, idea_idx))

        for ds_name, idea_idx in fabricated_datasets:
            penalty = 0.05
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
    return validation_penalties


async def _process_each_idea(db, state, llm, task_id, idea_list, high_papers,
                              valid_paper_ids, validation_penalties, ideas_to_skip):
    """Process each idea: enrich method, novelty check, score."""
    id_to_title = {p.id: p.title for p, _ in high_papers}

    for idx, item in enumerate(idea_list.ideas):
        if idx in ideas_to_skip:
            logger.info("Task %s: skipping duplicate idea '%s'", task_id[:8], item.title[:40])
            continue

        # Validate related_paper_ids
        valid_ids = [pid for pid in item.related_paper_ids if pid in valid_paper_ids]
        invalid_count = len(item.related_paper_ids) - len(valid_ids)
        if invalid_count > 0:
            logger.warning("Idea '%s': %d invalid paper IDs filtered out", item.title[:40], invalid_count)

        # Two-step: Enrich method_sketch with per-idea RAG retrieval
        enriched_method = await _enrich_method_sketch(
            db, state, llm, item, valid_ids, high_papers, valid_paper_ids, task_id
        )

        # Post-enrichment baseline check
        validation_penalties = await _post_enrichment_baseline_check(
            db, llm, state, task_id, idx, item, enriched_method,
            validation_penalties
        )

        idea_data = {
            "title": item.title,
            "description": item.description,
            "motivation": item.motivation,
            "method_sketch": enriched_method,
            "expected_contribution": item.expected_contribution,
            "related_paper_ids_json": json.dumps(valid_ids, ensure_ascii=False),
        }
        idea = paper_repo.save_idea(db, task_id, idea_data)

        # Novelty check
        novelty_penalty = await _check_novelty(db, state, llm, item, enriched_method, task_id)

        # Initial scoring
        try:
            score = await _score_idea(db, state, llm, idea)
            adjusted_novelty = max(0, score.novelty - novelty_penalty)
            idea_score_val = (
                0.20 * adjusted_novelty + 0.20 * score.feasibility + 0.20 * score.significance +
                0.20 * score.evidence_support + 0.10 * score.differentiation +
                0.10 * score.experimentability
            )
            final_score = idea_score_val - 0.08 * score.risk
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


async def _enrich_method_sketch(db, state, llm, item, valid_ids, high_papers, valid_paper_ids, task_id):
    """Enrich method_sketch with per-idea RAG retrieval."""
    enriched_method = item.method_sketch
    try:
        from app.services.rag_service import rag_retrieve

        idea_query = f"{item.title} {item.description[:200]}"
        idea_rag_results = rag_retrieve(
            query=idea_query,
            top_k=10,
            paper_ids=list(valid_paper_ids),
            section_filter=["method", "experiment"],
        )
        if idea_rag_results:
            _figure_pattern = re.compile(r'\[FIGURE:.*?\]|\[TABLE\]|\[/TABLE\]')
            rag_passages = "\n\n".join(
                f"[{r['paper_id'][:8]}] ({r['section']}) "
                f"{_figure_pattern.sub('', r['text'])[:600].strip()}"
                for r in idea_rag_results[:10]
            )
            id_to_abstract = {p.id: (p.abstract or '')[:200] for p, _ in high_papers}
            id_to_title = {p.id: p.title for p, _ in high_papers}
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

    except Exception as e:
        logger.warning("Task %s: method enrichment failed for idea '%s' (using original): %s",
                      task_id[:8], item.title[:40], e)
    return enriched_method


async def _post_enrichment_baseline_check(db, llm, state, task_id, idx, item,
                                           enriched_method, validation_penalties):
    """Check baselines in the enriched method sketch."""
    try:
        enriched_baseline_match = re.search(r'基线[：:]\s*(.+?)(?:\n|$)', enriched_method)
        if enriched_baseline_match:
            enriched_baselines = re.split(r'[、,，;；和与]\s*', enriched_baseline_match.group(1))
            new_fabricated = []
            for bname in enriched_baselines:
                bname = re.sub(r'^(比较|对比|与|和)\s*', '', bname.strip().rstrip('。.'))
                bname = re.sub(r'(进行对比|对比|的性能|等基线|等|基线|框架|系统)$', '', bname).strip()
                bname = re.sub(r'[（(].*?[)）]\s*$', '', bname).strip()
                if len(bname) > 2 and bname.lower() not in KNOWN_REAL_BASELINES:
                    if not re.match(r'^(标准|普通|基础|传统|常规|一般)', bname):
                        new_fabricated.append(bname)

            if new_fabricated:
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
        logger.warning("Task %s: post-enrichment baseline check failed: %s", task_id[:8], e)
    return validation_penalties


async def _check_novelty(db, state, llm, item, enriched_method, task_id) -> float:
    """Novelty check — search for similar existing work."""
    novelty_penalty = 0.0
    search_succeeded = False
    try:
        novelty_query = f"{item.title} {item.description[:100]}"
        existing_papers = await search_service.search_all_sources(novelty_query, limit=5)
        search_succeeded = True  # search returned (even if empty)
        existing_text = "\n".join(
            f"- {p.title}: {(p.abstract or '')[:150]}"
            for p in existing_papers[:5]
        ) or "(no similar papers found)"

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
            novelty_penalty = 0.1
            logger.info("Idea '%s' novelty check: NOT NOVEL (similar: %s)",
                       item.title[:40], novelty_result.similar_papers[:2])
        else:
            logger.info("Idea '%s' novelty check: NOVEL", item.title[:40])
    except Exception as e:
        logger.warning("Novelty check failed for idea '%s': %s", item.title[:40], e)

    # If search service was unavailable (rate limited / network error),
    # do NOT trust a "novel" verdict — apply a conservative penalty instead.
    if not search_succeeded:
        novelty_penalty = 0.05  # small penalty: can't verify novelty, don't reward
        logger.warning("Idea '%s': novelty search unavailable, applied conservative penalty",
                       item.title[:40])
    return novelty_penalty


async def _score_idea(db, state: ResearchState, llm, idea) -> IdeaScore:
    """Score a single idea using LLM."""
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

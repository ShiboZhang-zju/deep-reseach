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
    IDEA_EXTRACT_SYSTEM, IDEA_EXTRACT_USER,
)
from app.db.models import Paper, TaskPaper, Report
from app.db.repositories import paper_repo
from app.schemas.schemas import (
    IdeaList, IdeaScore, NoveltyCheck,
    IdeaValidationList, IdeaMethodExtract,
)
from app.services.event_service import emit_event
from app.services.search_service import search_service

logger = logging.getLogger(__name__)


def _is_real_match(name: str, text: str) -> bool:
    """Check if name appears as a real method/dataset name in text.

    Uses word-boundary matching for short names (<=4 chars) to avoid
    false positives like 'GAT' matching 'concatenate'.
    For longer names, substring match is acceptable.
    """
    if not name or not text:
        return False
    name_lower = name.lower()
    text_lower = text.lower()
    if len(name_lower) <= 4:
        # Short names: require word boundary to avoid false matches
        # Match as whole word or with common separators
        pattern = r'(?<![a-z0-9])' + re.escape(name_lower) + r'(?![a-z0-9])'
        return bool(re.search(pattern, text_lower))
    return name_lower in text_lower


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
    # P0-2: Added test oracle / SE baselines
    "toga", "codet5", "graphcodebert", "codellama", "starcoder",
    "gcn", "gat", "graphsage", "gin", "mpnn",
    "llamaguard", "deepseek-coder", "wizardcoder",
    "defects4j", "sf110",
    # P1: Added more APR/SE method names
    "chatrepair", "cigar", "repaircat", "fixgpt", "gpt-4o", "gpt-4o-mini",
    "gpt-3.5-turbo", "gpt-3.5", "deepseek", "deepseek-v3", "deepseek-r1",
    "deepseek coder", "claude", "claude-3", "claude-3.5", "claude-3 opus",
    "gemini", "mistral", "grok", "qwen", "qwen2.5", "glm-4",
    "kodezi", "kodezi chronos", "chronos-1", "chronos",
    "self-debug", "rtlfixer", "verigen", "rtlcoder",
    "tbar", "selfapr", "alpharepair", "recoder", "codereviewer",
    "hdldebugger", "scanfix", "bloomapr", "pracapr", "gamma",
    "autoprogram", "fixagent", "viscratch", "hafix", "knowbug",
    "codet5+", "plbart", "codegen", "polycoder",
    "virustotal", "sonarqube",
}

KNOWN_DATASETS = {
    "multwoz", "mmlu", "glue", "ms coco", "mscoco", "wikitext-103", "wikitext",
    "clevrer", "squad", "squad 2.0", "natural questions", "trivia qa", "triviaqa",
    "hotpotqa", "wikipedia", "cnn/dailymail", "cnn dailymail", "xsum", "wmt",
    "multiwoz", "babi", "dialogue", "persona-chat", "personachat",
    "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "fashion-mnist",
    "openbookqa", "arc", "hellaswag", "winogrande", "gsm8k", "math",
    "human eval", "humaneval", "mbpp", "codecontests",
    # P0-2: Added SE datasets
    "defects4j", "sf110", "codexglue", "concode", "tocode",
    # P1: Added more SE benchmarks
    "swe-bench", "swe-bench lite", "swe-bench verified", "gitbug-java", "gitbug-java",
    "debugbench", "mdeval", "llmseceval", "securityeval", "xsafety",
    "quixbugs", "codeflaws", "bugsinpy", "bugsjs", "bugsphp",
    "repairbench", "runbugrun", "codereval",
    "hackage", "tocode", "evor-bench",
    "appsurvey", "codeforces", "leetcode",
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
    # Build P-number -> paper_id mapping for hallucination-free reference
    paper_num_to_id = {}
    # P0-5: Use deep analysis when available, fallback to abstract + summary
    from app.agent.steps.analyze_papers import get_analyses_for_task, format_analysis_for_context
    analyses = get_analyses_for_task(db, task_id)

    paper_lines = []
    for i, (p, tp) in enumerate(high_papers):
        num = f"P{i+1}"
        paper_num_to_id[num] = p.id
        analysis = analyses.get(p.id)
        if analysis:
            paper_lines.append(
                f"[{num}] {p.title} ({p.year}) [{p.venue or 'N/A'}]:\n"
                f"{format_analysis_for_context(analysis)}"
            )
        else:
            paper_lines.append(
                f"[{num}] {p.title} ({p.year}) [{p.venue or 'N/A'}]: {(p.abstract or '')[:500]}\n"
                f"  方法摘要: {tp.summary or 'N/A'}"
            )
    high_papers_text = "\n".join(paper_lines)

    # P1-1: Build component combination matrix from extendable_components
    combination_context = _build_combination_context(analyses, high_papers)

    # RAG: Retrieve full-text passages for idea grounding
    rag_evidence = await _retrieve_idea_rag(state, valid_paper_ids, task_id)

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
    if combination_context:
        user_content += "\n\n" + combination_context
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
        db, state, llm, task_id, idea_list, high_papers, all_paper_abstracts_lower,
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


async def _retrieve_idea_rag(state, valid_paper_ids, task_id) -> str:
    """Retrieve RAG passages for idea grounding."""
    try:
        from app.services.rag_service import rag_retrieve
        rag_results = await rag_retrieve(
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


def _build_combination_context(analyses: dict, high_papers: list) -> str:
    """P1-1: Build component combination matrix from paper_analyses.

    Extracts extendable_components from each paper's analysis and presents
    them as a combination matrix, encouraging LLM to generate cross-paper
    innovation ideas.
    """
    if not analyses:
        return ""

    lines = ["## 论文可扩展组件矩阵（用于跨论文组合创新）\n"]
    lines.append("以下是从各论文中提取的可复用/可改进组件。生成Idea时，优先考虑**跨论文组合**这些组件：\n")

    for i, (paper, _) in enumerate(high_papers):
        analysis = analyses.get(paper.id)
        if not analysis or not analysis.extendable_components:
            continue
        if analysis.extendable_components.strip() and analysis.extendable_components.strip() != "论文未提及":
            lines.append(f"- [P{i+1}] {paper.title[:60]}:")
            # Split by newline or numbered list
            components = re.split(r'[\n;；]|(?<=\d\.)\s', analysis.extendable_components)
            for comp in components:
                comp = comp.strip().rstrip('。.')
                if comp and len(comp) > 3:
                    lines.append(f"  * {comp}")

    lines.append("\n**组合创新示例**：将论文A的组件X + 论文B的组件Y → 新方法。每个Idea应明确说明组合了哪些论文的哪些组件。")

    result = "\n".join(lines)
    if len(result) < 100:  # Not enough useful content
        return ""
    return result


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


async def _verify_baselines_and_datasets(db, state, llm, task_id, idea_list, high_papers,
                                          all_paper_abstracts_lower, validation_penalties, ideas_to_skip):
    """LLM-based baseline + dataset verification.

    Replaces white-list + search verification with LLM judgment.
    The LLM is given the known-real methods/datasets from paper_analyses
    as context, and uses its own knowledge to judge if each name is real.
    """
    try:
        # === Build known-real items list from paper_analyses ===
        from app.db.models import PaperAnalysis
        all_analyses = db.query(PaperAnalysis).filter(
            PaperAnalysis.task_id == task_id,
        ).all()

        # Extract method names and dataset names from paper analyses + paper titles
        known_methods = set()
        known_datasets = set()
        _stop_words = {'The', 'This', 'These', 'Using', 'Based', 'For', 'With', 'From', 'They',
                       'Our', 'We', 'Their', 'And', 'But', 'Not', 'Are', 'Was', 'Were', 'Has',
                       'Have', 'Had', 'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Can',
                       'All', 'Each', 'Some', 'Most', 'More', 'Less', 'Very', 'Such', 'Same',
                       'Other', 'Than', 'Then', 'When', 'Where', 'While', 'Which', 'What', 'Who',
                       'How', 'Why', 'That', 'Those', 'A', 'An', 'Towards', 'Exploring',
                       'Leveraging', 'Enhancing', 'Automated', 'Automatic', 'Assessing',
                       'Evaluating', 'Understanding', 'Combining', 'Beyond', 'Large', 'Language',
                       'Models', 'Deep', 'Learning', 'Neural', 'Network', 'Program', 'Code',
                       'Software', 'Bug', 'Fault', 'Repair', 'Fix', 'Debug', 'Error', 'Test',
                       'Survey', 'Review', 'Study', 'Analysis', 'Approach', 'Framework', 'System',
                       'Method', 'Tool', 'Benchmark', 'Evaluation', 'How', 'Is', 'Self'}

        for a in all_analyses:
            combined = f"{a.method_detail or ''} {a.experiment_setup or ''}"
            for match in re.findall(r'\b[A-Z][A-Za-z0-9-]{2,}\b', combined):
                if match not in _stop_words:
                    known_methods.add(match)
            for match in re.findall(r'\b[A-Z][A-Za-z0-9-]*(?:Bench|Eval|Test|Set|Data|Corpus|Suite|Base|DB|4J|Java|Py|JS|PHP)\b', combined):
                known_datasets.add(match)

        for p, _ in high_papers:
            if p.title:
                for w in re.findall(r'\b([A-Z][A-Za-z0-9-]{2,})\b', p.title)[:3]:
                    if w not in _stop_words:
                        known_methods.add(w)

        known_real_items = sorted(known_methods | known_datasets)[:100]
        known_real_text = ", ".join(known_real_items) if known_real_items else "(none)"
        logger.info("Task %s: built known-real list with %d items from paper analyses",
                    task_id[:8], len(known_real_items))

        # === LLM extraction with known-real context ===
        extraction_results: dict[int, IdeaMethodExtract] = {}
        for idx, item in enumerate(idea_list.ideas):
            if idx in ideas_to_skip:
                continue
            method = item.method_sketch or ""
            if not method.strip():
                continue
            try:
                extraction = await llm_extract_method_components(
                    db, state, task_id, item.title, method, known_real_text
                )
                extraction_results[idx] = extraction
            except Exception as e:
                logger.warning("Task %s: LLM extraction failed for idea %d: %s", task_id[:8], idx+1, e)

        # === Apply penalties based on LLM judgment ===
        fabricated_baselines: list[tuple[str, int]] = []
        for idx, extraction in extraction_results.items():
            if extraction.has_fake_content and extraction.fake_items:
                for fake_name in extraction.fake_items:
                    fabricated_baselines.append((fake_name, idx))
                    logger.warning("Task %s: idea %d LLM flagged fabricated: %s",
                                 task_id[:8], idx+1, fake_name)

        idea_fabricated_count: dict[int, list[str]] = {}
        for name, idea_idx in fabricated_baselines:
            idea_fabricated_count.setdefault(idea_idx, []).append(name)

        for idea_idx, fake_names in idea_fabricated_count.items():
            penalty = min(0.2, 0.08 * len(fake_names))
            validation_penalties[idea_idx] = validation_penalties.get(idea_idx, 0) + penalty
            logger.warning("Task %s: idea %d '%s' has %d FABRICATED baselines: %s (penalty: +%.2f)",
                          task_id[:8], idea_idx+1, idea_list.ideas[idea_idx].title[:40],
                          len(fake_names), fake_names, penalty)

        verified_count = sum(len(e.baselines) for e in extraction_results.values()) - len(fabricated_baselines)
        logger.info("Task %s: baseline verification done (LLM-based) — %d verified, %d fabricated",
                    task_id[:8], verified_count, len(fabricated_baselines))
        paper_repo.save_trace(db, task_id, "idea_baseline_verification", "observation",
                              output_data={"verified": verified_count,
                                           "fabricated": len(fabricated_baselines),
                                           "fabricated_names": [n for n, _ in fabricated_baselines],
                                           "method": "llm_judgment"})
        db.commit()

    except Exception as e:
        logger.warning("Task %s: baseline verification failed (non-fatal): %s", task_id[:8], e)
        paper_repo.save_trace(db, task_id, "idea_baseline_verification", "observation",
                              output_data={"status": "failed", "error": str(e)[:200]})
        db.commit()
    return validation_penalties


async def _process_each_idea(db, state, llm, task_id, idea_list, high_papers,
                              valid_paper_ids, validation_penalties, ideas_to_skip):
    """Process each idea: enrich method, novelty check, score."""
    id_to_title = {p.id: p.title for p, _ in high_papers}

    # Build P-number -> paper_id mapping (same as in generate_and_score_ideas)
    paper_num_to_id = {}
    for i, (p, _) in enumerate(high_papers):
        paper_num_to_id[f"P{i+1}"] = p.id

    for idx, item in enumerate(idea_list.ideas):
        if idx in ideas_to_skip:
            logger.info("Task %s: skipping duplicate idea '%s'", task_id[:8], item.title[:40])
            continue

        # Validate related_paper_ids — support both [P1] format and raw UUIDs
        valid_ids = []
        for pid in item.related_paper_ids:
            # Try P-number format first (e.g., "P1", "P3")
            if pid.upper().startswith("P") and pid[1:].isdigit():
                paper_id = paper_num_to_id.get(pid.upper())
                if paper_id:
                    valid_ids.append(paper_id)
                else:
                    logger.warning("Idea '%s': invalid paper number '%s' (not in list)", item.title[:40], pid)
            # Fall back to raw UUID (legacy)
            elif pid in valid_paper_ids:
                valid_ids.append(pid)
            else:
                logger.warning("Idea '%s': invalid paper ID '%s' (not in database)", item.title[:40], pid)

        invalid_count = len(item.related_paper_ids) - len(valid_ids)
        if invalid_count > 0:
            logger.warning("Idea '%s': %d invalid paper IDs filtered out", item.title[:40], invalid_count)

        # P0-1: Reject ideas with NO valid paper references — they are hallucinated
        if not valid_ids:
            logger.warning("Task %s: idea '%s' has NO valid paper references — REJECTING (hallucination guard)",
                          task_id[:8], item.title[:40])
            paper_repo.save_trace(db, task_id, "idea_rejected_no_refs", "observation",
                                  output_data={"title": item.title[:200],
                                               "reason": "no valid related_paper_ids",
                                               "raw_ids": item.related_paper_ids})
            db.commit()
            continue

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
                0.05 * score.experimentability + 0.05 * score.potential_impact
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
        idea_rag_results = await rag_retrieve(
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


async def llm_extract_method_components(db, state, task_id, idea_title: str,
                                         method_sketch: str, known_real_text: str = "(none)") -> IdeaMethodExtract:
    """P0-3: Use LLM to extract structured components from method_sketch.

    Replaces regex-based extraction with LLM-based parsing for accuracy.
    Returns structured data including baselines, datasets, metrics, and
    a has_fake_content flag for hallucination detection.
    """
    messages = [
        {"role": "system", "content": IDEA_EXTRACT_SYSTEM},
        {"role": "user", "content": IDEA_EXTRACT_USER.format(
            topic=state.normalized_topic,
            title=idea_title,
            method_sketch=method_sketch,
            known_real_items=known_real_text,
        )},
    ]
    result = await llm_extract_with_retry(messages)
    return result


async def llm_extract_with_retry(messages, max_retries: int = 2):
    """Call LLM for IdeaMethodExtract with retry on parse failure."""
    from app.llm.factory import get_llm
    llm = get_llm()
    for attempt in range(max_retries + 1):
        try:
            return await llm.chat_json(messages, IdeaMethodExtract)
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(1)
            else:
                raise


async def _post_enrichment_baseline_check(db, llm, state, task_id, idx, item,
                                           enriched_method, validation_penalties):
    """Check baselines in the enriched method sketch.

    P0-3: Uses LLM extraction instead of regex.
    """
    try:
        # P0-3: Use LLM extraction instead of regex
        extraction = await llm_extract_method_components(
            db, state, task_id, item.title, enriched_method
        )

        new_fabricated = []
        for bname in extraction.baselines:
            bname = bname.strip()
            if len(bname) > 1 and bname.lower() not in KNOWN_REAL_BASELINES:
                if not re.match(r'^(标准|普通|基础|传统|常规|一般)', bname):
                    new_fabricated.append(bname)

        # Also check LLM-flagged fake items
        if extraction.has_fake_content:
            for fake_name in extraction.fake_items:
                if fake_name not in new_fabricated:
                    new_fabricated.append(fake_name)

        if new_fabricated:
            new_results = await asyncio.gather(*[_verify_baseline_name(n) for n in new_fabricated])
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


async def _verify_baseline_name(name: str) -> tuple[str, bool]:
    """Verify a baseline name via search. Returns (name, is_real).

    On search failure, returns False (do NOT give benefit of doubt —
    consistent with the main baseline verification logic).
    """
    try:
        results = await search_service.search_all_sources(name, limit=3)
        if not results:
            return name, False
        for r in results[:3]:
            if name.lower() in (r.title or "").lower():
                return name, True
        return name, False
    except Exception:
        return name, False


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

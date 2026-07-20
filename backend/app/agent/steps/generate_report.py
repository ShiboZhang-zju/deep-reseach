"""Step: Generate research report (STORM-style two-step)."""

import logging
import re

from app.agent.state import ResearchState
from app.agent.prompts import (
    REPORT_SYSTEM, REPORT_USER,
    REPORT_OUTLINE_SYSTEM, REPORT_OUTLINE_USER,
    REPORT_SECTION_SYSTEM, REPORT_SECTION_USER,
)
from app.db.models import Paper, TaskPaper, Report
from app.db.repositories import paper_repo
from app.schemas.schemas import ReportOutline
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)


async def generate_report(db, state: ResearchState, llm, cluster_list=None) -> str:
    """Two-step report generation (STORM-style: outline → fill section by section).

    Step 1: Generate a structured outline based on clusters + papers.
    Step 2: For each section, select relevant papers + RAG evidence, generate content.
    Step 3: Assemble all sections + references.
    """
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
    paper_by_index = {i + 1: p for i, p in enumerate(all_papers)}

    # Build clusters text from in-memory cluster_list (passed from caller)
    clusters_text = ""
    if cluster_list and cluster_list.clusters:
        cluster_lines = []
        for i, c in enumerate(cluster_list.clusters):
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
        outline = None

    if outline and outline.sections:
        report_text = await _fill_sections(
            db, state, llm, outline, all_papers, paper_by_index, wiki_context_text
        )
    else:
        report_text = await _one_shot_report(
            db, state, llm, all_papers, papers_text, wiki_context_text
        )

    # Post-check: detect placeholder text
    _check_placeholders(report_text, state.task_id)
    # P1-6: validate [Px] citations reference real papers
    report_text = _validate_and_clean_citations(db, report_text, all_papers, state.task_id)

    paper_repo.save_report(db, state.task_id, report_text)
    # Record LLM token usage
    total_tokens = 0
    if hasattr(llm, "get_last_usage") and llm.get_last_usage():
        total_tokens = llm.get_last_usage().get("total_tokens", 0)
    paper_repo.save_trace(db, state.task_id, "generate_report", "action",
                          output_data={"length": len(report_text),
                                       "method": "two_step" if outline else "one_shot",
                                       "sections": len(outline.sections) if outline else 0},
                          tokens=total_tokens)
    db.commit()
    emit_event(state.task_id, "report_ready", {"length": len(report_text)})
    return report_text


async def _fill_sections(db, state, llm, outline, all_papers, paper_by_index, wiki_context_text) -> str:
    """Fill each section of the outline concurrently."""
    import asyncio

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
            section_paper_indices = section.paper_indices or []
            if not section_paper_indices:
                section_paper_indices = list(range(1, len(all_papers) + 1))

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
            except Exception as e:
                logger.debug("Section RAG retrieval failed (non-fatal): %s", e)

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

    section_tasks = [
        generate_one_section(section, i)
        for i, section in enumerate(outline.sections)
    ]
    section_contents = await asyncio.gather(*section_tasks)

    # === Step 3: Assemble + add references ===
    report_text = "\n\n".join(section_contents)

    ref_lines = ["## 参考文献\n"]
    for i, p in enumerate(all_papers):
        ref_lines.append(f"[P{i+1}] {p.title} ({p.year}). DOI: {p.doi or 'N/A'}")
    report_text += "\n\n" + "\n".join(ref_lines)
    return report_text


async def _one_shot_report(db, state, llm, all_papers, papers_text, wiki_context_text) -> str:
    """Fallback: one-shot report generation."""
    logger.info("Task %s: using one-shot report generation (fallback)", state.task_id[:8])
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
    return await llm.chat(messages, temperature=0.5)


_PLACEHOLDER_PATTERNS = [
    r'（.*?保持不变.*?）', r'（.*?保持原.*?）', r'（.*?新增内容.*?）',
    r'（.*?此部分.*?）', r'（.*?其余.*?不变.*?）', r'（.*?省略.*?）',
    r'（.*?详见.*?）', r'（.*?同上.*?）', r'（.*?参见.*?）',
    r'（.*?不?变.*?）', r'（.*?未变.*?）',
    r'\(.*?remain.*?unchanged.*?\)', r'\(.*?see above.*?\)', r'\(.*?omitted.*?\)',
    r'^（.*?补充.*?）$', r'^（.*?完善.*?）$',
]


def _check_placeholders(report_text: str, task_id: str):
    """Log warning if report contains placeholder text."""
    has_placeholders = any(re.search(p, report_text) for p in _PLACEHOLDER_PATTERNS)
    if has_placeholders:
        logger.warning("Task %s: report contains placeholder text", task_id[:8])


def _validate_and_clean_citations(db, report_text: str, all_papers: list, task_id: str) -> str:
    """P1-6: Validate [Px] citations reference real papers.

    - Collects all [P1], [P2], ... references from the report body.
    - Compares against the actual paper list (1-indexed).
    - Logs warnings for fabricated citations.
    - Does NOT delete fabricated citations from text (would break readability),
      but logs them for visibility and emits a trace.
    """
    from app.db.repositories import paper_repo

    # Find all [P<number>] references in the report
    cited_indices = set()
    for m in re.finditer(r'\[P(\d+)\]', report_text):
        try:
            cited_indices.add(int(m.group(1)))
        except ValueError:
            continue

    valid_indices = set(range(1, len(all_papers) + 1))
    fabricated = cited_indices - valid_indices

    if fabricated:
        logger.warning(
            "Task %s: report contains %d FABRICATED citations: %s (valid range: 1-%d)",
            task_id[:8], len(fabricated),
            sorted(fabricated)[:10], len(all_papers),
        )
        try:
            paper_repo.save_trace(db, task_id, "report_citation_check", "observation",
                                  output_data={
                                      "fabricated_citations": sorted(fabricated),
                                      "valid_range": f"1-{len(all_papers)}",
                                      "total_cited": len(cited_indices),
                                  })
            db.commit()
        except Exception as e:
            logger.debug("Failed to save citation check trace: %s", e)
    else:
        logger.info("Task %s: all %d citations are valid", task_id[:8], len(cited_indices))

    # Compute citation coverage: how many of the provided papers were actually cited
    cited_from_valid = cited_indices & valid_indices
    coverage = len(cited_from_valid) / max(len(all_papers), 1)
    if coverage < 0.3:
        logger.warning("Task %s: low citation coverage %.1f%% (%d/%d papers cited)",
                      task_id[:8], coverage * 100, len(cited_from_valid), len(all_papers))

    return report_text

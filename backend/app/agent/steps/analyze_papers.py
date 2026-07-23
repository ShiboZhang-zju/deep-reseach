"""Step: Deep paper analysis — structured LLM analysis of each high-priority paper.

This step sits between scoring and wiki/report generation:
  search → score → **analyze_papers** → wiki → report → ideas

For high-priority papers:
  1. Download PDF (reuse existing multi-source download)
  2. Extract text by section (PyMuPDF, skip images)
  3. LLM generates structured analysis (problem, method, experiment, results, limitations, extendable)
  4. Store in paper_analyses table

For medium/low papers:
  1. Use full abstract (no truncation)
  2. LLM generates brief structured analysis
  3. Store in paper_analyses table

The analysis results replace the shallow "500-char abstract + one-line summary"
used by report/idea generation, providing deep grounding knowledge.
"""

import asyncio
import json
import logging
import os
import re

from app.agent.state import ResearchState
from app.agent.prompts import (
    PAPER_ANALYSIS_SYSTEM,
    PAPER_ANALYSIS_USER,
    PAPER_ANALYSIS_USER_ABSTRACT_ONLY,
)
from app.db.models import Paper, TaskPaper, PaperAnalysis
from app.db.repositories import paper_repo
from app.schemas.schemas import PaperAnalysisSchema
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

# Section keywords for PDF text segmentation
SECTION_KEYWORDS = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background", "overview", "preliminar"],
    "method": ["method", "methodology", "approach", "framework", "design", "model architecture", "proposed"],
    "experiment": ["experiment", "evaluation", "results", "setup", "implementation", "ablation", "rq"],
    "conclusion": ["conclusion", "discussion", "future work", "limitation", "summary", "threats to validity"],
}

# Max tokens for full text sent to LLM (keep well within 128K limit)
MAX_FULL_TEXT_CHARS = 80000  # ~20K tokens, safe for gpt-4o


def extract_pdf_text_by_section(pdf_path: str) -> dict[str, str] | None:
    """Extract text from PDF, organized by section.

    Uses PyMuPDF to extract text (no images). Segments text into sections
    based on heading detection.

    Returns dict like {"abstract": "...", "method": "...", "experiment": "..."}
    or None if PDF cannot be parsed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available, cannot extract PDF text")
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("Failed to open PDF %s: %s", pdf_path, e)
        return None

    sections: dict[str, list[str]] = {
        "abstract": [],
        "introduction": [],
        "method": [],
        "experiment": [],
        "conclusion": [],
    }
    current_section = "introduction"  # default before we detect a heading

    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        full_text += page_text

        # Detect section headings
        for line in page_text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) > 80:
                # Too long to be a heading, add to current section
                sections[current_section].append(line_stripped)
                continue

            line_lower = line_stripped.lower()
            # Check if this line is a section heading
            section_detected = False
            for sec_name, keywords in SECTION_KEYWORDS.items():
                for kw in keywords:
                    # Heading match: line contains keyword AND is short (heading-like)
                    if kw in line_lower and len(line_lower) < 60:
                        # Additional check: heading usually starts with a number or the keyword
                        if (re.match(r"^(\d+\.?\s*)?" + re.escape(kw), line_lower) or
                            line_lower.startswith(kw)):
                            current_section = sec_name
                            section_detected = True
                            break
                if section_detected:
                    break

            if not section_detected:
                sections[current_section].append(line_stripped)

    doc.close()

    # Join section texts
    result = {}
    for sec_name, lines in sections.items():
        text = "\n".join(l for l in lines if l)
        if text.strip():
            result[sec_name] = text

    if not result:
        return None
    return result


def truncate_full_text(sections: dict[str, str], max_chars: int = MAX_FULL_TEXT_CHARS) -> str:
    """Format sections into a single text block, truncating if too long.

    Priority: abstract > method > experiment > conclusion > introduction
    (introduction is least important for analysis)
    """
    priority = ["abstract", "method", "experiment", "conclusion", "introduction"]
    parts = []
    total = 0

    for sec in priority:
        text = sections.get(sec, "")
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "...(truncated)"
        parts.append(f"[{sec.upper()}]\n{text}")
        total += len(text)

    return "\n\n".join(parts)


def get_paper_analysis(db, paper_id: str, task_id: str) -> PaperAnalysis | None:
    """Retrieve stored analysis for a paper."""
    return db.query(PaperAnalysis).filter(
        PaperAnalysis.paper_id == paper_id,
        PaperAnalysis.task_id == task_id,
    ).first()


def get_analyses_for_task(db, task_id: str) -> dict[str, PaperAnalysis]:
    """Get all paper analyses for a task, indexed by paper_id."""
    analyses = db.query(PaperAnalysis).filter(
        PaperAnalysis.task_id == task_id,
    ).all()
    return {a.paper_id: a for a in analyses}


def format_analysis_for_context(analysis: PaperAnalysis) -> str:
    """Format a PaperAnalysis into a compact text block for LLM context.

    This replaces the old '500-char abstract + one-line summary' format.
    """
    return (
        f"  问题: {analysis.problem or 'N/A'}\n"
        f"  方法: {analysis.method_detail or 'N/A'}\n"
        f"  实验: {analysis.experiment_setup or 'N/A'}\n"
        f"  结果: {analysis.key_results or 'N/A'}\n"
        f"  局限: {analysis.limitations or 'N/A'}\n"
        f"  可扩展组件: {analysis.extendable_components or 'N/A'}"
    )


async def analyze_papers(db, state: ResearchState, llm, task_id: str):
    """Analyze high and medium priority papers.

    High priority: download PDF → extract full text → LLM deep analysis
    Medium priority: use full abstract → LLM brief analysis

    Results stored in paper_analyses table, used by wiki/report/ideas.
    """
    from app.services.rag_service import download_pdf_multi_source

    # Get high-priority papers (PDF download + full analysis)
    high_papers = db.query(Paper).join(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority == "high",
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(20).all()

    # Get medium-priority papers (abstract-only analysis)
    medium_papers = db.query(Paper).join(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority == "medium",
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(30).all()

    total = len(high_papers) + len(medium_papers)
    logger.info("Task %s: analyzing %d papers (%d high + %d medium)",
                task_id[:8], total, len(high_papers), len(medium_papers))
    emit_event(task_id, "status", {"status": "analyzing_papers", "total": total})

    # Check which papers already have analyses (skip if done)
    existing = get_analyses_for_task(db, task_id)
    high_to_analyze = [p for p in high_papers if p.id not in existing]
    medium_to_analyze = [p for p in medium_papers if p.id not in existing]

    if not high_to_analyze and not medium_to_analyze:
        logger.info("Task %s: all papers already analyzed, skipping", task_id[:8])
        return

    # === Phase 1: Download PDFs for high-priority papers ===
    pdf_paths: dict[str, str | None] = {}
    if high_to_analyze:
        logger.info("Task %s: downloading PDFs for %d high-priority papers",
                    task_id[:8], len(high_to_analyze))
        emit_event(task_id, "status", {"status": "downloading_pdfs", "total": len(high_to_analyze)})

        # Use existing PDF cache check + download
        pdf_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), "pdf_cache")

        async def download_one(paper) -> tuple[str, str | None]:
            # Check cache first
            cache_path = os.path.join(pdf_cache_dir, f"{paper.id}.pdf")
            if os.path.exists(cache_path):
                logger.debug("PDF cache hit for paper %s", paper.id[:8])
                return paper.id, cache_path

            # Download
            try:
                pdf_bytes = await download_pdf_multi_source(paper)
                if pdf_bytes:
                    # Save to cache
                    os.makedirs(pdf_cache_dir, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        f.write(pdf_bytes)
                    logger.info("PDF downloaded and cached for paper %s (%d KB)",
                                paper.id[:8], len(pdf_bytes) // 1024)
                    return paper.id, cache_path
            except Exception as e:
                logger.warning("PDF download failed for paper %s: %s", paper.id[:8], e)
            return paper.id, None

        # Concurrent download (max 3)
        sem = asyncio.Semaphore(3)

        async def download_with_sem(paper):
            async with sem:
                return await download_one(paper)

        results = await asyncio.gather(*[download_with_sem(p) for p in high_to_analyze])
        pdf_paths = dict(results)

        success_count = sum(1 for v in pdf_paths.values() if v)
        logger.info("Task %s: PDF download done — %d/%d success",
                    task_id[:8], success_count, len(high_to_analyze))

    # === Phase 2: LLM analysis ===
    logger.info("Task %s: running LLM analysis for %d papers", task_id[:8], total)
    emit_event(task_id, "status", {"status": "analyzing_papers_llm", "total": total})

    # Analyze high-priority papers (with PDF full text)
    sem_analysis = asyncio.Semaphore(5)  # 5 concurrent LLM calls

    async def analyze_one_with_pdf(paper):
        async with sem_analysis:
            return await _analyze_paper_with_pdf(db, state, llm, task_id, paper, pdf_paths.get(paper.id))

    async def analyze_one_abstract_only(paper):
        async with sem_analysis:
            return await _analyze_paper_abstract_only(db, state, llm, task_id, paper)

    # Run high-priority analyses
    high_tasks = [analyze_one_with_pdf(p) for p in high_to_analyze]
    medium_tasks = [analyze_one_abstract_only(p) for p in medium_to_analyze]

    all_tasks = high_tasks + medium_tasks
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    success = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if isinstance(r, Exception))
    logger.info("Task %s: paper analysis done — %d success, %d failed",
                task_id[:8], success, failed)

    paper_repo.save_trace(db, task_id, "analyze_papers", "action",
                          output_data={
                              "total_papers": total,
                              "high_priority": len(high_to_analyze),
                              "medium_priority": len(medium_to_analyze),
                              "pdf_downloaded": sum(1 for v in pdf_paths.values() if v),
                              "analysis_success": success,
                              "analysis_failed": failed,
                          })
    db.commit()


async def _analyze_paper_with_pdf(db, state, llm, task_id, paper, pdf_path: str | None) -> bool:
    """Analyze a high-priority paper using PDF full text."""
    has_full_text = False
    full_text_section = ""

    if pdf_path and os.path.exists(pdf_path):
        sections = extract_pdf_text_by_section(pdf_path)
        if sections:
            full_text_section = f"论文全文（按章节提取）:\n{truncate_full_text(sections)}"
            has_full_text = True
            logger.debug("Paper %s: PDF extracted %d sections, %d chars",
                        paper.id[:8], len(sections), len(full_text_section))

    if not has_full_text:
        # Fallback: use full abstract (no truncation)
        logger.debug("Paper %s: no PDF, using full abstract", paper.id[:8])

    user_content = PAPER_ANALYSIS_USER.format(
        title=paper.title,
        year=paper.year or "N/A",
        venue=paper.venue or "N/A",
        citations=paper.citation_count or 0,
        abstract=paper.abstract or "(no abstract available)",
        full_text_section=full_text_section if has_full_text else "(无PDF全文可用)",
    )

    return await _run_analysis_llm(db, llm, task_id, paper, user_content, has_full_text)


async def _analyze_paper_abstract_only(db, state, llm, task_id, paper) -> bool:
    """Analyze a medium-priority paper using full abstract only."""
    user_content = PAPER_ANALYSIS_USER_ABSTRACT_ONLY.format(
        title=paper.title,
        year=paper.year or "N/A",
        venue=paper.venue or "N/A",
        citations=paper.citation_count or 0,
        abstract=paper.abstract or "(no abstract available)",
    )

    return await _run_analysis_llm(db, llm, task_id, paper, user_content, has_full_text=False)


async def _run_analysis_llm(db, llm, task_id, paper, user_content: str, has_full_text: bool) -> bool:
    """Call LLM for paper analysis and store result."""
    try:
        messages = [
            {"role": "system", "content": PAPER_ANALYSIS_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        analysis = await llm.chat_json(messages, PaperAnalysisSchema)

        # Store in DB
        db_analysis = PaperAnalysis(
            task_id=task_id,
            paper_id=paper.id,
            problem=analysis.problem,
            method_detail=analysis.method_detail,
            experiment_setup=analysis.experiment_setup,
            key_results=analysis.key_results,
            limitations=analysis.limitations,
            extendable_components=analysis.extendable_components,
            source_sections=json.dumps(analysis.source_sections, ensure_ascii=False),
            has_full_text=has_full_text,
        )
        db.add(db_analysis)
        db.commit()

        logger.info("Paper %s: analysis done (full_text=%s, method=%d chars)",
                    paper.id[:8], has_full_text, len(analysis.method_detail))
        return True

    except Exception as e:
        logger.error("Paper %s: analysis failed: %s", paper.id[:8], e)
        db.rollback()
        return False

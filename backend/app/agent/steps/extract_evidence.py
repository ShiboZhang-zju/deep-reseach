"""Step: Extract evidence units from papers.

Phase 2: Replaces the 'analyze_papers' summary-only approach with
granular, traceable evidence units linked to research questions.
"""

import asyncio
import json
import logging
import os
import re

from app.agent.state import ResearchState
from app.agent.prompts import EVIDENCE_EXTRACT_SYSTEM, EVIDENCE_EXTRACT_USER
from app.db.models import Paper, TaskPaper, EvidenceUnit, PaperRole, ResearchQuestion
from app.db.repositories import paper_repo
from app.schemas.schemas import EvidenceExtractionList, PaperRoleClassificationSchema
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

# Max chars per chunk for evidence extraction
CHUNK_SIZE = 4000
MAX_CHUNKS_PER_PAPER = 15


async def extract_evidence_units(db, state: ResearchState, llm, task_id: str):
    """Extract evidence units from high-priority papers.

    For each paper:
    1. Get full text (PDF chunks or abstract)
    2. Split into chunks
    3. LLM extracts evidence units per chunk
    4. Store with provenance (paper_id, section, page_number, original_span)
    """
    from app.agent.steps.analyze_papers import get_analyses_for_task, extract_pdf_text_by_section

    # Get high-priority papers
    from sqlalchemy.orm import joinedload
    all_tps = db.query(TaskPaper).options(
        joinedload(TaskPaper.paper)
    ).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    ).order_by(TaskPaper.final_score.desc().nullslast()).limit(30).all()

    papers = [(tp.paper, tp) for tp in all_tps if tp.paper]

    if not papers:
        logger.warning("Task %s: no papers for evidence extraction", task_id[:8])
        return

    # Check for existing evidence to skip
    existing_count = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
    ).count()
    if existing_count > 0:
        logger.info("Task %s: %d evidence units already exist, skipping extraction", task_id[:8], existing_count)
        return

    logger.info("Task %s: extracting evidence from %d papers", task_id[:8], len(papers))
    emit_event(task_id, "status", {"status": "extracting_evidence", "total": len(papers)})

    # Get active research questions for linking
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.status.in_(["open", "partially_covered"]),
    ).all()

    semaphore = asyncio.Semaphore(5)
    total_evidence = 0

    async def extract_one(paper, tp):
        nonlocal total_evidence
        async with semaphore:
            return await _extract_from_paper(db, llm, task_id, paper, tp, questions)

    results = await asyncio.gather(*[extract_one(p, tp) for p, tp in papers], return_exceptions=True)

    for r in results:
        if isinstance(r, int):
            total_evidence += r

    # Classify paper roles
    await _classify_paper_roles(db, llm, task_id, papers)

    logger.info("Task %s: extracted %d evidence units from %d papers",
                task_id[:8], total_evidence, len(papers))

    paper_repo.save_trace(db, task_id, "extract_evidence_units", "action",
                          output_data={"total_evidence": total_evidence, "papers_processed": len(papers)})
    db.commit()


async def _extract_from_paper(db, llm, task_id, paper, tp, questions) -> int:
    """Extract evidence from a single paper."""
    # Try to get PDF full text
    sections = None
    pdf_path = _get_pdf_path(paper.id)

    if pdf_path and os.path.exists(pdf_path):
        try:
            from app.agent.steps.analyze_papers import extract_pdf_text_by_section
            sections = extract_pdf_text_by_section(pdf_path)
        except Exception as e:
            logger.debug("PDF extraction failed for paper %s: %s", paper.id[:8], e)

    evidence_count = 0

    if sections:
        # Extract from PDF sections
        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                continue
            # Split section into chunks
            chunks = [sec_text[i:i+CHUNK_SIZE] for i in range(0, len(sec_text), CHUNK_SIZE)][:3]
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    evidence_list = await _llm_extract_evidence(
                        llm, task_id, paper, chunk, sec_name,
                        page_number=chunk_idx + 1,
                    )
                    for ev in evidence_list:
                        eu = EvidenceUnit(
                            task_id=task_id,
                            paper_id=paper.id,
                            evidence_type=ev.evidence_type,
                            normalized_claim=ev.normalized_claim,
                            original_span=ev.original_span[:500] if ev.original_span else "",
                            section=sec_name,
                            page_number=chunk_idx + 1,
                            dataset_name=ev.dataset_name,
                            metric_name=ev.metric_name,
                            result_value=ev.result_value,
                            extraction_method="pdf_fulltext",
                            extraction_confidence=0.8,
                            verification_status="unverified",
                        )
                        db.add(eu)
                        evidence_count += 1
                except Exception as e:
                    logger.debug("Evidence extraction failed for paper %s chunk %s.%d: %s",
                                paper.id[:8], sec_name, chunk_idx, e)
    else:
        # Abstract fallback
        abstract = paper.abstract or ""
        if len(abstract) > 50:
            try:
                evidence_list = await _llm_extract_evidence(
                    llm, task_id, paper, abstract, "abstract", page_number=None,
                )
                for ev in evidence_list:
                    eu = EvidenceUnit(
                        task_id=task_id,
                        paper_id=paper.id,
                        evidence_type=ev.evidence_type,
                        normalized_claim=ev.normalized_claim,
                        original_span=ev.original_span[:500] if ev.original_span else "",
                        section="abstract",
                        page_number=None,
                        dataset_name=ev.dataset_name,
                        metric_name=ev.metric_name,
                        result_value=ev.result_value,
                        extraction_method="abstract_only",
                        extraction_confidence=0.4,
                        verification_status="abstract_only",
                    )
                    db.add(eu)
                    evidence_count += 1
            except Exception as e:
                logger.debug("Abstract evidence extraction failed for paper %s: %s", paper.id[:8], e)

    db.flush()
    return evidence_count


async def _llm_extract_evidence(llm, task_id, paper, text_chunk, section, page_number):
    """Call LLM to extract evidence units from a text chunk."""
    from app.agent.prompts import EVIDENCE_EXTRACT_SYSTEM, EVIDENCE_EXTRACT_USER

    messages = [
        {"role": "system", "content": EVIDENCE_EXTRACT_SYSTEM},
        {"role": "user", "content": EVIDENCE_EXTRACT_USER.format(
            title=paper.title or "",
            section=section,
            text_chunk=text_chunk[:CHUNK_SIZE],
        )},
    ]
    return await llm.chat_json(messages, EvidenceExtractionList)


def _get_pdf_path(paper_id: str) -> str | None:
    """Get cached PDF path for a paper."""
    pdf_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))), "pdf_cache")
    path = os.path.join(pdf_cache_dir, f"{paper_id}.pdf")
    return path if os.path.exists(path) else None


async def _classify_paper_roles(db, llm, task_id, papers):
    """Classify papers into roles (survey, seminal, benchmark, etc.)."""
    from app.agent.prompts import PAPER_ROLE_SYSTEM, PAPER_ROLE_USER

    semaphore = asyncio.Semaphore(5)

    async def classify_one(paper, tp):
        async with semaphore:
            try:
                # Simple heuristic classification (can be enhanced with LLM)
                roles = []
                title_lower = (paper.title or "").lower()
                abstract_lower = (paper.abstract or "").lower()
                combined = title_lower + " " + abstract_lower

                if any(w in combined for w in ["survey", "review", "tutorial", "comprehensive"]):
                    roles.append("survey")
                if (paper.citation_count or 0) > 100:
                    roles.append("seminal")
                if any(w in combined for w in ["benchmark", "dataset", "evaluation"]):
                    roles.append("benchmark")
                if any(w in combined for w in ["negative", "failure", "limitation", "cannot"]):
                    roles.append("negative_result")
                if not roles:
                    roles.append("method")

                for role in roles:
                    pr = PaperRole(
                        task_id=task_id,
                        paper_id=paper.id,
                        role=role,
                        confidence=0.6,
                        reason=f"heuristic: title/abstract contains '{role}' indicators",
                    )
                    db.add(pr)
            except Exception as e:
                logger.debug("Role classification failed for paper %s: %s", paper.id[:8], e)

    await asyncio.gather(*[classify_one(p, tp) for p, tp in papers], return_exceptions=True)
    db.flush()

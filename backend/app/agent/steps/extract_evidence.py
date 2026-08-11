"""Step: Extract evidence units from papers.

Phase 2.1 comprehensive rewrite:
- (#4) PDF download happens BEFORE extraction
- (#5) Real PDF page numbers, not chunk_idx+1
- (#6) Validate original_span exists in source chunk
- (#7) Save span_start, span_end, source_chunk_hash
- (#8) Incremental by paper/chunk/source_hash — skip already-extracted chunks
- (#9) Abstract-only evidence can be upgraded to full-text
- (#10) Idempotent — re-runs don't produce duplicates
- (#11) Each concurrent LLM call uses its own Session
"""

import asyncio
import hashlib
import json
import logging
import os

from app.agent.state import ResearchState
from app.agent.prompts import EVIDENCE_EXTRACT_SYSTEM, EVIDENCE_EXTRACT_USER
from app.db.models import (
    Paper, TaskPaper, EvidenceUnit, PaperRole, ResearchQuestion,
)
from app.db.session import SessionLocal
from app.db.repositories import paper_repo
from app.schemas.schemas import EvidenceExtractionList
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4000


def compute_chunk_hash(text: str) -> str:
    """SHA-256 of chunk text for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_span_in_chunk(original_span: str, chunk_text: str) -> tuple[int, int] | None:
    """(#6) Validate that original_span actually exists in the source chunk.

    Returns (start, end) character offsets, or None if not found.
    """
    if not original_span or not chunk_text:
        return None
    # Try exact match first
    pos = chunk_text.find(original_span)
    if pos >= 0:
        return pos, pos + len(original_span)
    # Try first 80 chars (LLM may have slightly modified the span)
    snippet = original_span[:80]
    if snippet:
        pos = chunk_text.find(snippet)
        if pos >= 0:
            return pos, pos + len(original_span)
    # Try last 80 chars
    snippet = original_span[-80:]
    if snippet:
        pos = chunk_text.find(snippet)
        if pos >= 0:
            return pos, pos + len(original_span)
    return None


async def extract_evidence_units(db, state: ResearchState, llm, task_id: str,
                                  round_number: int = 0):
    """Extract evidence units from papers found in this round (or all if round=0).

    Phase 2.1:
    - (#1) Called per-round from search loop
    - (#4) PDF download before extraction
    - (#8) Incremental — skips chunks already extracted
    - (#10) Idempotent — no duplicates on re-run
    - (#11) Concurrent LLM calls use separate sessions
    """
    from app.agent.steps.analyze_papers import extract_pdf_text_by_section
    from app.services.rag_service import download_pdf_multi_source
    from app.config import settings

    # Get papers to process
    query = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    )
    if round_number > 0:
        query = query.filter(TaskPaper.discovered_round == round_number)
    all_tps = query.order_by(
        TaskPaper.final_score.desc().nullslast()
    ).limit(settings.evidence_max_papers).all()

    papers = [(tp.paper, tp) for tp in all_tps if tp.paper]

    if not papers:
        logger.info("Task %s: no papers for evidence extraction (round %d)", task_id[:8], round_number)
        return 0

    logger.info("Task %s: extracting evidence from %d papers (round %d)",
                task_id[:8], len(papers), round_number)
    emit_event(task_id, "status", {"status": "extracting_evidence", "total": len(papers), "round": round_number})

    # (#4) Download PDFs first (before extraction)
    pdf_paths = await _download_pdfs(papers, task_id)

    # P0/P1: process in small batches so finished papers are committed (and
    # survive a crash/OOM) before starting the next batch, and so peak memory
    # is bounded to `batch_size` PDFs rather than all of them at once.
    batch_size = max(1, settings.evidence_batch_size)
    total_evidence = 0
    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start:batch_start + batch_size]
        semaphore = asyncio.Semaphore(batch_size)
        tasks_list = [
            _extract_from_paper_safe(
                task_id, paper, tp, pdf_paths.get(paper.id), llm, round_number, semaphore
            )
            for paper, tp in batch
        ]
        results = await asyncio.gather(*tasks_list, return_exceptions=True)
        for r in results:
            if isinstance(r, int):
                total_evidence += r
            elif isinstance(r, Exception):
                logger.error("Evidence extraction task failed: %s", r)
        logger.info("Task %s: evidence batch %d-%d done, cumulative=%d",
                    task_id[:8], batch_start, batch_start + len(batch), total_evidence)

    # Classify paper roles
    await _classify_paper_roles_safe(db, task_id, papers)

    logger.info("Task %s: extracted %d evidence units from %d papers (round %d)",
                task_id[:8], total_evidence, len(papers), round_number)

    paper_repo.save_trace(db, task_id, "extract_evidence_units", "action",
                          round_number=round_number,
                          output_data={
                              "total_evidence": total_evidence,
                              "papers_processed": len(papers),
                              "round": round_number,
                          })
    db.commit()
    return total_evidence


async def _download_pdfs(papers, task_id: str) -> dict[str, str]:
    """(#4) Download PDFs for papers before extraction."""
    from app.services.rag_service import download_pdf_multi_source

    pdf_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))), "pdf_cache")
    os.makedirs(pdf_cache_dir, exist_ok=True)

    pdf_paths = {}
    semaphore = asyncio.Semaphore(3)

    async def download_one(paper):
        async with semaphore:
            cache_path = os.path.join(pdf_cache_dir, f"{paper.id}.pdf")
            if os.path.exists(cache_path):
                return paper.id, cache_path
            try:
                pdf_bytes = await download_pdf_multi_source(paper)
                if pdf_bytes:
                    with open(cache_path, "wb") as f:
                        f.write(pdf_bytes)
                    return paper.id, cache_path
            except Exception as e:
                logger.debug("PDF download failed for paper %s: %s", paper.id[:8], e)
            return paper.id, None

    results = await asyncio.gather(*[download_one(p) for p, _ in papers], return_exceptions=True)
    for r in results:
        if isinstance(r, tuple):
            pdf_paths[r[0]] = r[1]
    return pdf_paths


async def _extract_from_paper_safe(task_id, paper, tp, pdf_path, llm, round_number, semaphore) -> int:
    """(#11) Wrapper that creates its own DB session for concurrent extraction.

    P2: guarded by a per-paper timeout so one slow/hung paper cannot stall the
    whole round. On timeout/error the paper's partial work is rolled back and we
    return 0; already-committed papers are unaffected.
    """
    from app.config import settings

    db = SessionLocal()
    try:
        async with semaphore:
            return await asyncio.wait_for(
                _extract_from_paper(db, llm, task_id, paper, tp, pdf_path, round_number),
                timeout=settings.evidence_paper_timeout_s,
            )
    except asyncio.TimeoutError:
        logger.warning("Evidence extraction timed out for paper %s after %ds",
                       paper.id[:8], settings.evidence_paper_timeout_s)
        db.rollback()
        return 0
    except Exception as e:
        logger.error("Evidence extraction failed for paper %s: %s", paper.id[:8], e)
        db.rollback()
        return 0
    finally:
        db.close()


async def _extract_from_paper(db, llm, task_id, paper, tp, pdf_path, round_number) -> int:
    """Extract evidence from a single paper with provenance validation.

    P1: the PDF is parsed exactly once (sections + per-page text together) and
    released immediately, instead of opening the same file twice and keeping all
    page text resident across the whole round.
    """
    evidence_count = 0

    sections = None
    page_texts: dict[int, str] = {}
    if pdf_path and os.path.exists(pdf_path):
        try:
            sections, page_texts = _parse_pdf_once(pdf_path)
        except Exception as e:
            logger.debug("PDF parse failed for paper %s: %s", paper.id[:8], e)

    if sections:
        # (#5) Use real PDF page numbers
        evidence_count += await _extract_from_sections(
            db, llm, task_id, paper, sections, page_texts, round_number
        )

        # (#9) Upgrade abstract-only evidence to full-text
        await _upgrade_abstract_evidence(db, paper, task_id)
    else:
        # Abstract fallback
        abstract = paper.abstract or ""
        if len(abstract) > 50:
            evidence_count += await _extract_from_abstract(db, llm, task_id, paper, abstract, round_number)

    # P1: free page text ASAP so peak memory is bounded per paper.
    page_texts.clear()
    if sections:
        sections.clear()

    db.commit()
    return evidence_count


async def _extract_from_sections(db, llm, task_id, paper, sections, page_texts, round_number) -> int:
    """Extract evidence from PDF sections with page numbers and chunk hashes.

    P1: `page_texts` is passed in (parsed once by the caller) instead of
    re-opening the PDF here.
    """
    from app.config import settings

    min_chunk_chars = max(50, settings.evidence_min_chunk_chars)
    max_chunks = max(1, settings.evidence_max_chunks_per_paper)
    chunk_concurrency = max(1, settings.evidence_chunk_concurrency)

    # 1) Collect candidate chunks across sections (bounded), de-duplicating and
    #    skipping already-extracted chunks. The most information-dense sections
    #    (method/experiment/conclusion) are prioritized over introduction.
    section_priority = {
        "method": 0, "experiment": 1, "conclusion": 2,
        "abstract": 3, "introduction": 4,
    }
    candidates = []  # (priority, sec_name, chunk_text, chunk_hash)
    seen_hashes: set[str] = set()
    for sec_name, sec_text in sections.items():
        if not sec_text.strip():
            continue
        prio = section_priority.get(sec_name, 5)
        for chunk_start in range(0, len(sec_text), CHUNK_SIZE):
            chunk_text = sec_text[chunk_start:chunk_start + CHUNK_SIZE]
            if len(chunk_text) < min_chunk_chars:
                continue
            chunk_hash = compute_chunk_hash(chunk_text)
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            existing = db.query(EvidenceUnit).filter(
                EvidenceUnit.task_id == task_id,
                EvidenceUnit.paper_id == paper.id,
                EvidenceUnit.source_chunk_hash == chunk_hash,
            ).first()
            if existing:
                continue
            candidates.append((prio, sec_name, chunk_text, chunk_hash))

    # Keep only the top `max_chunks` by section priority.
    candidates.sort(key=lambda c: c[0])
    candidates = candidates[:max_chunks]
    if not candidates:
        db.flush()
        return 0

    # 2) Extract chunks concurrently (bounded), then persist results in the
    #    caller's session serially (SQLAlchemy sessions are not thread-safe, but
    #    these coroutines share one event loop; DB writes happen after gather).
    sem = asyncio.Semaphore(chunk_concurrency)

    async def _extract_one(chunk_text):
        async with sem:
            try:
                result = await _llm_extract_evidence(llm, paper, chunk_text, "")
                # _llm_extract_evidence returns an EvidenceExtractionList model;
                # the actual units are on .evidence_units.
                return result.evidence_units
            except Exception as e:
                logger.debug("Chunk extraction failed for paper %s: %s", paper.id[:8], e)
                return []

    extraction_results = await asyncio.gather(
        *[_extract_one(c[2]) for c in candidates]
    )

    evidence_count = 0
    for (prio, sec_name, chunk_text, chunk_hash), evidence_list in zip(candidates, extraction_results):
        page_num = _find_page_for_text(chunk_text, page_texts, sec_name)
        for ev in evidence_list:
            # (#6) Validate original_span exists in chunk
            span_pos = find_span_in_chunk(ev.original_span, chunk_text)
            if span_pos is None:
                logger.debug("Paper %s: original_span not found in chunk, skipping evidence",
                             paper.id[:8])
                continue

            eu = EvidenceUnit(
                task_id=task_id,
                paper_id=paper.id,
                evidence_type=ev.evidence_type,
                normalized_claim=ev.normalized_claim,
                original_span=ev.original_span[:500] if ev.original_span else "",
                section=sec_name,
                page_number=page_num,
                page_start=page_num,
                page_end=page_num,
                span_start=span_pos[0],
                span_end=span_pos[1],
                source_chunk_hash=chunk_hash,
                dataset_name=ev.dataset_name,
                metric_name=ev.metric_name,
                result_value=ev.result_value,
                extraction_method="pdf_fulltext",
                extraction_confidence=0.8,
                verification_status="unverified",
            )
            db.add(eu)
            evidence_count += 1

    db.flush()
    return evidence_count


async def _extract_from_abstract(db, llm, task_id, paper, abstract, round_number) -> int:
    """Extract evidence from abstract (fallback)."""
    chunk_hash = compute_chunk_hash(abstract)

    # (#10) Check if already extracted
    existing = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
        EvidenceUnit.paper_id == paper.id,
        EvidenceUnit.source_chunk_hash == chunk_hash,
    ).count()
    if existing > 0:
        logger.debug("Paper %s: abstract already extracted, skipping", paper.id[:8])
        return 0

    try:
        result = await _llm_extract_evidence(llm, paper, abstract, "abstract")
        evidence_list = result.evidence_units
        evidence_count = 0
        for ev in evidence_list:
            span_pos = find_span_in_chunk(ev.original_span, abstract)
            eu = EvidenceUnit(
                task_id=task_id,
                paper_id=paper.id,
                evidence_type=ev.evidence_type,
                normalized_claim=ev.normalized_claim,
                original_span=ev.original_span[:500] if ev.original_span else "",
                section="abstract",
                page_number=None,
                page_start=None,
                page_end=None,
                span_start=span_pos[0] if span_pos else None,
                span_end=span_pos[1] if span_pos else None,
                source_chunk_hash=chunk_hash,
                dataset_name=ev.dataset_name,
                metric_name=ev.metric_name,
                result_value=ev.result_value,
                extraction_method="abstract_only",
                extraction_confidence=0.4,
                verification_status="abstract_only",
            )
            db.add(eu)
            evidence_count += 1
        db.flush()
        return evidence_count
    except Exception as e:
        logger.debug("Abstract evidence extraction failed for paper %s: %s", paper.id[:8], e)
        return 0


async def _upgrade_abstract_evidence(db, paper, task_id):
    """(#9) Mark abstract-only evidence as upgraded when full-text is available."""
    abstract_evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
        EvidenceUnit.paper_id == paper.id,
        EvidenceUnit.extraction_method == "abstract_only",
    ).all()

    for eu in abstract_evidence:
        eu.verification_status = "upgraded"
        eu.extraction_confidence = 0.6  # Slightly higher than abstract_only

    if abstract_evidence:
        logger.info("Paper %s: upgraded %d abstract-only evidence to 'upgraded'",
                    paper.id[:8], len(abstract_evidence))
    db.flush()


async def _llm_extract_evidence(llm, paper, text_chunk, section):
    """Call LLM to extract evidence units from a text chunk.

    O6: For abstract-only extraction, add an explicit instruction to surface
    implicit capability boundaries (scope statements like "we focus on X"
    imply "non-X is not covered"), which are a key source of gap signals when
    full text is unavailable. These are still tied to a concrete original_span.
    """
    section_hint = ""
    if section == "abstract":
        section_hint = (
            "\n\nNote: This is an abstract. In addition to explicit claims, surface any "
            "implicit capability boundary the abstract states — e.g. a scope restriction "
            "(\"we focus on ...\", \"limited to ...\", \"for the setting of ...\") implies the "
            "complementary setting is NOT addressed. Only do this when the scope wording is "
            "actually present in the text; classify such units as evidence_type \"limitation\"."
        )
    elif section == "conclusion":
        section_hint = (
            "\n\nNote: This section aggregates Discussion / Limitations / Threats to Validity / "
            "Future Work. Prioritize extracting every stated boundary, failure condition, "
            "assumption, and acknowledged untested setting as evidence_type "
            "\"limitation\", \"negative_result\", or \"future_work\"."
        )
    messages = [
        {"role": "system", "content": EVIDENCE_EXTRACT_SYSTEM},
        {"role": "user", "content": EVIDENCE_EXTRACT_USER.format(
            title=paper.title or "",
            section=section,
            text_chunk=text_chunk[:CHUNK_SIZE],
        ) + section_hint},
    ]
    return await llm.chat_json(messages, EvidenceExtractionList)


def _get_pdf_path(paper_id: str) -> str | None:
    """Get cached PDF path for a paper."""
    pdf_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))), "pdf_cache")
    path = os.path.join(pdf_cache_dir, f"{paper_id}.pdf")
    return path if os.path.exists(path) else None


def _parse_pdf_once(pdf_path: str) -> tuple[dict[str, str] | None, dict[int, str]]:
    """P1: Open the PDF exactly once and return both section text and per-page
    text, then close it. Replaces the old pattern of opening the same file twice
    (once for sections, once for page texts) and keeping everything resident.
    """
    import re
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available, cannot extract PDF text")
        return None, {}

    try:
        from app.agent.steps.analyze_papers import SECTION_KEYWORDS
    except Exception:
        SECTION_KEYWORDS = {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("Failed to open PDF %s: %s", pdf_path, e)
        return None, {}

    sections: dict[str, list[str]] = {
        "abstract": [], "introduction": [], "method": [],
        "experiment": [], "conclusion": [],
    }
    current_section = "introduction"
    page_texts: dict[int, str] = {}

    try:
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            page_texts[i + 1] = page_text
            for line in page_text.split("\n"):
                line_stripped = line.strip()
                if not line_stripped or len(line_stripped) > 80:
                    sections[current_section].append(line_stripped)
                    continue
                line_lower = line_stripped.lower()
                detected = False
                for sec_name, keywords in SECTION_KEYWORDS.items():
                    for kw in keywords:
                        if kw in line_lower and len(line_lower) < 60:
                            if (re.match(r"^(\d+\.?\s*)?" + re.escape(kw), line_lower)
                                    or line_lower.startswith(kw)):
                                current_section = sec_name
                                detected = True
                                break
                    if detected:
                        break
                if not detected:
                    sections[current_section].append(line_stripped)
    finally:
        doc.close()

    result: dict[str, str] = {}
    for sec_name, lines in sections.items():
        text = "\n".join(l for l in lines if l)
        if text.strip():
            result[sec_name] = text

    return (result or None), page_texts


def _extract_page_texts(pdf_path: str) -> dict[int, str]:
    """Extract text per page from PDF for page number mapping."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_texts = {}
        for i, page in enumerate(doc):
            page_texts[i + 1] = page.get_text("text")
        doc.close()
        return page_texts
    except Exception as e:
        logger.debug("Page text extraction failed: %s", e)
        return {}


def _find_page_for_text(chunk_text: str, page_texts: dict[int, str], section: str) -> int | None:
    """(#5) Find the actual PDF page number for a chunk of text."""
    if not page_texts:
        return None
    # Use first 100 chars of chunk to find page
    snippet = chunk_text[:100]
    for page_num, page_text in page_texts.items():
        if snippet in page_text:
            return page_num
    # Try last 100 chars
    snippet = chunk_text[-100:]
    for page_num, page_text in page_texts.items():
        if snippet in page_text:
            return page_num
    return None


async def _classify_paper_roles_safe(db, task_id, papers):
    """Classify papers into roles — uses main session."""
    from app.db.models import PaperRole

    for paper, tp in papers:
        # (#10) Check if roles already exist for this paper
        existing = db.query(PaperRole).filter(
            PaperRole.task_id == task_id,
            PaperRole.paper_id == paper.id,
        ).count()
        if existing > 0:
            continue

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
                reason=f"heuristic: {role} indicators in title/abstract",
            )
            db.add(pr)

    db.flush()
    db.commit()

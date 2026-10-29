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

from sqlalchemy.exc import OperationalError
from app.agent.state import ResearchState
from app.agent.prompts import EVIDENCE_EXTRACT_SYSTEM, EVIDENCE_EXTRACT_USER
from app.db.models import (
    Paper, TaskPaper, EvidenceUnit, PaperRole, ResearchQuestion,
)
from app.db.session import SessionLocal
from app.db.lock_retry import (
    ASYNC_BACKOFF, _sleep_for, commit_with_retry, flush_with_retry, is_lock_error,
)
from app.db.repositories import paper_repo
from app.schemas.schemas import EvidenceExtractionList
from app.services.event_service import emit_event

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4000


def compute_chunk_hash(text: str) -> str:
    """SHA-256 of chunk text for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locate_span_in_chunk(original_span: str, chunk_text: str) -> tuple[int, int, str] | None:
    """(#6) Validate that original_span actually exists in the source chunk.

    Returns (start, end, match_quality) where match_quality is:
      - "exact": the span was found verbatim in the chunk
      - "anchored": the span was located via its leading/trailing 80 chars
        (the LLM slightly reworded the middle), so the offsets are anchor-based
    Returns None when the span cannot be located at all.
    """
    if not original_span or not chunk_text:
        return None
    # Try exact match first
    pos = chunk_text.find(original_span)
    if pos >= 0:
        return pos, pos + len(original_span), "exact"
    # Try first 80 chars (LLM may have slightly modified the span)
    snippet = original_span[:80]
    if snippet:
        pos = chunk_text.find(snippet)
        if pos >= 0:
            return pos, pos + len(original_span), "anchored"
    # Try last 80 chars
    snippet = original_span[-80:]
    if snippet:
        pos = chunk_text.find(snippet)
        if pos >= 0:
            return pos, pos + len(original_span), "anchored"
    return None


def find_span_in_chunk(original_span: str, chunk_text: str) -> tuple[int, int] | None:
    """Offsets-only wrapper around locate_span_in_chunk (kept for callers that
    do not care about match quality)."""
    located = locate_span_in_chunk(original_span, chunk_text)
    return (located[0], located[1]) if located else None


# Why each paper's PDF could not be retrieved in the most recent extraction run,
# keyed by task id then paper id. Threaded from the download step to the
# persistence step so the cause survives into `task_papers`.
_download_diagnoses: dict[str, dict[str, str | None]] = {}


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
    from app.services.rag_service import download_pdf_classified
    from app.config import settings

    # Get papers to process — the *whole* candidate set, not a per-round slice.
    #
    # Two allocation rules used to fight each other and both are now gone:
    #   1. A per-round `evidence_max_papers` cap. With 146 candidates and a cap of
    #      15, a five-round task could assess at most 75 papers — the corpus was
    #      unreachable by construction, and which papers got read depended on how
    #      many rounds happened to run.
    #   2. A priority-tier quota on top of that cap. It correctly stopped mid-tier
    #      papers from crowding out topic-critical ones, but it did so by dropping
    #      papers rather than by making extraction cheaper.
    #
    # Coverage is now unconditional: every candidate is extracted. What made that
    # affordable is concurrency (see the pool below) plus the fact that a paper
    # whose evidence is already stored is skipped cheaply — so re-entering this
    # phase in a later round costs almost nothing for papers already done, and
    # rounds accumulate coverage instead of re-rolling the same truncated slice.
    query = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_id,
        TaskPaper.priority.in_(["high", "medium"]),
    )
    # Round > 0 means a "continue/regenerate" pass: new papers only. But round 0
    # re-runs (resume, or a second attempted solution) must see the full corpus,
    # so only the incremental case narrows.
    if round_number > 0:
        query = query.filter(TaskPaper.discovered_round == round_number)

    all_tps = query.order_by(TaskPaper.final_score.desc().nullslast()).all()
    papers = [(tp.paper, tp) for tp in all_tps if tp.paper]

    if not papers:
        logger.info("Task %s: no papers for evidence extraction (round %d)", task_id[:8], round_number)
        return 0

    n_high = sum(1 for tp in all_tps if tp.priority == "high")
    logger.info(
        "Task %s: extracting evidence from all %d candidate papers (round %d) — "
        "high=%d medium=%d, concurrency=%d",
        task_id[:8], len(papers), round_number, n_high, len(all_tps) - n_high,
        settings.evidence_extract_concurrency,
    )

    emit_event(task_id, "status", {"status": "extracting_evidence", "total": len(papers), "round": round_number})

    # Prewarm: fetch the whole round's PDFs before any extraction work begins.
    #
    # This used to be the same call as the one below, i.e. downloads happened as
    # the first thing extraction did. Splitting them matters because the two have
    # different time/cost profiles: downloading is network-bound and recoverable
    # (retries absorb rate-limit windows), extraction is LLM-bound and expensive.
    # Interleaving them means a transient retrieval failure becomes permanent
    # evidence loss, and the round's duration is unpredictable. Prewarming makes
    # the corpus a settled input: by the time extraction runs, availability is
    # already known and disclosed.
    if settings.enable_pdf_prewarm:
        prewarm = await prewarm_pdfs(papers, task_id)
        logger.info(
            "Task %s: prewarm complete — %d/%d retrievable, %d unavailable "
            "(%s); extraction proceeds with what is on disk",
            task_id[:8], prewarm["retrieved"], prewarm["total"],
            prewarm["unavailable"],
            ", ".join("{}={}".format(k, v) for k, v in sorted(prewarm["reasons"].items())) or "none",
        )

    # (#4) Download PDFs first (before extraction)
    pdf_paths = await _download_pdfs(
        papers, task_id, concurrency=settings.pdf_prewarm_concurrency)

    # One shared semaphore spanning the whole paper set, rather than a fresh
    # per-batch one.
    #
    # The old code created a `Semaphore(batch_size)` inside each batch and
    # awaited the batch before starting the next, so concurrency never exceeded
    # `batch_size` and the duration scaled as ceil(N/batch_size) x per-paper-time.
    # A single pool keeps exactly `evidence_extract_concurrency` papers in flight
    # continuously: no barrier between batches, so a slow paper no longer holds
    # back the papers queued behind it, and throughput no longer degrades as the
    # N/1 remainder shrinks.
    #
    # Peak cost is bounded by the sampling of that pool, and each paper carries
    # its own timeout (`evidence_paper_timeout_s`) inside `_extract_from_paper_safe`,
    # so a hung PDF cannot stall the round.
    concurrency = max(1, settings.evidence_extract_concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    total_evidence = 0
    done = 0

    results = await asyncio.gather(*[
        _extract_from_paper_safe(
            task_id, paper, tp, pdf_paths.get(paper.id), llm, round_number, semaphore
        )
        for paper, tp in papers
    ], return_exceptions=True)

    for r in results:
        done += 1
        if isinstance(r, int):
            total_evidence += r
        elif isinstance(r, Exception):
            logger.error("Evidence extraction task failed: %s", r)

    # Single commit after the pool drains.
    #
    # The old code committed per batch specifically because a write transaction
    # held open across later minutes-long LLM calls starves concurrent tasks past
    # busy_timeout (2026-09-09 3-way retest: two of three tasks died "database is
    # locked"). That reasoning is preserved rather than dropped: every per-paper
    # write happens in its own short-lived session (`_extract_from_paper_safe`
    # opens and closes a SessionLocal per paper) which is what bounds the write
    # transaction. This session only writes the phase-tail trace, so it stays
    # outside the LLM window entirely.
    db.commit()
    logger.info("Task %s: evidence extraction pool drained, %d papers, cumulative=%d",
                task_id[:8], done, total_evidence)

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


async def prewarm_pdfs(papers, task_id: str) -> dict:
    """Fetch this round's PDFs before extraction, and report what is retrievable.

    Purpose is *decision*, not just motion: the caller learns the corpus that
    actually exists before spending LLM budget on it, and the number plus the
    reason distribution are surfaced in the phase summary so a truncated corpus
    is visible instead of silently reducing evidence yield.

    Downloads are idempotent against `pdf_cache`, so papers already on disk cost
    one `os.path.exists` and no network. Returns a summary dict:
    {"total", "retrieved", "unavailable", "reasons": {reason: count}}.
    """
    from app.services.rag_service import PDF_RETRY_MAX_ATTEMPTS, PDF_RETRY_BACKOFF_SECONDS
    from app.config import settings

    papers = [(p, tp) for p, tp in papers if p is not None]
    if not papers:
        return {"total": 0, "retrieved": 0, "unavailable": 0, "reasons": {}}

    # A prewarm batch has a bounded worst case: every paper needs its full retry
    # budget plus inter-attempt backoff. Logging the bound up front keeps the
    # worst-case delay visible rather than showing up as an unexplained stall.
    worst_case = len(papers) / max(1, settings.pdf_prewarm_concurrency) * (
        PDF_RETRY_MAX_ATTEMPTS * 30 + sum(PDF_RETRY_BACKOFF_SECONDS))
    logger.info(
        "Task %s: prewarming %d PDFs (concurrency %d, worst case ~%.0fs before extraction)",
        task_id[:8], len(papers), settings.pdf_prewarm_concurrency, worst_case,
    )
    emit_event(task_id, "status", {
        "status": "prewarming_pdfs", "total": len(papers), "round": None,
    })

    pdf_paths = await _download_pdfs(
        papers, task_id, concurrency=settings.pdf_prewarm_concurrency)

    diagnoses = _download_diagnoses.get(task_id, {})
    reasons: dict[str, int] = {}
    for paper, _tp in papers:
        if pdf_paths.get(paper.id):
            continue
        reason = diagnoses.get(paper.id) or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1

    retrieved = sum(1 for paper, _tp in papers if pdf_paths.get(paper.id))
    summary = {
        "total": len(papers),
        "retrieved": retrieved,
        "unavailable": len(papers) - retrieved,
        "reasons": reasons,
    }
    emit_event(task_id, "status", {
        "status": "prewarm_done",
        "retrieved": retrieved,
        "total": len(papers),
    })
    return summary


async def _download_pdfs(papers, task_id: str, concurrency: int = 3) -> dict[str, str]:
    """(#4) Download PDFs for papers before extraction."""
    from app.services.rag_service import download_pdf_classified

    pdf_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))), "pdf_cache")
    os.makedirs(pdf_cache_dir, exist_ok=True)

    pdf_paths = {}
    diagnoses = {}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def download_one(paper):
        async with semaphore:
            cache_path = os.path.join(pdf_cache_dir, f"{paper.id}.pdf")
            if os.path.exists(cache_path):
                return paper.id, cache_path, None
            try:
                outcome = await download_pdf_classified(paper)
                if outcome.content:
                    with open(cache_path, "wb") as f:
                        f.write(outcome.content)
                    return paper.id, cache_path, None
                # No bytes: the outcome says *why*, and that reason is persisted
                # by the caller. Previously this was a bare `return None` with no
                # record, so an unretrievable paper was re-attempted every round
                # and stayed indistinguishable from one that was read and found
                # empty. Measured impact before this: 42 of 122 papers with a
                # pdf_url had a cached PDF and only 22 produced any evidence.
                logger.info("PDF unavailable for paper %s (reason=%s)",
                            paper.id[:8], outcome.gap_reason)
                return paper.id, None, outcome.gap_reason
            except Exception as e:
                logger.warning("PDF download failed for paper %s: %s: %s",
                               paper.id[:8], type(e).__name__, e)
            return paper.id, None, "unknown"

    results = await asyncio.gather(*[download_one(p) for p, _ in papers], return_exceptions=True)
    for r in results:
        if isinstance(r, tuple) and len(r) == 3:
            pdf_paths[r[0]] = r[1]
            diagnoses[r[0]] = r[2]

    # Aggregate so the phase summary distinguishes "we tried and failed" from
    # "there was nothing to try". Without this, a round where 80 of 122 papers
    # could not be retrieved reads exactly like a round that simply found little
    # evidence — which is how a retrieval failure turns into a scientific claim.
    retrieved = sum(1 for v in pdf_paths.values() if v)
    logger.info("PDF availability: %d/%d papers retrieved (task %s)",
                retrieved, len(papers), task_id[:8])
    _download_diagnoses[task_id] = diagnoses
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


def _record_extraction_gap(db, tp, status: str, reason: str | None) -> None:
    """Record whether this paper's full text was actually assessed.

    Needed so that "we could not retrieve/parse the full text" stays distinct
    from "we read the full text and found nothing" all the way downstream.
    Best-effort: a bookkeeping write must never break extraction itself.
    """
    try:
        tp.extraction_status = status
        tp.extraction_gap_reason = reason
    except Exception:  # pragma: no cover - defensive, columns may be absent pre-migration
        logger.debug("Could not record extraction status for task_paper %s", getattr(tp, "id", "?"))


# Verification states that mean the unit carries a locatable source span and is
# therefore admissible downstream. Observed values in a production DB:
# `verified` (span located in the source text), `abstract_only` (derived from an
# abstract, not a full text) and `unverified` (span could not be located).
# `upgraded` is set by `_upgrade_abstract_evidence` once an abstract-level unit
# has been superseded by a full-text one, so it counts as usable too.
#
# `unverified` does not qualify: a paper whose units were all rejected had a
# failed extraction and must be retried. Neither does `abstract_only` on its own
# — it should be retried once a PDF exists so the abstract-level result can be
# upgraded. Including either would freeze a degraded result in place forever.
_USABLE_VERIFICATION_STATUSES = ("verified", "upgraded")


def _has_usable_evidence(db, task_id: str, paper_id: str) -> bool:
    """Is usable evidence for this paper already recorded for this task?

    Counts units that already have a locatable source span. Note that
    `abstract_only` is excluded on purpose: it marks an abstract-level result,
    and a later round that has since obtained the PDF should be free to upgrade
    it. `upgraded` is included because it means the abstract-level unit was
    already replaced by a full-text one.

    Reads through the ORM rather than issuing raw SQL, so the caller may pass
    either a Session or a plain connection; when `db` is None (the per-paper
    worker runs on its own session) a short-lived session is opened instead.

    Never raises: any failure is reported as "not extracted". The skip is an
    optimisation, so a lookup problem must cost time, never silently drop a paper
    from the corpus.
    """
    try:
        from app.db.models import EvidenceUnit
        from app.db.session import SessionLocal

        def _query(session):
            return (session.query(EvidenceUnit.id)
                    .filter(EvidenceUnit.task_id == task_id,
                            EvidenceUnit.paper_id == paper_id,
                            EvidenceUnit.verification_status.in_(_USABLE_VERIFICATION_STATUSES))
                    .first() is not None)

        if db is not None:
            return _query(db)
        session = SessionLocal()
        try:
            return _query(session)
        finally:
            session.close()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Evidence lookup failed for paper %s: %s", str(paper_id)[:8], e)
        return False


async def _extract_from_paper(db, llm, task_id, paper, tp, pdf_path, round_number) -> int:
    """Extract evidence from a single paper with provenance validation.

    P1: the PDF is parsed exactly once (sections + per-page text together) and
    released immediately, instead of opening the same file twice and keeping all
    page text resident across the whole round.
    """
    # Skip work already done. Now that the phase targets the *whole* corpus on
    # every entry, a resume or a later round would otherwise re-parse and re-send
    # every paper it already processed — turning full coverage into a repeated
    # full cost.
    #
    # The expensive part of extraction is the LLM call, so the skip is keyed on
    # whether usable evidence is *already recorded* for this paper, not on a
    # provenance flag alone. Keying it on `extraction_status` had two failures:
    #   - It applies per TaskPaper, but a PDF belongs to a Paper. Within one task
    #     a paper can be reached on several rounds, and each round used to carry a
    #     fresh TaskPaper, so the same paper was re-extracted repeatedly — paying
    #     full LLM cost to rediscover claims that were already stored.
    #   - `abstract_only` was deliberately re-processed so a later round could
    #     upgrade it. That is worth the cost only when a full text now exists; if
    #     the PDF is still missing, re-running reproduces the same abstract-level
    #     result at full price.
    #
    # Scope note: the lookup is scoped to this task, and that scoping is load
    # bearing. Evidence rows are owned by the task that produced them and
    # coverage counts this task's own units, so treating "another task already
    # read this paper" as "skip it" would leave this task with zero rows for that
    # paper — indistinguishable from having found nothing, i.e. a fresh instance
    # of the absence-of-evidence error this module keeps having to avoid.
    #
    # The cost that scoping leaves on the table is bounded and real: a second
    # task touching the same paper re-reads it (34% of (task, paper) pairs
    # overlap across tasks in a production DB). Collapsing that requires promoting
    # evidence to a corpus-level asset with shared ownership, which trades the
    # self-containment above for a dedup win.
    if _has_usable_evidence(db, task_id, paper.id):
        logger.debug("Skipping paper %s: evidence already recorded for this task", paper.id[:8])
        return 0

    evidence_count = 0

    sections = None
    page_texts: dict[int, str] = {}
    if pdf_path and os.path.exists(pdf_path):
        try:
            sections, page_texts = _parse_pdf_once(pdf_path)
        except Exception as e:
            logger.debug("PDF parse failed for paper %s: %s", paper.id[:8], e)

    if sections:
        # Full text was actually assessed — record it so this paper is never
        # conflated with one whose PDF was never retrieved.
        _record_extraction_gap(db, tp, "full_text", None)

        # (#5) Use real PDF page numbers
        evidence_count += await _extract_from_sections(
            db, llm, task_id, paper, sections, page_texts, round_number
        )

        # (#9) Upgrade abstract-only evidence to full-text
        await _upgrade_abstract_evidence(db, paper, task_id)
    else:
        # Abstract fallback.
        #
        # Reaching this branch means the full text was never assessed — either
        # the PDF could not be downloaded or it failed to parse. That is a
        # different fact from "the full text was read and contained nothing
        # relevant" ("not tested" vs "tested negative"), and the two used to be
        # indistinguishable: both produced zero evidence units and left
        # `extraction_method` unset. Downstream coverage then treated an
        # unretrievable paper exactly like an empty one, biasing conclusions
        # toward "no prior work exists". The gap is recorded on the TaskPaper so
        # the readiness gate and reports can disclose it.
        abstract = paper.abstract or ""
        # Prefer the download verdict over a generic label: "paywall" is
        # actionable (stop retrying, disclose it) whereas "pdf_not_retrieved" is
        # not. Falls back cleanly when the download step recorded nothing.
        verdict = _download_diagnoses.get(task_id, {}).get(paper.id)
        if pdf_path:
            gap_reason = "pdf_parse_failed"
        else:
            gap_reason = verdict or "pdf_not_retrieved"
        if len(abstract) > 50:
            evidence_count += await _extract_from_abstract(db, llm, task_id, paper, abstract, round_number)
            _record_extraction_gap(db, tp, "abstract_only", gap_reason)
        else:
            _record_extraction_gap(db, tp, "not_assessed", gap_reason)

    # P1: free page text ASAP so peak memory is bounded per paper.
    page_texts.clear()
    if sections:
        sections.clear()

    db.commit()
    return evidence_count


def interleave_section_chunks(by_section: dict[str, list], max_chunks: int,
                              section_priority: dict[str, int]) -> list:
    """Spend a paper's chunk budget across sections instead of on the longest one.

    Sections are visited in priority order and contribute one chunk per pass, so
    the budget truncates each section's tail rather than dropping whole sections.
    Gap mining needs both halves of a paper: limitation statements (conclusion /
    discussion) and measured comparisons (method / experiment).
    """
    ordered = sorted(by_section, key=lambda name: section_priority.get(name, 5))
    selected: list = []
    depth = 0
    while len(selected) < max_chunks:
        progressed = False
        for name in ordered:
            if len(selected) >= max_chunks:
                break
            chunks = by_section[name]
            if depth < len(chunks):
                selected.append(chunks[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return selected


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
    #    skipping already-extracted chunks.
    section_priority = {
        # Conclusion first: it aggregates Discussion / Limitations / Threats to
        # Validity / Future Work, and gap mining cannot admit a research question
        # without a limitation-type signal. Method and experiment then supply the
        # comparisons, datasets and metrics the same admission needs.
        "conclusion": 0, "method": 1, "experiment": 2,
        "abstract": 3, "introduction": 4,
    }
    by_section: dict[str, list] = {}
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
            # Dedup is scoped to (task, paper, chunk_hash) — i.e. "has this task
            # already processed this chunk", not "has this chunk produced this
            # claim". The distinction matters because a chunk legitimately yields
            # many claims: measured in a production DB, every one of the 6,437
            # multi-row (task, paper, chunk) groups held *distinct* claims (one
            # chunk produced 29 of them, a related-work paragraph enumerating
            # baselines). Deduping on the claim would silently collapse an
            # enumerating chunk down to a single row.
            #
            # Not widened to "any task that read this chunk" either: evidence rows
            # stay owned by the task that produced them, so skipping a chunk
            # because another task read it would leave this task with no rows for
            # that paper, which coverage cannot distinguish from having found
            # nothing.
            existing = db.query(EvidenceUnit).filter(
                EvidenceUnit.task_id == task_id,
                EvidenceUnit.paper_id == paper.id,
                EvidenceUnit.source_chunk_hash == chunk_hash,
            ).first()
            if existing:
                continue
            by_section.setdefault(sec_name, []).append((prio, sec_name, chunk_text, chunk_hash))

    # Take chunks round-robin across sections instead of by global priority. A
    # long method section used to consume the whole per-paper budget, so
    # conclusions never reached the model at all: on a real run
    # limitation/negative_result units were 8% of the extracted evidence and 9
    # of 12 research questions then failed the (pre-v7) question-level
    # limitation gate. Since opportunity-ontology-v7 the limitation signal is
    # a per-type candidate contract (boundary_gap / missing_evaluation), but
    # the section rotation stays: those lenses still cannot be satisfied
    # without conclusion-section evidence.
    candidates = interleave_section_chunks(by_section, max_chunks, section_priority)
    if not candidates:
        db.commit()
        return 0

    # Release the WAL read snapshot BEFORE the (minutes-long) LLM extraction.
    # The dedup SELECTs above opened a read transaction that used to be held
    # across `await gather(...)`: committing a write on top of a stale snapshot
    # raises SQLITE_BUSY_SNAPSHOT, which does NOT honor busy_timeout and
    # silently discarded the whole paper's evidence on commit ("database is
    # locked", observed 2026-08-26 and 2026-09-04). All reads are done; the
    # write transaction opened below is short, so busy_timeout applies.
    db.commit()

    # 2) Extract chunks concurrently (bounded), then persist results in the
    #    caller's session serially (SQLAlchemy sessions are not thread-safe, but
    #    these coroutines share one event loop; DB writes happen after gather).
    sem = asyncio.Semaphore(chunk_concurrency)

    async def _extract_one(section_name, chunk_text):
        async with sem:
            try:
                result = await _llm_extract_evidence(llm, paper, chunk_text, section_name)
                # _llm_extract_evidence returns an EvidenceExtractionList model;
                # the actual units are on .evidence_units.
                return result.evidence_units
            except Exception as e:
                logger.debug("Chunk extraction failed for paper %s: %s", paper.id[:8], e)
                return []

    # The section name drives the per-section extraction hints. It used to be
    # hard-coded to "", so the hints written for conclusion and abstract — the
    # two sections that carry limitation statements — never applied to anything.
    extraction_results = await asyncio.gather(
        *[_extract_one(c[1], c[2]) for c in candidates]
    )

    evidence_count = 0
    _pending_evidence = []
    for (prio, sec_name, chunk_text, chunk_hash), evidence_list in zip(candidates, extraction_results):
        page_num = _find_page_for_text(chunk_text, page_texts, sec_name)
        for ev in evidence_list:
            normalized_claim = (ev.normalized_claim or "").strip()
            if len(normalized_claim) < 5:
                logger.debug("Paper %s: evidence missing a valid normalized_claim, skipping",
                             paper.id[:8])
                continue
            # (#6) Validate original_span exists in chunk
            located = locate_span_in_chunk(ev.original_span, chunk_text)
            if located is None:
                logger.debug("Paper %s: original_span not found in chunk, skipping evidence",
                             paper.id[:8])
                continue
            span_start, span_end, match_quality = located

            # The span was just located inside the very chunk it was extracted
            # from, and the unit carries full provenance (chunk hash, page,
            # offsets). That is exactly what downstream gap admission means by
            # "full-text locatable", so record it as verified instead of the
            # previous hard-coded "unverified" — which silently made every
            # full-text unit inadmissible for gap mining. Confidence reflects
            # how strong the match was (exact vs anchor-based).
            eu = EvidenceUnit(
                task_id=task_id,
                paper_id=paper.id,
                evidence_type=ev.evidence_type,
                normalized_claim=normalized_claim,
                original_span=ev.original_span[:500] if ev.original_span else "",
                section=sec_name,
                page_number=page_num,
                page_start=page_num,
                page_end=page_num,
                span_start=span_start,
                span_end=span_end,
                source_chunk_hash=chunk_hash,
                dataset_name=ev.dataset_name,
                metric_name=ev.metric_name,
                result_value=ev.result_value,
                extraction_method="pdf_fulltext",
                extraction_confidence=0.9 if match_quality == "exact" else 0.7,
                verification_status="verified",
                discovered_round=round_number,
            )
            _pending_evidence.append(eu)
            evidence_count += 1

    # Lock retry (2026-09-10, task ea57b3c0): a sibling task's long write
    # transaction (or a filesystem scanner on the DB file) can hold the write
    # lock past busy_timeout — the old code lost the WHOLE paper's evidence
    # to one locked flush (five papers dropped in one extract phase). The
    # multi-row INSERT is milliseconds once the lock frees, so rollback +
    # re-add + retry almost always succeeds.
    #
    # Retries are async (asyncio.sleep) rather than the sync helper's
    # time.sleep: this runs on the event loop, so a blocking wait would stall
    # every other task and HTTP request for the whole backoff.
    for attempt in range(len(ASYNC_BACKOFF) + 1):
        try:
            for eu in _pending_evidence:
                db.add(eu)
            db.flush()
            break
        except OperationalError as exc:
            if not is_lock_error(exc) or attempt == len(ASYNC_BACKOFF):
                raise
            db.rollback()
            await asyncio.sleep(_sleep_for(ASYNC_BACKOFF, attempt))
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

    # Same read/write split as _extract_from_sections: drop the dedup read
    # snapshot before the LLM call so the eventual write commit opens a fresh
    # short transaction instead of upgrading a minutes-old stale one.
    db.commit()

    try:
        result = await _llm_extract_evidence(llm, paper, abstract, "abstract")
        evidence_list = result.evidence_units
        evidence_count = 0
        for ev in evidence_list:
            normalized_claim = (ev.normalized_claim or "").strip()
            if len(normalized_claim) < 5:
                logger.debug("Paper %s: abstract evidence missing a valid normalized_claim, skipping",
                             paper.id[:8])
                continue
            span_pos = find_span_in_chunk(ev.original_span, abstract)
            eu = EvidenceUnit(
                task_id=task_id,
                paper_id=paper.id,
                evidence_type=ev.evidence_type,
                normalized_claim=normalized_claim,
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
                discovered_round=round_number,
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
    elif section in ("method", "experiment"):
        section_hint = (
            "\n\nNote: Besides what the work achieves, record the boundaries it states here: the "
            "settings, datasets, model scales and baselines it actually evaluates, and any "
            "restriction or assumption that is literally written in the text (\"we assume ...\", "
            "\"only for ...\", \"we do not evaluate ...\"). Measured comparisons against named "
            "baselines are what make a claim checkable — classify those as \"comparison\" and "
            "stated restrictions as \"limitation\". Never infer a restriction that the text does "
            "not state."
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

    flush_with_retry(db)
    commit_with_retry(db)

"""Backfill full-text chunks for papers the super audit needs to adjudicate.

Why this exists
---------------
The audit's blind review sheet shows a quotable passage next to each candidate
so a human can decide FULL / PARTIAL / NONE. That requires full text, and full
text coverage is only 2.2% (559 / 24,952 papers), which is why the 2026-09-16
audit produced snippets for just 8.8% of targets.

Coverage is low for two compounding reasons:

1. `ENABLE_RAG_INDEXING=false` in .env skips the whole PDF pipeline. It was
   turned off because PyMuPDF/PyTorch parsing could native-segfault under
   Windows, killing the process with no catchable exception.
2. Even enabled, `_run_rag_indexing` indexes only `limit(20)` high-priority
   papers PER TASK (`runner.py`), which caps global coverage no matter how many
   topics run.

The audit has a different requirement from the production pipeline: production
needs *some* grounding per topic and falls back to abstracts, whereas the audit
needs the specific candidate papers it surfaced to be quotable.

So this tool is deliberately separate from the production chain:
- it targets the paper set the audit actually cites, not task priority;
- it runs as its own process, so a parse segfault cannot take down the API;
- it writes chunks only; embedding/ChromaDB is skipped because the audit reads
  `paper_chunks.text` directly and never does vector retrieval.

Usage:
    cd backend
    python -m eval.production_e2e.backfill_pdf_chunks --audit-dir <audit run dir> --limit 8
    python -m eval.production_e2e.backfill_pdf_chunks --audit-dir <dir>            # all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("backfill_pdf")


def _collect_audit_candidates(audit_dir: Path) -> tuple[set[str], set[str]]:
    """Paper ids and normalised titles referenced by an audit run's records.

    Two keys are needed because audit records predating 2026-09-16 do not carry
    `local_paper_id` (search_candidates dropped it when building candidates), so
    those runs can only be joined by title. Titles are a weaker key, hence they
    are matched on `normalized_title` and only used when an id is unavailable.
    """
    records_path = audit_dir / "candidate_killer_papers.jsonl"
    if not records_path.exists():
        raise SystemExit(f"[backfill] no audit records at {records_path}")
    ids: set[str] = set()
    titles: set[str] = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for cand in (rec.get("candidate_papers") or []):
            pid = cand.get("local_paper_id")
            if pid:
                ids.add(pid)
            title = (cand.get("title") or "").strip()
            if title:
                titles.add(" ".join(title.lower().split()))
    return ids, titles


def _select_targets(db, ids: set[str], titles: set[str], limit: int | None) -> list:
    """Papers matching the audit's candidates that lack chunks but have a PDF.

    Ordering prefers host classes that are actually downloadable. A first probe
    (8 papers) spent its whole budget on publisher sites that answer 403/303 to
    non-browser clients (ScienceDirect, Springer behind an IdP), so ordering by
    citation count alone wastes the run on the papers most likely to fail.
    arXiv/OpenReview/PMC/ACL host open PDFs.
    """
    from sqlalchemy import text

    clauses = []
    params: dict = {}
    if ids:
        placeholders = ", ".join(f":p{i}" for i in range(len(ids)))
        params.update({f"p{i}": pid for i, pid in enumerate(ids)})
        clauses.append(f"p.id IN ({placeholders})")
    if titles:
        placeholders = ", ".join(f":t{i}" for i in range(len(titles)))
        params.update({f"t{i}": t for i, t in enumerate(titles)})
        clauses.append(f"LOWER(TRIM(p.title)) IN ({placeholders})")
    if not clauses:
        return []

    open_hosts = ("arxiv.org", "openreview.net", "ncbi.nlm.nih.gov",
                  "aclweb.org", "biorxiv.org", "hal.science", "europepmc.org")
    host_case = " ".join(
        f"WHEN p.pdf_url LIKE '%{h}%' THEN 0" for h in open_hosts)

    sql = f"""
        SELECT p.id, p.title, p.pdf_url
        FROM papers p
        WHERE ({' OR '.join(clauses)})
          AND p.pdf_url IS NOT NULL AND p.pdf_url != ''
          AND NOT EXISTS (SELECT 1 FROM paper_chunks pc WHERE pc.paper_id = p.id)
        ORDER BY
          CASE {host_case} ELSE 1 END,
          p.citation_count DESC NULLS LAST
    """
    rows = db.execute(text(sql), params).fetchall()
    return rows[:limit] if limit else rows


async def _parse_one(paper, llm) -> tuple[bool, int, str]:
    """Download + parse a single PDF into chunks. Returns (ok, n_chunks, note).

    Deliberately uses the plain-text PyMuPDF path only: no inline figure
    extraction, no VLM calls, no pdfplumber tables, no PaddleOCR. Those stages
    are what the audit does not need and what historically destabilised the
    pipeline. Chunk quality is adequate for adjudicating a claim.
    """
    import httpx

    paper_id = paper.id
    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
            resp = await client.get(paper.pdf_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; research-audit/1.0)"})
            resp.raise_for_status()
            pdf_bytes = resp.content
        if not pdf_bytes:
            return False, 0, "empty response"
    except Exception as exc:
        return False, 0, f"download: {type(exc).__name__}: {str(exc)[:80]}"

    try:
        from app.services.rag_service import _split_into_chunks
        chunks = await asyncio.to_thread(_extract_plain_chunks, pdf_bytes, paper_id)
    except Exception as exc:
        return False, 0, f"parse: {type(exc).__name__}: {str(exc)[:80]}"

    if not chunks:
        return False, 0, "no extractable text (scanned?)"

    try:
        from app.db.session import SessionLocal
        from app.db.repositories.paper_repo import save_chunks
        # `_split_into_chunks` returns ParsedChunk dataclasses; save_chunks takes
        # dicts (same conversion the production path does).
        chunks_data = [{
            "chunk_index": c.chunk_index,
            "section": c.section,
            "chunk_type": c.chunk_type,
            "text": c.text,
            "image_paths": c.image_paths,
            "page_number": c.page_number,
            "word_count": c.word_count,
            "has_pdf": c.has_pdf,
            "extraction_method": c.extraction_method,
        } for c in chunks]
        db = SessionLocal()
        try:
            save_chunks(db, paper_id, chunks_data)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        return False, 0, f"save: {type(exc).__name__}: {str(exc)[:80]}"

    return True, len(chunks), ""


def _extract_plain_chunks(pdf_bytes: bytes, paper_id: str) -> list:
    """Text-only chunking. Runs in a worker thread (CPU-bound, C library)."""
    import fitz
    from app.services.rag_service import ParsedChunk, _split_into_chunks

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        stream = "\n".join(page.get_text("text") for page in doc)
    finally:
        # C resource; without an explicit close it survives until GC.
        doc.close()

    if len(stream.strip()) < 500:
        return []
    return _split_into_chunks(stream, paper_id)


async def _run(args) -> None:
    from app.db.session import SessionLocal

    audit_dir = Path(args.audit_dir)
    ids, titles = _collect_audit_candidates(audit_dir)
    print(f"[backfill] audit candidates: {len(ids)} with a local id, "
          f"{len(titles)} distinct titles")

    db = SessionLocal()
    try:
        targets = _select_targets(db, ids, titles, args.limit)
    finally:
        db.close()

    print(f"[backfill] {len(targets)} paper(s) need full text "
          f"(have pdf_url, no chunks)")
    if args.dry_run:
        for p in targets:
            print(f"   {p.id[:8]} | {(p.title or '')[:70]}")
        return
    if not targets:
        return

    llm = None
    ok = failed = 0
    total_chunks = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    for i, paper in enumerate(targets, 1):
        started = time.time()
        success, n_chunks, note = await _parse_one(paper, llm)
        if success:
            ok += 1
            total_chunks += n_chunks
            print(f"[{i}/{len(targets)}] OK   {paper.id[:8]} {n_chunks:3d} chunks "
                  f"({time.time() - started:.1f}s) {(paper.title or '')[:48]}")
        else:
            failed += 1
            key = note.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
            print(f"[{i}/{len(targets)}] FAIL {paper.id[:8]} {note}")

    print()
    print(f"[backfill] done in {time.time() - t0:.0f}s: "
          f"{ok} ok / {failed} failed, {total_chunks} chunks")
    if reasons:
        print(f"[backfill] failure reasons: {reasons}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill full-text chunks for audit candidate papers")
    p.add_argument("--audit-dir", required=True,
                   help="audit run dir containing candidate_killer_papers.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of papers to parse (for a small-batch probe)")
    p.add_argument("--dry-run", action="store_true",
                   help="list the papers that would be parsed and exit")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

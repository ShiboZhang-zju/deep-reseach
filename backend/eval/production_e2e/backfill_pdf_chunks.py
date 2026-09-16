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
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger("backfill_pdf")

# Mean chunk length below this marks a paper as still carrying output from the
# pre-fix extractor (~65 words/chunk) rather than a re-parsed one (~200). Used to
# resume an interrupted --reparse without needing a migration marker column.
DEGRADED_AVG_WORDS = 120.0


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


def _to_chunk_dicts(chunks: list) -> list[dict]:
    """ParsedChunk dataclasses -> dicts (save_chunks takes dicts)."""
    return [{
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


async def _build_chunks(paper) -> tuple[bool, list, str]:
    """Download and parse, WITHOUT touching the database.

    Split out from storage so `--reparse` can build the replacement text first
    and only then delete the old chunks.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
            resp = await client.get(paper.pdf_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; research-audit/1.0)"})
            resp.raise_for_status()
            pdf_bytes = resp.content
        if not pdf_bytes:
            return False, [], "empty response"
    except Exception as exc:
        return False, [], f"download: {type(exc).__name__}: {str(exc)[:80]}"

    try:
        chunks = await asyncio.to_thread(_extract_plain_chunks, pdf_bytes, paper.id)
    except Exception as exc:
        return False, [], f"parse: {type(exc).__name__}: {str(exc)[:80]}"

    if not chunks:
        return False, [], "no extractable text (scanned?)"
    return True, chunks, ""


async def _persist_chunks(paper, chunks: list) -> tuple[bool, int, str]:
    from app.db.session import SessionLocal
    from app.db.repositories.paper_repo import save_chunks

    try:
        db = SessionLocal()
        try:
            save_chunks(db, paper.id, _to_chunk_dicts(chunks))
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        return False, 0, f"save: {type(exc).__name__}: {str(exc)[:80]}"
    return True, len(chunks), ""


async def _parse_one(paper, llm) -> tuple[bool, int, str]:
    """Download, parse, and store. Returns (ok, n_chunks, note)."""
    success, chunks, note = await _build_chunks(paper)
    if not success:
        return False, 0, note
    return await _persist_chunks(paper, chunks)


def _extract_plain_chunks(pdf_bytes: bytes, paper_id: str) -> list:
    """Text-only chunking. Runs in a worker thread (CPU-bound, C library).

    Extraction detail matters more than it looks. `get_text("text")` returns the
    PDF's VISUAL layout, i.e. one line per rendered line, and
    `_split_into_chunks` treats each line as a semantic unit. Feeding raw visual
    lines through therefore produced badly degraded chunks (measured on the first
    run of this tool):

    * 69% of chunks landed in the default `method` section, because a heading
      glued to body text ("3.1 XNER We present...") never satisfies the
      header rule (whole line, <80 chars) so it matched the `method` keyword
      list by substring and every following chunk inherited it;
    * 53% of chunks were under 50 words and 13k under 15 words, because visual
      line breaks (`4.4.1 XNER`, wrapped fragments) became standalone chunks;
    * hyphenated words stayed split ("formances are far below", "world knowl-
      edge"), corrupting the text a reviewer has to read.

    So visual lines are re-joined into paragraphs before chunking. That is the
    fix: chunking quality is bounded by what the extractor hands it, and no
    amount of post-filtering recovers a paragraph that arrived pre-shattered.
    """
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [_page_to_paragraphs(page) for page in doc]
    finally:
        # C resource; without an explicit close it survives until GC.
        doc.close()

    stream = _join_pages(pages)
    if len(stream.strip()) < 500:
        return []

    from app.services.rag_service import _split_into_chunks

    chunks = _split_into_chunks(stream, paper_id)
    return _drop_reference_chunks(chunks)


# A visual line that ends WITHOUT terminal punctuation is almost certainly
# wrapped mid-sentence; the next line continues it.
_SENTENCE_END = re.compile(r'[.!?:;]\s*$')
# A numbered/lettered heading such as "3.1 XNER", "2 Related Work", "IV. RESULTS".
_HEADING = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+([A-Z][^.]{0,70})$')
_PAGE_NUMBER = re.compile(r'^\s*\d{1,3}\s*$')
_ARXIV_HEADER = re.compile(r'^\s*arXiv:\d{4}\.\d{4,5}', re.I)
# Journal boilerplate printed on every page of a download.
_BOILERPLATE = re.compile(
    r'(all rights reserved|downloaded from|see https?://|'
    r'this article is protected|published by|licensed under|'
    r'^\s*\d+\s*/\s*\d+\s*$)', re.I)


def _is_boilerplate(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line) or _ARXIV_HEADER.match(line)
                or _BOILERPLATE.search(line))


def _heal_hyphenation(prev: str, nxt: str) -> tuple[str, str] | None:
    """Join `prev` and `nxt` when a word was split across a line break.

    Returns the replacement pair, or None when the two lines are unrelated.
    Only a trailing letter-dash joins (never an em/en dash, which is real
    punctuation), and both halves must be alphanumeric — otherwise a genuine
    list dash at a line end would be swallowed.
    """
    m = re.search(r'([A-Za-z]{2,})-$', prev.rstrip())
    if not m or not nxt:
        return None
    head = nxt[0]
    if not head.isalnum():
        return None
    return prev.rstrip()[:-1], nxt


def _split_leading_heading(paragraph: str, max_heading_words: int = 8) -> list[str]:
    """Split a heading glued to the start of body text into two lines.

    PDFs routinely render a section title and the first body sentence on one
    visual line ("2.1 Concept Memory We introduce a concept memory module.").
    Joining visual lines into paragraphs preserves that, and the downstream
    header rule — which requires the WHOLE line to be a short, header-looking
    string — then never fires. Measured effect before this split: sections
    collapsed to `unknown` for 29 of 31 chunks in one paper, which would stop
    the snippet picker from preferring body prose.

    Only a numbered heading at position 0 is split, and only when a body-like
    clause follows it, so a legitimate sentence is never truncated.
    """
    m = re.match(r'^\s*((?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?)\s+(.+)$', paragraph)
    if not m:
        return [paragraph]
    number, rest = m.group(1), m.group(2).strip()
    words = rest.split()

    # Capitalisation cannot locate the boundary: a Title Case heading
    # ("Concept Memory") looks exactly like a sentence start, so any rule keyed
    # on "capital followed by lowercase" cuts after "Concept". Length is the
    # signal instead. Try LONGEST heading first: a short-prefix-first search
    # returns "Concept" and leaves "Memory We introduce..." as the body, which
    # passes every prose check while silently truncating the title.
    limit = min(max_heading_words, max(len(words) - 3, 0))
    for n in range(limit, 0, -1):
        title = " ".join(words[:n])
        body = " ".join(words[n:])
        if re.search(r'[.;:,]$', title):
            continue
        if not re.match(r'[A-Z(\[]', body):
            continue
        if not re.search(r'[a-z]{3,}', body):
            continue
        return [f"{number} {title}", body]
    return [paragraph]


def _page_to_paragraphs(page) -> list[str]:
    """Reconstruct paragraphs from one page's visual lines."""
    raw = page.get_text("text")
    lines = [ln.rstrip() for ln in raw.split("\n")]

    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank visual line = paragraph separator in most PDFs.
            merged.append("")
            continue
        if _is_boilerplate(stripped):
            continue
        if merged and merged[-1]:
            healed = _heal_hyphenation(merged[-1], stripped)
            if healed is not None:
                merged[-1], stripped = healed[0] + healed[1], ""
                if not stripped:
                    continue
            # Continuation only if the previous line looks mid-sentence and the
            # current line is not itself a heading.
            if (not _SENTENCE_END.search(merged[-1])
                    and not _HEADING.match(stripped)
                    and not merged[-1].endswith("-")):
                merged[-1] = merged[-1] + " " + stripped
                continue
        merged.append(stripped)

    # Emit heading and body as separate lines, then collapse blank runs so the
    # downstream chunker's whole-line header rule can still match a heading.
    out: list[str] = []
    for para in merged:
        if not para:
            if out and out[-1] != "":
                out.append("")
            continue
        for piece in _split_leading_heading(para):
            out.append(piece)
    return out


def _join_pages(pages: list[list[str]]) -> str:
    out: list[str] = []
    for paras in pages:
        out.extend(paras)
        if out and out[-1] != "":
            out.append("")
    return "\n".join(out)


def _drop_reference_chunks(chunks: list) -> list:
    """Remove chunks that are bibliography, not prose.

    A reference list is a dense run of citation-shaped lines; it is where the
    first version of this tool handed reviewers its worst evidence. One measured
    snippet was literally `arXiv preprint arXiv:2308.08155. 12 Zhihong Xu, ...`
    — a bibliography entry that shared the claim's keywords and would have been
    read as prior art.
    """
    _CITATION = re.compile(
        r'(arxiv preprint|doi:\s*10\.|\bet al\.,?\s*\d{4}|'
        r'\bpp\.\s*\d+\s*[-–]\s*\d+|\bvol\.\s*\d+|\bno\.\s*\d+|'
        r'\d{4}\.\s*[A-Z][a-z]+,)', re.I)
    kept = []
    for c in chunks:
        text = c.text or ""
        # A heading that announces the bibliography poisons the whole chunk.
        if re.search(r'^\s*(references|bibliography)\s*$', text, re.I | re.M):
            continue
        if len(text.split()) < 25:
            # Too short to be prose; almost always a stray heading, a table row,
            # or a wrapped fragment. The first run emitted 13k chunks under 15
            # words, including bare section numbers like "4.4.1 XNER".
            continue
        hits = len(_CITATION.findall(text))
        # Density matters: a methods paragraph may legitimately cite one paper,
        # a bibliography chunk cites continuously.
        if hits >= 3 and hits / max(len(text.split()), 1) > 0.01:
            continue
        kept.append(c)
    return kept


def _select_reparse_targets(db, limit: int | None,
                            degraded_only: bool = False) -> list:
    """Papers we previously parsed with the degraded extractor.

    Identified by having `pymupdf_inline` chunks AND a PDF URL. Re-parsing is
    needed because the extraction fix (paragraph re-joining, heading splitting,
    boilerplate stripping) changes the output text, so existing chunks stay
    degraded until rebuilt.

    `degraded_only` narrows this to papers whose chunks still look like the old
    output. That makes an interrupted run resumable: the extractor's fixes raise
    average chunk length from ~65 to ~200 words, so mean length cleanly separates
    re-parsed papers from not-yet-done ones without a migration marker.
    """
    from sqlalchemy import text

    if degraded_only:
        sql = """
            SELECT p.id, p.title, p.pdf_url
            FROM papers p
            JOIN (
                SELECT paper_id, AVG(word_count) avg_w, COUNT(*) n
                FROM paper_chunks GROUP BY paper_id
            ) c ON c.paper_id = p.id
            WHERE p.pdf_url IS NOT NULL AND p.pdf_url != ''
              AND c.avg_w < :threshold
            ORDER BY p.citation_count DESC NULLS LAST
        """
        rows = db.execute(text(sql), {"threshold": DEGRADED_AVG_WORDS}).fetchall()
    else:
        sql = """
            SELECT DISTINCT p.id, p.title, p.pdf_url
            FROM papers p
            JOIN paper_chunks pc ON pc.paper_id = p.id
            WHERE pc.extraction_method = 'pymupdf_inline'
              AND p.pdf_url IS NOT NULL AND p.pdf_url != ''
            ORDER BY p.citation_count DESC NULLS LAST
        """
        rows = db.execute(text(sql)).fetchall()
    return rows[:limit] if limit else rows


def _delete_chunks(db, paper_id: str) -> int:
    """Remove existing chunks for a paper, returning how many were deleted."""
    from sqlalchemy import text

    return db.execute(text("DELETE FROM paper_chunks WHERE paper_id = :p"),
                      {"p": paper_id}).rowcount


async def _run(args) -> None:
    from app.db.session import SessionLocal

    audit_dir = Path(args.audit_dir)
    ids, titles = _collect_audit_candidates(audit_dir)
    print(f"[backfill] audit candidates: {len(ids)} with a local id, "
          f"{len(titles)} distinct titles")

    db = SessionLocal()
    try:
        if args.reparse:
            targets = _select_reparse_targets(db, args.limit,
                                              degraded_only=args.degraded_only)
        else:
            targets = _select_targets(db, ids, titles, args.limit)
    finally:
        db.close()

    what = "to re-parse (existing degraded chunks)" if args.reparse else \
        "need full text (have pdf_url, no chunks)"
    print(f"[backfill] {len(targets)} paper(s) {what}")
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
        # Order matters and is the whole safety property of --reparse: build the
        # new chunks FIRST, and only replace the old ones once parsing succeeded.
        # Deleting up front would turn any download/parse failure into data loss
        # for a paper that previously had usable text.
        if args.reparse:
            success, chunks, note = await _build_chunks(paper)
            if success:
                d = SessionLocal()
                try:
                    _delete_chunks(d, paper.id)
                    d.commit()
                finally:
                    d.close()
                success, n_chunks, note = await _persist_chunks(paper, chunks)
            else:
                n_chunks = 0
        else:
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
    p.add_argument("--reparse", action="store_true",
                   help="re-extract papers that already have chunks, replacing them "
                        "with output from the current extractor. Old chunks are only "
                        "deleted after the replacement parses successfully.")
    p.add_argument("--degraded-only", action="store_true",
                   help="with --reparse, target only papers whose chunks still look "
                        "like pre-fix output, so an interrupted run can resume.")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

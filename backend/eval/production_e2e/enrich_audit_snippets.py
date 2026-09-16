"""Attach quotable full-text snippets to an existing audit run's candidates.

Why this exists
---------------
The blind review sheet must show a passage from each candidate so a reviewer can
adjudicate FULL / PARTIAL / NONE. Candidates were recorded as a snapshot BEFORE
the PDF backfill, so the 2026-09-16 audit run has snippets on only 8 of 91
targets even though `paper_chunks` now holds 61,879 chunks.

Re-running the audit would rebuild them, but a full pass is ~5.5 h (21 targets
took 76 min) and repeating it burns LLM budget for a purely local lookup.

Enrichment is enough. A snippet is a property of (paper, claim) — not of the
search that found the paper — so it can be computed afterwards from the chunk
table without touching query generation or the public APIs. The candidates
themselves are unchanged, which also keeps results comparable with the run that
was already produced.

Scoring: pick the chunk that shares the most distinctive claim terms, so the
displayed passage is the one most likely to bear on the claim. A single
shared common word is not enough — see the MIN_TERM_HITS floor, which exists for
the same reason the retriever's threshold does.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "is", "are", "be", "as", "at", "from", "that", "this", "its", "it", "we",
    "can", "does", "how", "what", "when", "which", "under", "into", "via",
    "using", "used", "use", "based", "method", "methods", "approach",
    "approaches", "system", "systems", "model", "models", "paper", "study",
    "propose", "proposes", "proposed", "novel", "new", "results", "result",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


def _terms(text: str, max_terms: int = 12) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in _STOPWORDS or tok.isdigit() or tok in out:
            continue
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


# A bibliography line, a citation entry, or a table row. Showing one of these
# as "the passage that matches your claim" is actively misleading: a reviewer
# reads it as evidence about the claim when it is only a coincidental keyword
# overlap. Measured before this filter: 11% of snippets were bibliography or
# table content, one of them literally
# `arXiv preprint arXiv:2308.08155. 12 Zhihong Xu, ...`.
_CITATION = re.compile(
    r'(arxiv preprint|doi:\s*10\.|\bet al\.,?\s*\d{4}|'
    r'\bpp\.\s*\d+\s*[-–]\s*\d+|\bvol\.\s*\d+|\bno\.\s*\d+)', re.I)
_TABLE_ROW = re.compile(r'(\d+[.:%]\s*){4,}')
# Sections that never contain the prose a reviewer needs.
_NON_BODY_SECTIONS = {"references", "bibliography", "acknowledgments",
                      "acknowledgements", "appendix"}


def _looks_like_body(text: str) -> bool:
    """Reject snippets that are not continuous prose."""
    stripped = text.strip()
    if len(stripped.split()) < 15:
        return False
    if _TABLE_ROW.search(stripped):
        return False
    if _CITATION.search(stripped) and len(_CITATION.findall(stripped)) >= 2:
        return False
    # A passage that is mostly numerals/symbols is a table, not a sentence.
    letters = sum(1 for c in stripped if c.isalpha())
    if letters / max(len(stripped), 1) < 0.6:
        return False
    return True


def _pick_snippet(db, paper_id: str, claim: str, max_chars: int = 300) -> dict | None:
    """Best matching chunk for `claim`, or None.

    Scored in SQL by counting distinct claim terms present, so a long chunk
    cannot win merely by being long. Candidate chunks are filtered for prose
    quality in Python, because "matches the most claim terms" and "is worth
    showing a reviewer" are different questions — a bibliography entry dense
    with the claim's topic words scores highly on the first and fails the second.
    """
    from sqlalchemy import text

    terms = _terms(claim)
    if not terms:
        return None
    parts, params = [], {"pid": paper_id}
    for i, term in enumerate(terms):
        key = f"t{i}"
        params[key] = f"%{term}%"
        parts.append(f"(pc.text LIKE :{key})")
    expr = " + ".join(parts)
    # Same rationale as the retriever's threshold: one incidental shared word is
    # not evidence the passage is about the claim.
    min_hits = 2 if len(terms) >= 3 else 1

    # Fetch a few, not just the top one: the best-scoring chunk is often a
    # bibliography, and the next-best may be clean prose.
    rows = db.execute(text(f"""
        SELECT pc.section, substr(pc.text, 1, :n) AS snippet, ({expr}) AS n_hits
        FROM paper_chunks pc
        WHERE pc.paper_id = :pid AND ({expr}) >= :min_hits
        ORDER BY n_hits DESC
        LIMIT 6
    """), {**params, "n": max_chars, "min_hits": min_hits}).fetchall()

    for row in rows:
        section = (row.section or "").strip().lower()
        if section in _NON_BODY_SECTIONS:
            continue
        snippet = " ".join(row.snippet.split())
        if not _looks_like_body(snippet):
            continue
        return {"section": row.section, "snippet": snippet}
    return None


def _run(args) -> None:
    from app.db.session import SessionLocal

    audit_dir = Path(args.audit_dir)
    records_path = audit_dir / "candidate_killer_papers.jsonl"
    if not records_path.exists():
        raise SystemExit(f"[enrich] no records at {records_path}")
    records = [json.loads(l) for l in
               records_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    db = SessionLocal()
    enriched = skipped = no_text = unresolved = 0
    try:
        # Resolve titles -> ids once; the join key is the normalised title for
        # candidates recorded before local_paper_id was captured.
        titles = {
            " ".join((c.get("title") or "").lower().split())
            for r in records for c in (r.get("candidate_papers") or [])
            if c.get("source") == "local_corpus" and c.get("title")
        }
        title_to_id = {}
        if titles:
            from sqlalchemy import text
            params = {f"t{i}": t for i, t in enumerate(titles)}
            ph = ", ".join(f":t{i}" for i in range(len(titles)))
            for pid, title in db.execute(text(
                    f"SELECT id, LOWER(TRIM(title)) FROM papers WHERE LOWER(TRIM(title)) IN ({ph})"),
                    params).fetchall():
                title_to_id.setdefault(title, pid)

        for rec in records:
            claim = rec.get("claim") or ""
            for cand in (rec.get("candidate_papers") or []):
                if cand.get("snippet"):
                    skipped += 1
                    continue
                pid = cand.get("local_paper_id")
                if not pid:
                    title = " ".join((cand.get("title") or "").lower().split())
                    pid = title_to_id.get(title)
                if not pid:
                    unresolved += 1
                    continue
                hit = _pick_snippet(db, pid, claim)
                if not hit:
                    no_text += 1
                    continue
                cand["snippet"] = hit["snippet"]
                cand["section"] = hit["section"]
                cand["match_type"] = "full_text"
                cand["local_paper_id"] = pid
                enriched += 1
    finally:
        db.close()

    print(f"[enrich] enriched={enriched} already_had={skipped} "
          f"no_matching_text={no_text} title_unresolved={unresolved}")
    if args.dry_run:
        print("[enrich] --dry-run: not writing")
        return

    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[enrich] rewrote {records_path}")

    # Rebuild the blind sheet so the reviewer sees the new snippets. It is a
    # derived view of the records, so regenerating is safe.
    if args.rebuild_sheet:
        from eval.production_e2e.super_audit import blind_review_section
        md = ["# Super Audit — BLIND human review sheet", "",
              "Review each submission WITHOUT knowing which system produced it.",
              "Fill every field; identities are restored automatically afterwards.",
              "", ""]
        for rec in records:
            md.append(blind_review_section(
                rec["submission_id"], rec.get("_topic") or rec.get("topic_id") or "",
                rec.get("claim") or "", rec.get("candidate_papers") or [],
                rec.get("query_failure")))
        (audit_dir / "human_review_blind.md").write_text("\n".join(md), encoding="utf-8")
        print("[enrich] rebuilt human_review_blind.md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach full-text snippets to audit candidates")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebuild-sheet", action="store_true",
                   help="also regenerate human_review_blind.md from the records")
    return p


def main() -> None:
    _run(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Local corpus paper source.

Searches papers ALREADY in our own database, not the public web. This is the
source the super audit was missing: the audit previously queried only OpenAlex /
Semantic Scholar / arXiv, so a killer paper that production had already
downloaded and indexed was invisible to the audit.

Why it matters for adversarial auditing
---------------------------------------
The strongest place to find prior art for a claim is the corpus the research
pipeline itself gathered, because:

* full text is available: 5k+ PaperChunk rows back ~560 PDFs, so a claim can be
  checked against actual method/result text, not just a title match;
* already-extracted evidence (28k EvidenceUnit rows) can be surfaced verbatim as
  the span that covers (or fails to cover) the claim;
* it is a genuinely different retrieval path from the public APIs, which is what
  the audit's "multi-source" requirement is for. During the 2026-09-15 run S2 and
  arXiv were fully rate-limited, leaving OpenAlex as the only live source -- the
  local corpus keeps the audit multi-source even when public APIs are down.

Design notes
------------
* No FTS virtual table exists (verified: no `%fts%` objects in sqlite_master), so
  matching is done with parameterised LIKE over normalised terms. This is
  intentionally conservative: we prefer precision (fewer, on-topic hits) because
  the audit caps candidates at 12 and a flood of fuzzy matches would crowd out
  the public sources.
* The session is created per search call and closed in `finally`; the audit runs
  this concurrently from asyncio, and SQLite sessions are not thread-safe.
* Any DB failure degrades to an empty result rather than raising, matching the
  source-level isolation contract in `search_candidates`.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import text

from app.paper_sources.base import PaperSource, RawPaper

logger = logging.getLogger(__name__)

# Words that carry no retrieval signal and would match almost every paper.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "is", "are", "be", "as", "at", "from", "that", "this", "its", "it", "we",
    "can", "do", "does", "how", "what", "when", "which", "under", "into", "via",
    "using", "used", "use", "based", "method", "methods", "approach", "approaches",
    "system", "systems", "model", "models", "paper", "study", "novel", "new",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


def _terms(query: str, max_terms: int = 8) -> list[str]:
    """Extract distinctive lowercase terms from a query string."""
    tokens = _TOKEN_RE.findall((query or "").lower())
    out: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or tok.isdigit():
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= max_terms:
            break
    return out


class LocalCorpusSource(PaperSource):
    """Search papers we already hold (metadata + full-text chunks + evidence)."""

    name = "local_corpus"
    # No network: a single query is a few indexed-ish LIKE scans over local tables.
    base_url = ""

    async def _do_search(self, query: str, limit: int, headers: dict) -> list[RawPaper]:
        return self._search_sync(query, limit)

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        try:
            return await self._do_search(query, limit, {})
        except Exception as exc:  # never let a local DB hiccup break the audit
            logger.warning("LocalCorpus search failed for %r: %s", query[:60], exc)
            return []

    # ------------------------------------------------------------------ helpers

    def _search_sync(self, query: str, limit: int) -> list[RawPaper]:
        terms = _terms(query)
        if not terms:
            return []

        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            hits: dict[str, RawPaper] = {}
            # Tier order matters and was arrived at empirically (2026-09-15).
            #
            # Scoring metadata/title FIRST is deliberate, though it feels
            # backwards for a full-text corpus. Matching freely against 5k
            # full-text chunks admits any paper that happens to use three common
            # words anywhere in 40 pages, and because long survey papers have the
            # most chunks they dominated every query regardless of topic (three
            # different queries all returned the same surveys). Title/abstract
            # are short and topical, so requiring a strong match there first
            # gives precision; the chunk/evidence tiers then only supply the
            # quotable snippet for papers already selected as on-topic.
            self._collect_paper_hits(db, terms, hits, limit)
            if len(hits) < limit:
                self._collect_evidence_hits(db, terms, hits, limit)
            if len(hits) < limit:
                self._collect_chunk_hits(db, terms, hits, limit)
            # Order by evidential strength: a full-text match beats an extracted
            # evidence span, which beats a metadata-only match; within a tier the
            # paper matching more distinct query terms comes first.
            tier = {"full_text": 2, "evidence": 1, "metadata": 0}
            ranked = sorted(
                hits.values(),
                key=lambda p: (tier.get((p.raw_data or {}).get("match_type"), 0),
                               (p.raw_data or {}).get("n_hits", 0),
                               p.citation_count or 0),
                reverse=True,
            )
            return ranked[:limit]
        finally:
            db.close()

    @staticmethod
    def _match_expr(column: str, terms: list[str], prefix: str) -> tuple[str, dict]:
        """Build a hit COUNTER and its bind params.

        Returns `((col LIKE :p0) + (col LIKE :p1) + ... AS hits, params)`. A
        plain OR plus citation ordering was the first attempt and it was unusable:
        one incidental word was enough to admit a paper, so every query returned
        the same handful of mega-cited papers (Mizar, LLaMA, XAI) regardless of
        topic. Counting per-term hits lets us keep only papers that match enough
        distinct terms to actually be about the query.
        """
        parts = []
        params = {}
        for i, term in enumerate(terms):
            key = f"{prefix}{i}"
            params[key] = f"%{term}%"
            parts.append(f"({column} LIKE :{key})")
        return " + ".join(parts), params

    def _min_hits(self, terms: list[str]) -> int:
        """How many distinct terms a document must match to be admitted."""
        n = len(terms)
        if n <= 2:
            return n
        if n <= 4:
            return 2
        return 3

    def _collect_chunk_hits(self, db, terms, hits: dict, limit: int) -> None:
        """Attach a quotable full-text snippet, but only to already-selected papers.

        This tier must NOT admit new papers: free matching over 5k multi-page
        chunks is far too permissive (see the tier comment in `_search_sync`).
        Its job is to upgrade a metadata/evidence hit into a full-text hit so the
        reviewer sees the sentence that actually bears on the claim.
        """
        if not hits:
            return
        expr, params = self._match_expr("pc.text", terms, "c")
        params["min_hits"] = self._min_hits(terms)
        params["lim"] = max(limit * 6, 30)
        rows = db.execute(text(
            f"""
            SELECT p.id, p.title, p.year, p.venue, p.abstract, p.citation_count,
                   p.doi, p.url, pc.section,
                   substr(pc.text, 1, 400) AS snippet,
                   ({expr}) AS n_hits
            FROM paper_chunks pc
            JOIN papers p ON p.id = pc.paper_id
            WHERE ({expr}) >= :min_hits
            ORDER BY n_hits DESC, p.citation_count DESC
            LIMIT :lim
            """
        ), params).fetchall()
        for r in rows:
            if r.id not in hits:
                continue  # never admit a new paper from this tier
            self._add_hit(hits, r, match_type="full_text", section=r.section,
                          snippet=r.snippet, limit=limit, n_hits=r.n_hits)

    def _collect_evidence_hits(self, db, terms, hits: dict, limit: int) -> None:
        # evidence_units has no `evidence_text` column: the quotable span lives in
        # `original_span`, and the normalised restatement in `normalized_claim`.
        # Match either, so a differently-worded extraction still surfaces.
        expr, params = self._match_expr(
            "coalesce(eu.original_span, '') || ' ' || coalesce(eu.normalized_claim, '')",
            terms, "e")
        params["min_hits"] = self._min_hits(terms)
        params["lim"] = max(limit * 3, 15)
        rows = db.execute(text(
            f"""
            SELECT p.id, p.title, p.year, p.venue, p.abstract, p.citation_count,
                   p.doi, p.url, eu.section,
                   substr(coalesce(eu.original_span, eu.normalized_claim), 1, 400) AS snippet,
                   ({expr}) AS n_hits
            FROM evidence_units eu
            JOIN papers p ON p.id = eu.paper_id
            WHERE ({expr}) >= :min_hits
            ORDER BY n_hits DESC, p.citation_count DESC
            LIMIT :lim
            """
        ), params).fetchall()
        for r in rows:
            self._add_hit(hits, r, match_type="evidence", section=r.section,
                          snippet=r.snippet, limit=limit, n_hits=r.n_hits)

    def _collect_paper_hits(self, db, terms, hits: dict, limit: int) -> None:
        # Title matches are worth more than abstract matches, so the title
        # counter counts double. Metadata stays the weakest tier overall.
        expr_t, params_t = self._match_expr("title", terms, "t")
        expr_a, params_a = self._match_expr("abstract", terms, "a")
        params = dict(params_t)
        params.update(params_a)
        # Strict threshold: a title term counts double, so >=4 means roughly
        # "two topical title terms" or "four distinct abstract terms". Lowering
        # this floods results with generically-worded papers.
        params["min_hits"] = 4
        params["lim"] = max(limit * 3, 15)
        rows = db.execute(text(
            f"""
            SELECT id, title, year, venue, abstract, citation_count, doi, url,
                   NULL AS section, NULL AS snippet,
                   (({expr_t}) * 2 + ({expr_a})) AS n_hits
            FROM papers
            WHERE (({expr_t}) * 2 + ({expr_a})) >= :min_hits
            ORDER BY n_hits DESC, citation_count DESC
            LIMIT :lim
            """
        ), params).fetchall()
        for r in rows:
            self._add_hit(hits, r, match_type="metadata", section=None,
                          snippet=None, limit=limit, n_hits=r.n_hits)

    @staticmethod
    def _add_hit(hits: dict, row, match_type: str, section, snippet, limit: int,
                 n_hits: int = 0) -> None:
        # The cap is enforced by the caller between tiers; here we dedupe and
        # upgrade, so a stronger match for an already-seen paper can replace a
        # weaker one. match_type lives in raw_data (RawPaper has no such field).
        pid = row.id
        existing = hits.get(pid)
        tier = {"metadata": 0, "evidence": 1, "full_text": 2}
        if existing is not None:
            prev = (existing.raw_data or {}).get("match_type")
            prev_hits = (existing.raw_data or {}).get("n_hits", 0)
            # Keep the better tier; within a tier keep the higher term-match
            # count, so the snippet we show a reviewer is the most on-topic one.
            if (tier.get(match_type, 0), n_hits) <= (tier.get(prev, 0), prev_hits):
                return
        elif len(hits) >= limit:
            return
        hits[pid] = RawPaper(
            title=(row.title or "").strip(),
            abstract=row.abstract or "",
            authors=[],
            year=row.year,
            venue=row.venue or "",
            doi=row.doi or "",
            url=row.url or "",
            citation_count=row.citation_count or 0,
            source="local_corpus",
            # Already in our corpus, so the full text is on hand by construction.
            is_oa=True,
            raw_data={
                "local_paper_id": pid,
                "match_type": match_type,
                "section": section,
                "snippet": snippet,
                "n_hits": n_hits,
            },
        )

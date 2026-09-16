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
            # Which tier may ADMIT a paper is the load-bearing decision here, and
            # it was arrived at empirically (2026-09-15). Metadata/title is the
            # admitting tier: matching freely against 5k full-text chunks admits
            # any paper that happens to use a few common words anywhere in 40
            # pages, and long surveys own the most chunks so they dominated every
            # query regardless of topic (three unrelated queries all returned the
            # same surveys).
            #
            # BUT the later tiers must still RUN even once the cap is full — that
            # is the whole point of holding a full-text corpus. Gating them behind
            # `len(hits) < limit` starved them completely: on 2026-09-15's audit
            # all 392 local_corpus candidates were metadata-only, because a topic
            # with hundreds of papers fills the 12 slots in tier 1 and the
            # chunk/evidence tiers never executed. Result: every candidate reached
            # the blind review sheet with `snippet: null`, leaving the reviewer to
            # judge FULL/PARTIAL/NONE from a title alone.
            #
            # So: tier 1 admits, tiers 2-3 upgrade-in-place. `_collect_*` already
            # refuses to add papers absent from `hits`, so running them
            # unconditionally cannot loosen precision.
            self._collect_paper_hits(db, terms, hits, limit)
            self._collect_evidence_hits(db, terms, hits, limit)
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
        # Restrict to the papers already selected. Fetching the globally
        # best-matching chunks and filtering afterwards does not work: the top-N
        # chunks come from arbitrary papers (long surveys produce the most), so
        # the chunks belonging to our candidates get cut off by LIMIT before the
        # admit=False guard ever sees them. Observed as a seeded paper that
        # matched terms in its own chunk yet never upgraded past `metadata`.
        params["min_hits"] = self._min_hits(terms)
        params["lim"] = max(limit * 6, 30)
        id_keys = []
        for i, pid in enumerate(hits):
            key = f"pid{i}"
            params[key] = pid
            id_keys.append(f":{key}")
        rows = db.execute(text(
            f"""
            SELECT p.id, p.title, p.year, p.venue, p.abstract, p.citation_count,
                   p.doi, p.url, pc.section,
                   substr(pc.text, 1, 400) AS snippet,
                   ({expr}) AS n_hits
            FROM paper_chunks pc
            JOIN papers p ON p.id = pc.paper_id
            WHERE p.id IN ({', '.join(id_keys)})
              AND ({expr}) >= :min_hits
            ORDER BY n_hits DESC, p.citation_count DESC
            LIMIT :lim
            """
        ), params).fetchall()
        for r in rows:
            self._add_hit(hits, r, match_type="full_text", section=r.section,
                          snippet=r.snippet, limit=limit, n_hits=r.n_hits,
                          admit=False)

    def _collect_evidence_hits(self, db, terms, hits: dict, limit: int) -> None:
        # evidence_units has no `evidence_text` column: the quotable span lives in
        # `original_span`, and the normalised restatement in `normalized_claim`.
        # Match either, so a differently-worded extraction still surfaces.
        expr, params = self._match_expr(
            "coalesce(eu.original_span, '') || ' ' || coalesce(eu.normalized_claim, '')",
            terms, "e")
        params["min_hits"] = self._min_hits(terms)
        params["lim"] = max(limit * 3, 15)
        # Restricted to already-selected papers, same reasoning as the chunk
        # tier: filtering after a global ORDER BY ... LIMIT silently drops the
        # rows belonging to our candidates.
        id_keys = []
        for i, pid in enumerate(hits):
            key = f"eid{i}"
            params[key] = pid
            id_keys.append(f":{key}")
        rows = db.execute(text(
            f"""
            SELECT p.id, p.title, p.year, p.venue, p.abstract, p.citation_count,
                   p.doi, p.url, eu.section,
                   substr(coalesce(eu.original_span, eu.normalized_claim), 1, 400) AS snippet,
                   ({expr}) AS n_hits
            FROM evidence_units eu
            JOIN papers p ON p.id = eu.paper_id
            WHERE p.id IN ({', '.join(id_keys)})
              AND ({expr}) >= :min_hits
            ORDER BY n_hits DESC, p.citation_count DESC
            LIMIT :lim
            """
        ), params).fetchall()
        for r in rows:
            self._add_hit(hits, r, match_type="evidence", section=r.section,
                          snippet=r.snippet, limit=limit, n_hits=r.n_hits,
                          admit=False)

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

    _TIER = {"metadata": 0, "evidence": 1, "full_text": 2}

    @classmethod
    def _add_hit(cls, hits: dict, row, match_type: str, section, snippet, limit: int,
                 n_hits: int = 0, admit: bool = True) -> None:
        """Insert a paper, or upgrade one already present to a stronger match.

        `admit=False` restricts this to papers already in `hits`, which is how the
        evidence/full-text tiers are kept from loosening precision (see
        `_search_sync`).

        `n_hits` is NOT comparable across tiers -- the metadata tier scores a
        weighted (title*2 + abstract) count while the chunk/evidence tiers count
        raw term hits -- so it is stored per tier and never used to re-rank a
        paper against one from another tier. Comparing them was a latent bug:
        upgrading a metadata hit (weighted score, e.g. 8) to a full-text hit
        (raw count, e.g. 3) would have DEMOTED the paper in the final sort.
        """
        pid = row.id
        existing = hits.get(pid)
        if existing is not None:
            prev_tier = (existing.raw_data or {}).get("match_type")
            prev_hits = (existing.raw_data or {}).get("n_hits", 0)
            # Better tier wins outright. Within the same tier keep the higher
            # term-match count, so the snippet shown to a reviewer is the most
            # on-topic one available.
            if (cls._TIER.get(match_type, 0), n_hits) <= (cls._TIER.get(prev_tier, 0), prev_hits):
                return
        elif not admit or len(hits) >= limit:
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

"""Citation-snowball paper source.

Finds work that CITES a paper we already hold, restricted to a topic. This is the
recall path the audit was missing: keyword search over the whole of OpenAlex
returns whatever shares a few words with the claim, and the 2026-09-16 run showed
39% of candidates shared no content word at all with the claim they were meant to
test (a MoE visual-salience claim drew "Obesity and Cancer").

Why the combination, and why not bare snowball
----------------------------------------------
A bare "who cites this?" query is WORSE than keyword search, not better. Measured:
the R language paper has 353,375 citers and the top hits are lme4, limma and
phyloseq — generic high-citation infrastructure, none of it about a specific
claim. Ranking is dominated by popularity, so the noise is systematic rather than
random.

The useful query restricts the citation set to a topic:

    filter=cites:W<id>,title.search:<topic terms>

Measured on a real quantization paper (79 citers):
* bare `cites:`                        -> 79 results, top hits are broad surveys
* `cites:` + `search=`                 -> 62 results, LLM-FP4 / Outlier Suppression+
* `cites:` + `title.search=`           -> 34 results, ALL quantization-specific
* `cites:` sorted by citation count    -> surveys dominate again

So `title.search` is the variant used here. It answers a much sharper question
than keyword search can — "what later work cites this and is *about* this
topic" — because the candidate set is already constrained by a real citation
edge rather than by lexical coincidence.

Scope note: OpenAlex only exposes forward citations (who cites X) via `filter`.
Backward (`referenced_works`) requires fetching and parsing each work, which is a
much larger change; forward citation is also the direction that matters for
killing a claim, since a killer paper must postdate the work being challenged.
"""

from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.paper_sources.base import PaperSource, RawPaper

logger = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


class CitationSnowballSource(PaperSource):
    """Papers that cite a given work, filtered to a topic."""

    name = "citation_snowball"
    base_url = OPENALEX_WORKS

    def __init__(self, seed_ids: list[str] | None = None, topic: str = ""):
        # Seed works are the production papers for the claim under audit; the
        # topic is what the claim is about.
        self.seed_ids = [s for s in (seed_ids or []) if s]
        self.topic = topic

    async def _do_search(self, query: str, limit: int, headers: dict) -> list[RawPaper]:
        if not self.seed_ids:
            return []
        topic = query or self.topic
        topics = _topic_terms(topic)
        if not topics:
            return []

        papers: list[RawPaper] = []
        seen: set[str] = set()

        def absorb(found: list[RawPaper]) -> bool:
            for paper in found:
                key = paper.doi or (paper.title or "").lower()
                if key in seen:
                    continue
                seen.add(key)
                papers.append(paper)
            return len(papers) >= limit

        # Recall strategy, most PRECISE first. Both extremes were measured and
        # rejected, so the order below is deliberate:
        #
        # * single narrow term (`entropy-adaptive`) -> 0 hits: the seeds are
        #   broad surveys whose citers never put that exact compound in a title.
        # * single generic term (`gradients`) -> 12 hits but junk: it returned
        #   MARL policy-gradient papers for a claim about orthogonal LoRA
        #   gradient projection. A term that appears throughout the seed's whole
        #   field cannot discriminate within its citation set.
        #
        # So the unit of query is a combination of 2+ claim terms rather than one
        # word. `title.search` treats space-separated words as AND, which is
        # exactly the conjunction needed: the citing paper must be about BOTH.
        # Only if no combination lands do we widen to single terms.
        def combos() -> list[str]:
            if len(topics) >= 2:
                # Adjacent pairs first: extracted terms are roughly ordered by
                # specificity, so neighbours are the tightest natural couples.
                pairs = [f"{topics[i]} {topics[i + 1]}"
                         for i in range(len(topics) - 1)]
                if len(topics) >= 4:
                    pairs.append(f"{topics[0]} {topics[2]}")
                return pairs
            return list(topics)

        for combo in combos()[:4]:
            for seed in self.seed_ids[:4]:
                if absorb(await self._query_citing(seed, combo, limit, headers,
                                                   mode="title")):
                    return papers[:limit]
        if papers:
            return papers[:limit]

        # Widen: individual terms, most specific first.
        for term in topics[:4]:
            for seed in self.seed_ids[:4]:
                if absorb(await self._query_citing(seed, term, limit, headers,
                                                   mode="title")):
                    return papers[:limit]
        if papers:
            return papers[:limit]

        # Last resort: any citing paper mentioning the claim.
        for seed in self.seed_ids[:4]:
            if absorb(await self._query_citing(seed, "", limit, headers,
                                               mode="search")):
                return papers[:limit]
        return papers[:limit]

    async def _query_citing(self, seed: str, topic_term: str, limit: int,
                            headers: dict, mode: str = "title") -> list[RawPaper]:
        if mode == "title" and not topic_term:
            return []
        filters = f"cites:{seed}"
        params: dict = {"sort": "cited_by_count:desc"}
        if mode == "title":
            # A citing paper whose TITLE contains the topic word: the closest
            # available proxy for "later work about this topic".
            params["filter"] = f"{filters},title.search:{topic_term}"
        else:
            # A citing paper whose full text/abstract mentions the claim.
            params["filter"] = filters
            params["search"] = self.topic
        params["per_page"] = min(max(limit, 5), 50)
        if settings.openalex_email:
            params["mailto"] = settings.openalex_email
        if getattr(settings, "openalex_api_key", None):
            params["api_key"] = settings.openalex_api_key
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(OPENALEX_WORKS, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # source-level isolation
            logger.warning("citation_snowball %s/%s failed: %s: %s",
                           seed, topic_term or mode, type(exc).__name__, exc)
            return []

        out = []
        for item in data.get("results") or []:
            oa = (item.get("primary_location") or {}).get("source") or {}
            out.append(RawPaper(
                title=(item.get("title") or "").strip(),
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                authors=[],
                year=item.get("publication_year"),
                venue=oa.get("display_name") or "",
                doi=(item.get("doi") or "").replace("https://doi.org/", ""),
                url=item.get("id") or "",
                citation_count=item.get("cited_by_count", 0) or 0,
                source=self.name,
                is_oa=bool((item.get("open_access") or {}).get("is_oa")),
                raw_data={
                    "openalex_id": (item.get("id") or "").replace(
                        "https://openalex.org/", ""),
                    "cites_seed": seed,
                    "topic_term": topic_term,
                },
            ))
        return out

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        mailto = settings.openalex_email
        headers = {"User-Agent": f"DeepResearch/1.0 (mailto:{mailto})" if mailto
                   else "DeepResearch/1.0"}
        try:
            return await self._do_search(query, limit, headers)
        except Exception as exc:
            logger.warning("citation_snowball search failed: %s", exc)
            return []


# Terms too generic to constrain a citation set usefully; `title.search:model`
# would return essentially every citing paper.
_GENERIC = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "is", "are", "be", "as", "at", "from", "that", "this", "its", "it", "we",
    "can", "do", "does", "how", "what", "when", "which", "under", "into", "via",
    "using", "used", "use", "based", "method", "methods", "approach",
    "approaches", "system", "systems", "model", "models", "paper", "study",
    "novel", "new", "results", "result", "learning", "neural", "network",
    "networks", "task", "tasks", "data", "analysis", "evaluation", "evaluate",
    "performance", "training", "train", "large", "small", "deep", "general",
    "framework", "efficient", "efficiency", "improve", "improves", "improved",
    "artificial", "intelligence", "effect", "effects", "impact", "testing",
    "experiment", "experiments", "minimal", "measure", "measuring", "measurement",
}

# Verb/adverb/adjective endings. A topic term must be a content NOUN to be
# discriminative in `title.search`: measured failure where a claim about
# "entropy-adaptive noise schedules for denoising" yielded the term "integrating",
# which matched ecology papers ("Picante: R tools for integrating phylogenies").
# The discriminative words (entropy, denoising, schedule) were present in the
# claim but ranked after the junk because terms were taken in reading order.
_NON_NOUN_SUFFIXES = (
    "ing", "ed", "ly", "ally", "ize", "izes", "ized", "ising", "izing",
    "ate", "ates", "ated", "ise", "ises", "ised", "fy", "fies", "fied",
)


def _looks_like_noun(tok: str) -> bool:
    """Reject verb/adverb forms that make poor `title.search` terms."""
    if len(tok) < 4:
        return False
    for suffix in _NON_NOUN_SUFFIXES:
        if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
            return False
    return True


def _topic_terms(text: str, max_terms: int = 6) -> list[str]:
    """Extract distinctive single NOUNS usable as `title.search` terms.

    Selection is by discriminativeness, not by position in the sentence:
    hyphenated compounds first (entropy-adaptive, post-training are near-unique
    topic markers), then unusual/long words, then the rest. Reading order was
    measured to be wrong — it yields the claim's verbs.
    """
    import re

    toks = []
    for raw in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or ""):
        tok = raw.lower()
        if tok in _GENERIC or tok in toks:
            continue
        if not _looks_like_noun(tok):
            continue
        toks.append(tok)

    def rank(t: str) -> tuple:
        return (
            0 if "-" in t else 1,          # compounds are the most specific
            -len(t),                        # longer words are rarer/sharper
            t,
        )

    toks.sort(key=rank)
    return toks[:max_terms]

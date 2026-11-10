"""Semantic recall: keyword search + embedding filter/rerank.

Why this exists
---------------
The pilot adjudication failed on retrieval, not on judgement. Its clearest case
(fbp-089, MoE router causal intervention) got 12 results that were all about
statistical causal mediation and medical GNNs — zero papers on its own topic.
The queries were fine at the word level; keyword search simply cannot express
a conjunction like "router intervention × hydra-effect compensation". Title-
derived queries made it worse: stripped of their subtitles, real neighbour
titles reduce to generic words ("Context-Adaptive Inference") and match generic
papers.

So the fix is NOT better query strings. It is to over-retrieve by keyword, then
use embeddings to decide what is actually about the same thing. The threshold
was measured, not chosen (scripts/_calibrate_sim.py, 2026-11-02, 37 gaps):

    claim vs its own audit neighbours : median 0.507, p10 0.376
    claim vs the off-topic fbp-089 pool: median 0.332, max 0.408

At 0.41 the filter keeps 82% of real neighbours and admits 0% of the known
noise. The two distributions overlap at the tail, so this is a SOFT filter with
mandatory disclosure (`kept` / `filtered` counts are returned and must be shown
to whatever consumes the results) — never a silent gate. A caller that hides
the filtered count would reproduce the original bug in a new form: "the
reviewer saw nothing" is not "nothing exists".

Degradation contract
--------------------
If embeddings are unavailable, returns the raw keyword results with
`semantic_filtered=False` and `degraded=<reason>`. Callers must surface that
flag: unfiltered noise is still better than a crashed review, but it must not
be mistaken for a successful semantic pass.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.services import embedding_service

logger = logging.getLogger(__name__)

# Over-retrieve, then narrow: keyword recall is cheap, and a filter can only
# remove candidates it was given.
DEFAULT_OVERFETCH = 3


def _paper_text(title: str | None, abstract: str | None,
                limit: int = 500) -> str:
    return f"{(title or '').strip()} {(abstract or '').strip()[:limit]}".strip()


def filter_by_similarity(claim: str, items: list[dict],
                         threshold: float, text_key: str = "text",
                         ) -> tuple[list[dict], dict]:
    """Keep items whose embedding is close enough to `claim`.

    `items` is a list of dicts, each carrying `text_key`. Returns the kept items
    (each annotated with `similarity`) sorted by descending similarity, plus a
    diagnostics dict that MUST be disclosed to the consumer.
    """
    if not items:
        return [], {"raw_count": 0, "kept_count": 0, "filtered_count": 0,
                    "threshold": threshold, "semantic_filtered": True,
                    "max_similarity": None}
    if not (claim or "").strip():
        return items, {"raw_count": len(items), "kept_count": len(items),
                       "filtered_count": 0, "threshold": threshold,
                       "semantic_filtered": False,
                       "degraded": "empty_claim", "max_similarity": None}
    try:
        vectors = embedding_service.embed_texts(
            [claim] + [it.get(text_key) or "" for it in items])
    except Exception as exc:
        logger.warning("Semantic recall degraded (embeddings unavailable): %s",
                       str(exc)[:200])
        return items, {"raw_count": len(items), "kept_count": len(items),
                       "filtered_count": 0, "threshold": threshold,
                       "semantic_filtered": False,
                       "degraded": str(exc)[:200], "max_similarity": None}

    scored = []
    for idx, item in enumerate(items):
        sim = embedding_service.cosine_similarity(vectors[0], vectors[idx + 1])
        scored.append({**item, "similarity": round(sim, 4)})
    scored.sort(key=lambda it: -it["similarity"])

    kept = [it for it in scored if it["similarity"] >= threshold]
    return kept, {
        "raw_count": len(items),
        "kept_count": len(kept),
        "filtered_count": len(items) - len(kept),
        "threshold": threshold,
        "semantic_filtered": True,
        "max_similarity": scored[0]["similarity"] if scored else None,
        # The best rejected score tells the consumer whether the filter cut a
        # clear boundary (best rejected << threshold) or just trimmed a
        # continuous tail (best rejected close to threshold).
        "best_rejected_similarity": (
            round(max((it["similarity"] for it in scored
                       if it["similarity"] < threshold), default=0.0), 4)),
    }


async def semantic_recall(claim: str, queries: list[str],
                          limit: int = 10,
                          allowed_sources: list[str] | None = None,
                          threshold: float | None = None,
                          overfetch: int = DEFAULT_OVERFETCH,
                          ) -> tuple[list[dict], dict]:
    """Keyword-retrieve across sources, then filter/rerank by embedding.

    Returns (kept, diagnostics). `kept` items are dicts with title/year/venue/
    abstract/similarity. Diagnostics include everything needed to disclose the
    filtering honestly.
    """
    from app.services.search_service import SearchService

    threshold = (settings.semantic_recall_threshold
                 if threshold is None else threshold)
    queries = [q.strip() for q in (queries or []) if q and q.strip()]
    if not queries:
        return [], {"raw_count": 0, "kept_count": 0, "filtered_count": 0,
                    "threshold": threshold, "semantic_filtered": True,
                    "queries": [], "max_similarity": None}

    svc = SearchService()
    try:
        raw = await svc.search_multiple_queries(
            queries, limit=max(limit * overfetch, limit),
            allowed_sources=allowed_sources)
    except Exception as exc:
        logger.warning("Semantic recall search failed: %s", str(exc)[:200])
        return [], {"raw_count": 0, "kept_count": 0, "filtered_count": 0,
                    "threshold": threshold, "semantic_filtered": False,
                    "degraded": f"search_failed: {str(exc)[:150]}",
                    "queries": queries, "max_similarity": None}

    seen, items = set(), []
    for p in raw:
        key = (p.title or "").strip().lower()[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        items.append({
            "title": p.title,
            "year": getattr(p, "year", None),
            "venue": getattr(p, "venue", None),
            "abstract": (getattr(p, "abstract", None) or "")[:700],
            "text": _paper_text(p.title, getattr(p, "abstract", None)),
        })

    kept, diag = filter_by_similarity(claim, items, threshold)
    diag["queries"] = queries
    # Hand back the best `limit` after filtering: the filter decides what is
    # *relevant*, the limit decides how much the consumer can read.
    kept = kept[:limit]
    diag["returned_count"] = len(kept)
    return kept, diag

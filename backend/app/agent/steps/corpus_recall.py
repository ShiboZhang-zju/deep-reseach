"""Corpus-wide prior-art recall for the gap audit's comparison pool.

Why this exists (killer-funnel diagnosis, 2026-11-02)
----------------------------------------------------
The fixed-budget pilot had 37 gaps independently reviewed against prior art.
For the 82 killer papers the reviewer named that were ALREADY in the task
corpus (title match >= 0.9), the funnel was:

    selected as an audit neighbour ..............  4  ( 4.9%)
    screened by gap-relevance, not selected .....  6  ( 7.3%)
    in the audit pool, cut by the 10-paper cap .. 17  (20.7%)
    in the corpus, never admitted to any pool ... 55  (67.1%)

The audit pool was built from (a) the gap's own adversarial search results and
(b) `collect_same_question_neighbors`, which walks only the evidence links of
the gap's *own* research questions. A direct killer that the task corpus holds
but that was never linked to those questions was structurally invisible. All 37
audits then ran on exactly 3 neighbours and reported `confirmed` — which is how
33 confirmed gaps produced 0 publishable gaps under independent review.

So the missing capability is not better queries; it is recall over material the
system already paid to download. This module supplies it: embed the gap's claim
once, embed the task's paper titles+abstracts, keep the closest ones.

Contract
--------
* Corpus-only: it never performs a network search and never invents papers.
* Soft and disclosed: the caller receives hit counts and the threshold, and is
  expected to trace them. A recall pass that returns nothing must look
  different from a recall pass that was never run.
* Fail-open: embeddings unavailable -> returns ([], diag with degraded=...).
  An infrastructure failure is not evidence about the gap's novelty.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.db.models import Paper, TaskPaper
from app.services import embedding_service

logger = logging.getLogger(__name__)

# Minimum characters of paper text before the paper is worth embedding. Title
# alone is allowed (many corpus rows have abstracts); an empty row is not.
_MIN_TEXT_CHARS = 12


def _paper_text(title: str | None, abstract: str | None, limit: int = 600) -> str:
    return f"{(title or '').strip()} {(abstract or '').strip()[:limit]}".strip()


def _exclude_paper_ids(db, gap) -> set[str]:
    """Papers that must not be used as the gap's own comparison material.

    A claim cannot be falsified by the very evidence it was derived from: the
    papers behind the gap's cited evidence are its premises, not prior art.
    """
    from app.agent.steps.audit_gaps import collect_gap_support_papers

    return set(collect_gap_support_papers(db, gap))


def recall_corpus_neighbors(db, gap, task_id: str,
                            top_k: int | None = None,
                            threshold: float | None = None,
                            ) -> tuple[list[str], dict]:
    """Return (paper_ids, diagnostics) for corpus papers close to the gap claim.

    Ordered by descending similarity. `paper_ids` excludes the gap's own
    supporting papers and excludes nothing else — the caller unions these into
    its candidate pool and lets the gap-relevance screen do the fine judgement.
    """
    top_k = settings.audit_corpus_recall_top_k if top_k is None else top_k
    threshold = (settings.audit_corpus_recall_threshold
                 if threshold is None else threshold)
    max_papers = max(int(settings.audit_corpus_recall_max_papers), 1)

    claim = _claim_text(gap)
    diag: dict = {
        "enabled": bool(settings.audit_corpus_recall_enabled),
        "claim_chars": len(claim),
        "threshold": threshold,
        "top_k": top_k,
        "corpus_count": 0,
        "candidate_count": 0,
        "kept_count": 0,
        "returned_count": 0,
        "max_similarity": None,
        "semantic_filtered": False,
    }
    if not settings.audit_corpus_recall_enabled:
        diag["degraded"] = "disabled"
        return [], diag
    if not claim:
        diag["degraded"] = "empty_claim"
        return [], diag

    rows = (db.query(Paper.id, Paper.title, Paper.abstract)
            .join(TaskPaper, TaskPaper.paper_id == Paper.id)
            .filter(TaskPaper.task_id == task_id)
            .limit(max_papers)
            .all())
    diag["corpus_count"] = len(rows)
    if not rows:
        diag["degraded"] = "empty_corpus"
        return [], diag

    excluded = _exclude_paper_ids(db, gap)
    items = []
    for paper_id, title, abstract in rows:
        if paper_id in excluded:
            continue
        text = _paper_text(title, abstract)
        if len(text) < _MIN_TEXT_CHARS:
            continue
        items.append({"paper_id": paper_id, "text": text})
    diag["candidate_count"] = len(items)
    diag["excluded_support_papers"] = len(excluded)
    if not items:
        diag["degraded"] = "no_usable_rows"
        return [], diag

    try:
        vectors = embedding_service.embed_texts([claim] + [it["text"] for it in items])
    except Exception as exc:  # fail-open: infra failure is not novelty evidence
        logger.warning("Gap %s: corpus recall degraded (embeddings unavailable): %s",
                       gap.id[:8], str(exc)[:200])
        diag["degraded"] = f"embedding_failed: {str(exc)[:150]}"
        return [], diag

    if len(vectors) != len(items) + 1:
        diag["degraded"] = "embedding_size_mismatch"
        return [], diag

    scored = []
    for idx, item in enumerate(items):
        sim = embedding_service.cosine_similarity(vectors[0], vectors[idx + 1])
        scored.append((round(sim, 4), item["paper_id"]))
    scored.sort(reverse=True)

    diag["max_similarity"] = scored[0][0] if scored else None
    diag["best_rejected_similarity"] = round(
        max((s for s, _ in scored if s < threshold), default=0.0), 4)
    kept = [pid for sim, pid in scored if sim >= threshold]
    diag["kept_count"] = len(kept)
    diag["filtered_count"] = len(scored) - len(kept)
    diag["semantic_filtered"] = True
    selected = kept[:max(top_k, 0)]
    diag["returned_count"] = len(selected)
    diag["top_similarities"] = [s for s, _ in scored[:5]]
    return selected, diag


def _claim_text(gap) -> str:
    """The gap material the recall is anchored on.

    `claimed_delta` is the audited object, so it leads. The context fields are
    appended because a bare delta ("...has not been measured") is a poor
    embedding anchor on its own — it matches any measurement paper.
    """
    parts = [
        (gap.claimed_delta or "").strip(),
        (gap.target_setting or "").strip(),
        (gap.missing_capability or "").strip(),
        (gap.observed_problem or "").strip()[:400],
    ]
    return "\n".join(part for part in parts if part)

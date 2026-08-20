"""Gap-specific prior-art relevance screening (0027, false-novelty audit).

Why this exists (fd688ba6): audit-recalled papers were stored with
TaskPaper.final_score = NULL (the gap-search step never scores them), and the
neighbor selector then ranked a broad RAG survey ahead of a directly relevant
abstention paper — so direct prior art never reached the NPA pool and a false
broad gap survived. final_score measures task/RQ relevance, NOT "how close is
this paper to THIS gap".

This module answers exactly that question, cheaply, on title+abstract, BEFORE
the deep NPA audit:

    audit search -> title+abstract gap scoring -> Top-M -> deep NPA -> Top-K

The LLM only emits qualitative judgments (yes/partial/no + addresses_claim_ids)
plus a rationale; relevance_score is aggregated by code with configurable
weights so a 0.83 is never taken verbatim from the model.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.config import settings
from app.db.repositories import gap_repo
from app.db.repositories.gap_repo import (
    get_gap_paper_relevance, list_atomic_claims, upsert_gap_paper_relevance,
)

logger = logging.getLogger(__name__)

# Allowable qualitative labels. Kept as plain strings (not Literal) so an
# out-of-contract value degrades gracefully instead of aborting the audit.
_OVERLAP_LEVELS = ("yes", "partial", "no")


class GapPaperRelevanceSchema(BaseModel):
    paper_id: str = Field(min_length=1)
    problem_overlap: str = Field(
        description="Does the paper address the same research problem as the gap? yes/partial/no")
    mechanism_overlap: str = Field(
        description="Does the paper use the same mechanism/method the gap targets? yes/partial/no")
    evaluation_overlap: str = Field(
        description="Does the paper evaluate the same setting/metric the gap targets? yes/partial/no")
    claim_overlap: str = Field(
        description="Does the paper make progress on any of the gap's atomic claims? yes/partial/no")
    addresses_claim_ids: list[str] = Field(
        default_factory=list,
        description="Gap atomic claim ids this paper directly addresses (subset of provided ids).")
    rationale: str = Field(
        default_factory=str,
        description="One sentence: what the paper does and what it does NOT cover.")


class GapPaperRelevanceList(BaseModel):
    papers: list[GapPaperRelevanceSchema] = Field(default_factory=list)


_GAP_RELEVANCE_SYSTEM = """You judge how CLOSE a paper is to one specific research gap.

You only read the gap's own material and one paper's title + abstract. Answer
whether the paper already makes progress on THIS gap — not whether it is a good
paper and not whether it is related to the general topic.

For each paper output structured judgments:

- problem_overlap: does it address the same underlying problem the gap claims is
  unsolved? yes = it solves/attempts the same problem; partial = adjacent but
  with a meaningfully different target; no = different problem.
- mechanism_overlap: does it use the mechanism/method the gap says is missing?
  yes / partial / no.
- evaluation_overlap: does it evaluate the same setting, metric, or benchmark
  the gap targets? yes / partial / no.
- claim_overlap: does it make progress on ANY of the gap's atomic claims?
  yes = at least one claim is substantially addressed; partial = touches but
  does not fully deliver a claim; no = none.

addresses_claim_ids: the subset of the provided atomic claim ids the paper
directly addresses. A survey or review that merely mentions the topic must NOT
be listed as addressing claims.

rationale: one sentence stating what the paper does and what it leaves open.

Never guess numbers. Output only the requested fields."""


def _aggregate_relevance(result: GapPaperRelevanceSchema) -> float:
    """Aggregate qualitative overlap labels into a single gap-relevance score.

    claim_overlap carries the most weight (it is the set-subtraction ground
    truth for the NPA residual), then problem, then evaluation. Weights are
    configurable (settings.gap_relevance_*_weight) and deliberately NOT
    hard-coded scientific truth.
    """
    def _level_value(label: str) -> float:
        label = (label or "").strip().lower()
        if label == "yes":
            return 1.0
        if label == "partial":
            return 0.5
        return 0.0

    claim = _level_value(result.claim_overlap)
    problem = _level_value(result.problem_overlap)
    evaluation = _level_value(result.evaluation_overlap)
    return round(
        settings.gap_relevance_claim_weight * claim
        + settings.gap_relevance_problem_weight * problem
        + settings.gap_relevance_evaluation_weight * evaluation,
        4,
    )


def _gap_context(gap, claims) -> str:
    """The ONLY gap material the scorer sees — no task/RQ context."""
    lines = [
        f"Gap type: {gap.gap_type or ''}",
        f"Target setting: {gap.target_setting or ''}",
        f"Missing capability: {gap.missing_capability or ''}",
        f"Claimed delta: {gap.claimed_delta or ''}",
        f"Existing coverage: {gap.existing_coverage or ''}",
    ]
    if claims:
        lines.append("Atomic claims:")
        for c in claims:
            lines.append(f"- [{c.id}] {c.claim_text}")
    return "\n".join(lines)


def _paper_prompt(paper) -> str:
    abstract = (getattr(paper, "abstract", None) or "").strip()
    lines = [f"Paper id: {paper.id}", f"Title: {paper.title or ''}"]
    if abstract:
        lines.append(f"Abstract: {abstract[:2000]}")
    else:
        lines.append("Abstract: (none — title only)")
    return "\n".join(lines)


async def score_gap_papers(
    db, llm, gap, paper_ids: list[str], task_id: str,
    max_papers: int = 0,
) -> list[tuple[str, float, GapPaperRelevanceSchema]]:
    """Cheap title/abstract relevance screening for audit-recalled papers.

    Batches papers (up to `max_papers`, default settings.gap_relevance_screen_top_m)
    and persists a GapPaperRelevance row per paper. Returns
    [(paper_id, relevance_score, schema), ...] sorted descending.
    """
    from app.db.models import Paper

    claims = list_atomic_claims(db, gap.id)
    gap_text = _gap_context(gap, claims)
    limit = max_papers or settings.gap_relevance_screen_top_m
    papers: list[Paper] = []
    for pid in paper_ids[:limit]:
        # Idempotent: skip papers already scored under the current version.
        existing = get_gap_paper_relevance(db, gap.id, pid)
        if existing is not None and existing.scoring_version == settings.gap_relevance_scoring_version:
            continue
        paper = db.get(Paper, pid)
        if paper is not None:
            papers.append(paper)

    results: list[tuple[str, float, GapPaperRelevanceSchema]] = []
    if not papers:
        # Re-read persisted rows so callers always get the full ranked list.
        for row in gap_repo.list_gap_paper_relevance(db, gap.id):
            results.append((row.paper_id, row.relevance_score, row))
        return results

    # Batch into groups of 5 to bound prompt size.
    batch_size = 5
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        paper_block = "\n\n".join(_paper_prompt(p) for p in batch)
        try:
            parsed = await llm.chat_json([
                {"role": "system", "content": _GAP_RELEVANCE_SYSTEM},
                {"role": "user", "content":
                    f"{gap_text}\n\nPapers to judge:\n\n{paper_block}"},
            ], GapPaperRelevanceList)
        except Exception as exc:
            logger.warning("Gap %s: gap-relevance scoring batch failed (non-fatal): %s",
                           gap.id[:8], exc)
            continue
        for item in parsed.papers:
            # Only accept papers we actually sent (guard against the model
            # hallucinating ids / reusing an id across batches).
            if item.paper_id not in {p.id for p in batch}:
                continue
            score = _aggregate_relevance(item)
            upsert_gap_paper_relevance(
                db, gap_id=gap.id, paper_id=item.paper_id, task_id=task_id,
                relevance_score=score, problem_overlap=item.problem_overlap,
                mechanism_overlap=item.mechanism_overlap,
                evaluation_overlap=item.evaluation_overlap,
                claim_overlap=item.claim_overlap,
                addresses_claim_ids=item.addresses_claim_ids,
                rationale=item.rationale,
                scoring_version=settings.gap_relevance_scoring_version,
            )
            results.append((item.paper_id, score, item))
        db.commit()

    results.sort(key=lambda r: r[1], reverse=True)
    return results


async def score_all_gap_candidates(
    db, llm, gap, candidate_paper_ids: list[str], task_id: str,
) -> list[tuple[str, float, GapPaperRelevanceSchema]]:
    """Screen all audit-recalled candidates; persisted rows are reused on
    replay so re-running never re-hits the LLM for already-scored papers."""
    return await score_gap_papers(db, llm, gap, candidate_paper_ids, task_id)

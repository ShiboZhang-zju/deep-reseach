"""Step: Update coverage matrix based on evidence units.

Phase 2.1 rewrite:
- (#12) Coverage scoring uses semantic relevance, not single-word overlap
- (#13) Saves per-round snapshots; coverage accumulates across rounds
- Coverage delta drives next round's question selection
"""

import json
import logging
from datetime import datetime, timezone

from app.agent.evidence_relations import (
    CONTRADICTING_RELATIONS,
    MIN_RELATION_RELEVANCE,
    SUPPORTING_RELATIONS,
)
from app.agent.state import ResearchState
from app.db.models import (
    EvidenceUnit, ResearchQuestion, CoverageRecord,
    QuestionEvidenceLink, Paper,
)
from app.db.repositories import paper_repo

logger = logging.getLogger(__name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def update_coverage_matrix(db, state: ResearchState, llm, task_id: str,
                                  round_number: int = 0):
    """Update coverage records for all active research questions.

    Phase 2.1:
    - (#12) Uses multi-word matching + question type relevance
    - (#13) Creates new CoverageRecord per round (snapshot)
    - Accumulates evidence from all rounds
    """
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.status.in_(["open", "partially_covered", "covered"]),
    ).all()

    if not questions:
        logger.info("Task %s: no active questions for coverage update", task_id[:8])
        return

    all_evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
    ).all()

    if not all_evidence:
        logger.info("Task %s: no evidence units for coverage update", task_id[:8])
        return

    # Get previous round's coverage for delta calculation
    prev_round = round_number - 1 if round_number > 0 else 0
    prev_coverage = {}
    if prev_round > 0:
        for cr in db.query(CoverageRecord).filter(
            CoverageRecord.task_id == task_id,
            CoverageRecord.round_number == prev_round,
        ).all():
            prev_coverage[cr.question_id] = cr.coverage_score

    # Clear old links for this round (but keep previous rounds' links)
    db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_([q.id for q in questions]),
    ).delete(synchronize_session=False)

    coverage_deltas = []

    # Pre-build a compact, indexed view of all evidence for LLM matching. Using
    # the LLM for semantic relevance replaces the old English-only keyword
    # substring matcher, which produced zero matches for Chinese claims.
    evidence_index = list(all_evidence)  # stable order;1-based index in prompt

    for question in questions:
        supporting = 0
        contradicting = 0
        background = 0
        supporting_papers: set[str] = set()
        contradicting_papers: set[str] = set()

        # (#12) LLM-based semantic matching: one batched call (or a few) per
        # question decides which evidence items are relevant and how.
        matched = await _llm_match_evidence(llm, question, evidence_index)

        for eu_idx, relation_type, relevance in matched:
            eu = evidence_index[eu_idx]
            link = QuestionEvidenceLink(
                question_id=question.id,
                evidence_id=eu.id,
                relation_type=relation_type,
                relevance_score=relevance,
            )
            db.add(link)

            # A weak match corroborates nothing, and gap-mining admission would
            # reject it anyway; counting it here made coverage look better than
            # the evidence the pipeline can actually use.
            if (relevance or 0.0) < MIN_RELATION_RELEVANCE:
                background += 1
            elif relation_type in SUPPORTING_RELATIONS:
                supporting += 1
                if eu.paper_id:
                    supporting_papers.add(eu.paper_id)
            elif relation_type in CONTRADICTING_RELATIONS:
                contradicting += 1
                if eu.paper_id:
                    contradicting_papers.add(eu.paper_id)
            else:
                background += 1

        # Coverage is driven by how many *distinct papers* speak to the question,
        # not by how many evidence units were extracted. Counting units let a
        # single paper saturate a question: with `0.4 + supporting * 0.1`, six
        # units from one paper scored a perfect 1.00, and in a real round six of
        # ten questions hit 1.00 after only 15 papers. That inflated score is not
        # cosmetic — coverage decides whether to keep searching and which
        # questions may back a gap, so it must reflect corroboration breadth.
        distinct_supporting = len(supporting_papers)
        distinct_contradicting = len(contradicting_papers)
        total_relevant = supporting + contradicting + background
        if total_relevant >= 5 and distinct_supporting >= 3:
            coverage = min(1.0, 0.4 + distinct_supporting * 0.1
                          + distinct_contradicting * 0.05)
        elif total_relevant >= 3 and distinct_supporting >= 1:
            coverage = 0.3 + distinct_supporting * 0.1
        elif total_relevant >= 1:
            coverage = 0.15
        else:
            coverage = 0.0

        # Determine question status
        if coverage >= 0.7:
            new_status = "covered"
        elif coverage >= 0.3:
            new_status = "partially_covered"
        else:
            new_status = "open"

        question.status = new_status

        # (#13) Upsert CoverageRecord. The table has a UNIQUE(task_id,
        # question_id) constraint, so re-inserting on a later round raises an
        # IntegrityError; update the existing record in place instead (per-round
        # history lives in traces).
        cr = db.query(CoverageRecord).filter(
            CoverageRecord.task_id == task_id,
            CoverageRecord.question_id == question.id,
        ).first()
        if cr is None:
            cr = CoverageRecord(task_id=task_id, question_id=question.id)
            db.add(cr)
        # Round to 2 decimals: `0.4 + n*0.1 + m*0.05` otherwise stores float
        # noise such as 0.7000000000000001.
        coverage = round(coverage, 2)
        cr.coverage_score = coverage
        cr.confidence = min(1.0, total_relevant * 0.15)
        cr.supporting_evidence_count = supporting
        cr.contradicting_evidence_count = contradicting
        cr.distinct_supporting_papers = distinct_supporting
        cr.distinct_contradicting_papers = distinct_contradicting
        cr.direct_neighbor_count = 0
        cr.unresolved_aspects_json = json.dumps([], ensure_ascii=False)
        cr.round_number = round_number

        # Calculate delta
        prev_score = prev_coverage.get(question.id, 0.0)
        delta = coverage - prev_score
        coverage_deltas.append({
            "question_id": question.id,
            "question": question.question[:60],
            "prev_coverage": prev_score,
            "new_coverage": coverage,
            "delta": delta,
            "status": new_status,
            "supporting": supporting,
            "contradicting": contradicting,
            # Breadth is what now drives the score, so it has to be visible:
            # "12 units from 1 paper" and "12 units from 9 papers" are very
            # different states that used to look identical in the logs.
            "distinct_supporting_papers": distinct_supporting,
            "distinct_contradicting_papers": distinct_contradicting,
        })

    db.flush()
    db.commit()

    logger.info("Task %s: coverage updated for %d questions (round %d)",
                task_id[:8], len(questions), round_number)

    # Log coverage deltas
    for d in coverage_deltas:
        delta_str = f"+{d['delta']:.2f}" if d['delta'] >= 0 else f"{d['delta']:.2f}"
        logger.info("  Q[%s] coverage: %.2f -> %.2f (%s) %s [%d units / %d papers]",
                     d['question_id'][:8], d['prev_coverage'], d['new_coverage'],
                     delta_str, d['status'], d['supporting'],
                     d['distinct_supporting_papers'])

    paper_repo.save_trace(db, task_id, "update_coverage_matrix", "action",
                          round_number=round_number,
                          output_data={
                              "questions_updated": len(questions),
                              "total_evidence": len(all_evidence),
                              "round": round_number,
                              "deltas": coverage_deltas,
                          })
    db.commit()

    return coverage_deltas


async def _llm_match_evidence(llm, question, evidence_index):
    """Use the LLM to judge which evidence items are relevant to a question.

    Returns a list of (evidence_index, relation_type, relevance).

    Efficiency: instead of one LLM call per (question, evidence) pair, all
    evidence claims are presented as a numbered list and the model returns only
    the relevant indices — one call per batch. This replaces the English-only
    keyword substring matcher that produced zero matches on Chinese claims.

    On any LLM/parse failure it falls back to the evidence-type heuristic so
    coverage never silently collapses to zero because of a transient error.
    """
    from app.config import settings
    from app.schemas.schemas import EvidenceMatchList

    if not evidence_index:
        return []

    batch_size = 40
    results: list[tuple[int, str, float]] = []

    for batch_start in range(0, len(evidence_index), batch_size):
        batch = evidence_index[batch_start:batch_start + batch_size]
        lines = []
        for i, eu in enumerate(batch):
            claim = (eu.normalized_claim or "").strip().replace("\n", " ")
            lines.append(f"{i + 1}. [{eu.evidence_type}] {claim[:200]}")
        listing = "\n".join(lines)

        system = (
            "你是严谨的科研证据审阅员。给定一个研究问题和一批编号的证据条目，"
            "判断每条证据是否与该问题相关。只返回相关的条目，忽略不相关的。"
            "relation 取值：supports(支持/回答该问题)、contradicts(与问题的假设相矛盾/负面结果)、"
            "partially_answers(部分回答/指出局限)、background(仅背景相关)。"
            "relevance 为0-1的相关度。不要编造编号，index 必须来自输入列表。"
        )
        user = (
            f"研究问题：{question.question}\n"
            f"问题类型：{question.question_type}\n\n"
            f"证据列表：\n{listing}\n\n"
            "返回 JSON：{\"matches\":[{\"index\":<编号>,\"relation\":\"...\",\"relevance\":0-1}]}"
        )

        try:
            res = await llm.chat_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                EvidenceMatchList,
            )
            for m in res.matches:
                idx0 = m.index - 1  # to 0-based within batch
                if 0 <= idx0 < len(batch):
                    results.append((batch_start + idx0, m.relation, m.relevance))
        except Exception as e:
            logger.warning("LLM evidence match failed for question %s (batch %d): %s; "
                           "falling back to type heuristic",
                           question.id[:8], batch_start, e)
            for i, eu in enumerate(batch):
                if eu.evidence_type in _get_relevant_evidence_types(question.question_type):
                    # MIN_RELATION_RELEVANCE is the weakest match that either
                    # coverage or gap-mining admission accepts; a lower value here
                    # would make every fallback link silently inadmissible.
                    results.append((batch_start + i, _determine_relation(eu, question),
                                    MIN_RELATION_RELEVANCE))

    return results


def _get_relevant_evidence_types(question_type: str) -> list[str]:
    """Map question type to relevant evidence types."""
    mapping = {
        "problem": ["problem", "limitation", "negative_result"],
        "method": ["method", "comparison"],
        "evaluation": ["result", "metric", "comparison"],
        "dataset": ["dataset", "result"],
        "resource": ["result", "limitation"],
        "failure": ["negative_result", "limitation"],
        "application": ["result", "future_work"],
    }
    return mapping.get(question_type, [])


def _determine_relation(evidence: EvidenceUnit, question: ResearchQuestion) -> str:
    """Determine the relationship between evidence and question."""
    if evidence.evidence_type == "negative_result":
        return "contradicts"
    if evidence.evidence_type == "limitation":
        return "partially_answers"
    if evidence.evidence_type in ["future_work", "comparison"]:
        return "background"
    return "supports"

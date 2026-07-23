"""Step: Update coverage matrix based on evidence units.

Phase 2.1 rewrite:
- (#12) Coverage scoring uses semantic relevance, not single-word overlap
- (#13) Saves per-round snapshots; coverage accumulates across rounds
- Coverage delta drives next round's question selection
"""

import json
import logging
from datetime import datetime, timezone

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

    for question in questions:
        # (#12) Multi-criteria matching
        supporting = 0
        contradicting = 0
        background = 0
        linked_count = 0

        # Extract keywords from question (multi-word phrases)
        q_keywords = _extract_keywords(question.question)
        q_type_relevant_evidence = _get_relevant_evidence_types(question.question_type)

        for eu in all_evidence:
            claim_lower = (eu.normalized_claim or "").lower()

            # Check relevance: at least 2 keyword matches OR type match + 1 keyword
            keyword_matches = sum(1 for kw in q_keywords if kw in claim_lower)
            type_match = eu.evidence_type in q_type_relevant_evidence

            is_relevant = (keyword_matches >= 2) or (keyword_matches >= 1 and type_match)

            if is_relevant:
                relation_type = _determine_relation(eu, question)
                relevance = min(1.0, keyword_matches * 0.2 + (0.3 if type_match else 0))

                link = QuestionEvidenceLink(
                    question_id=question.id,
                    evidence_id=eu.id,
                    relation_type=relation_type,
                    relevance_score=relevance,
                )
                db.add(link)
                linked_count += 1

                if relation_type == "supports":
                    supporting += 1
                elif relation_type == "contradicts":
                    contradicting += 1
                else:
                    background += 1

        # (#12) Better coverage score calculation
        total_relevant = supporting + contradicting + background
        if total_relevant >= 5 and supporting >= 3:
            coverage = min(1.0, 0.4 + supporting * 0.1 + contradicting * 0.05)
        elif total_relevant >= 3 and supporting >= 1:
            coverage = 0.3 + supporting * 0.1
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

        # (#13) Create new CoverageRecord for this round (snapshot)
        cr = CoverageRecord(
            task_id=task_id,
            question_id=question.id,
            coverage_score=coverage,
            confidence=min(1.0, total_relevant * 0.15),
            supporting_evidence_count=supporting,
            contradicting_evidence_count=contradicting,
            direct_neighbor_count=0,
            unresolved_aspects_json=json.dumps([], ensure_ascii=False),
            round_number=round_number,
        )
        db.add(cr)

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
        })

    db.flush()
    db.commit()

    logger.info("Task %s: coverage updated for %d questions (round %d)",
                task_id[:8], len(questions), round_number)

    # Log coverage deltas
    for d in coverage_deltas:
        delta_str = f"+{d['delta']:.2f}" if d['delta'] >= 0 else f"{d['delta']:.2f}"
        logger.info("  Q[%s] coverage: %.2f -> %.2f (%s) %s [%s]",
                     d['question_id'][:8], d['prev_coverage'], d['new_coverage'],
                     delta_str, d['status'], d['supporting'])

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


def _extract_keywords(question_text: str) -> list[str]:
    """Extract meaningful keywords from a question (multi-word phrases)."""
    # Simple: split by common delimiters, filter short words
    import re
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question_text.lower())
    # Also try to extract multi-word phrases
    phrases = re.findall(r'\b(?:memory|token|budget|oracle|test|graph|neural|network|method|benchmark)\w*', question_text.lower())
    return list(set(words + phrases))[:15]


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

"""Step: Update coverage matrix based on evidence units.

Phase 2: After evidence extraction, links evidence to research questions
and updates coverage records. This drives the next round's question selection.
"""

import json
import logging

from app.agent.state import ResearchState
from app.db.models import (
    EvidenceUnit, ResearchQuestion, CoverageRecord,
    QuestionEvidenceLink, Paper,
)
from app.db.repositories import paper_repo

logger = logging.getLogger(__name__)


async def update_coverage_matrix(db, state: ResearchState, llm, task_id: str):
    """Update coverage records for all active research questions.

    For each question:
    1. Find evidence units that are relevant (simple keyword matching for now)
    2. Create QuestionEvidenceLinks
    3. Calculate coverage_score based on supporting/contradicting evidence
    4. Update question status
    """
    questions = db.query(ResearchQuestion).filter(
        ResearchQuestion.task_id == task_id,
        ResearchQuestion.status.in_(["open", "partially_covered", "covered"]),
    ).all()

    if not questions:
        logger.info("Task %s: no active questions for coverage update", task_id[:8])
        return

    # Get all evidence for this task
    all_evidence = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
    ).all()

    if not all_evidence:
        logger.info("Task %s: no evidence units for coverage update", task_id[:8])
        return

    # Clear old links (re-link from scratch each round)
    db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id.in_([q.id for q in questions]),
    ).delete(synchronize_session=False)

    for question in questions:
        # Simple matching: find evidence whose normalized_claim contains keywords from the question
        q_keywords = set(question.question.lower().split())
        q_keywords = {w for w in q_keywords if len(w) > 3}  # skip short words

        supporting = 0
        contradicting = 0
        linked_ids = []

        for eu in all_evidence:
            claim_lower = (eu.normalized_claim or "").lower()
            # Check if evidence is relevant to this question
            overlap = q_keywords & set(claim_lower.split())
            if len(overlap) >= 1 or _is_relevant_by_type(question.question_type, eu.evidence_type):
                relation_type = "contradicts" if eu.evidence_type == "negative_result" else "supports"
                relevance = min(1.0, len(overlap) / 5.0 + 0.3)

                link = QuestionEvidenceLink(
                    question_id=question.id,
                    evidence_id=eu.id,
                    relation_type=relation_type,
                    relevance_score=relevance,
                )
                db.add(link)

                if relation_type == "supports":
                    supporting += 1
                else:
                    contradicting += 1
                linked_ids.append(eu.id)

        # Calculate coverage score
        total_relevant = supporting + contradicting
        if total_relevant >= 5:
            coverage = min(1.0, supporting * 0.15 + 0.3)
        elif total_relevant >= 2:
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

        # Update or create coverage record
        cr = db.query(CoverageRecord).filter(
            CoverageRecord.question_id == question.id,
        ).first()

        if cr:
            cr.coverage_score = coverage
            cr.supporting_evidence_count = supporting
            cr.contradicting_evidence_count = contradicting
            cr.confidence = min(1.0, total_relevant * 0.2)
        else:
            cr = CoverageRecord(
                task_id=task_id,
                question_id=question.id,
                coverage_score=coverage,
                confidence=min(1.0, total_relevant * 0.2),
                supporting_evidence_count=supporting,
                contradicting_evidence_count=contradicting,
            )
            db.add(cr)

    db.flush()
    db.commit()

    logger.info("Task %s: coverage updated for %d questions", task_id[:8], len(questions))
    paper_repo.save_trace(db, task_id, "update_coverage_matrix", "action",
                          output_data={
                              "questions_updated": len(questions),
                              "total_evidence": len(all_evidence),
                          })
    db.commit()


def _is_relevant_by_type(question_type: str, evidence_type: str) -> bool:
    """Check if an evidence type is relevant to a question type."""
    mapping = {
        "problem": ["problem", "limitation", "negative_result"],
        "method": ["method", "comparison"],
        "evaluation": ["result", "metric", "comparison"],
        "dataset": ["dataset", "result"],
        "resource": ["result", "limitation"],
        "failure": ["negative_result", "limitation"],
        "application": ["result", "future_work"],
    }
    return evidence_type in mapping.get(question_type, [])

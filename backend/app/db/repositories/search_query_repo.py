"""Search query repository — CRUD for SearchQueryRecord."""

import re
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.models import SearchQueryRecord


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize(text: str) -> str:
    """Normalize query text for uniqueness."""
    return re.sub(r'\s+', ' ', (text or '').strip().lower())


def save_search_query(db: Session, task_id: str, query_text: str, intent: str,
                      target_question_id: str | None, expected_evidence_type: str | None,
                      round_number: int) -> SearchQueryRecord:
    """Save a structured search query with normalized_query_text."""
    normalized = _normalize(query_text)

    # Check for existing (idempotent)
    existing = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task_id,
        SearchQueryRecord.round_number == round_number,
        SearchQueryRecord.normalized_query_text == normalized,
        SearchQueryRecord.target_question_id == target_question_id,
    ).first()
    if existing:
        return existing

    record = SearchQueryRecord(
        task_id=task_id,
        query_text=query_text,
        normalized_query_text=normalized,
        intent=intent,
        target_question_id=target_question_id,
        expected_evidence_type=expected_evidence_type,
        round_number=round_number,
        status="pending",
    )
    db.add(record)
    db.flush()
    return record


def update_query_results(db: Session, query_id: str, result_count: int,
                         new_paper_count: int, evidence_unit_count: int = 0,
                         status: str = "completed", error: str = None):
    """Update a search query with results and completion time."""
    record = db.get(SearchQueryRecord, query_id)
    if record:
        record.result_count = result_count
        record.new_paper_count = new_paper_count
        record.evidence_unit_count = evidence_unit_count
        record.status = status
        record.execution_error = error
        record.completed_at = _utcnow()
        db.flush()


def get_queries_for_round(db: Session, task_id: str, round_number: int) -> list[SearchQueryRecord]:
    """Get all search queries for a specific round."""
    return db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task_id,
        SearchQueryRecord.round_number == round_number,
    ).all()


def get_queries_for_question(db: Session, question_id: str) -> list[SearchQueryRecord]:
    """Get all search queries targeting a specific question."""
    return db.query(SearchQueryRecord).filter(
        SearchQueryRecord.target_question_id == question_id,
    ).all()


def get_last_round_with_queries(db: Session, task_id: str) -> int:
    """Get the highest round number that has queries."""
    result = db.query(SearchQueryRecord.round_number).filter(
        SearchQueryRecord.task_id == task_id,
    ).order_by(SearchQueryRecord.round_number.desc()).first()
    return result[0] if result else 0

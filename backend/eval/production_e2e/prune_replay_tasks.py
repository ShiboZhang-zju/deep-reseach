"""Prune replay tasks and all their dependent rows from the production DB.

Replay tasks (ResearchTask.status == 'replay_v7' by default) are eval-harness
artifacts; a failed or interrupted replay sweep leaves half-built tasks whose
gaps can hang in "auditing" and whose copied evidence pollutes per-topic
counters. This tool removes them — and ONLY them: every delete is keyed on
the task ids selected by the status filter, so production tasks are never
touched, and the shared papers table (no task_id column) is never touched.

Usage:
    cd backend
    python -m eval.production_e2e.prune_replay_tasks --status replay_v7
"""
from __future__ import annotations

import argparse
import io
import sys

from sqlalchemy import or_

from app.db.models import (
    Base,
    EvidenceUnit,
    GapCandidate,
    QuestionEvidenceLink,
    ResearchQuestion,
    ResearchTask,
)
from app.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", default="replay_v7",
                        help="task status filter identifying the replay tasks to prune")
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="prune exactly these task ids (e.g. empty orphan shells left "
                             "by an interrupted sweep) instead of the status filter")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        if args.task_ids:
            task_ids = args.task_ids
            tasks = session.query(ResearchTask).filter(
                ResearchTask.id.in_(task_ids)).all()
            print(f"pruning {len(tasks)} task(s) by explicit id "
                  f"({len(task_ids) - len(tasks)} not found)")
        else:
            replay_tasks = session.query(ResearchTask).filter(
                ResearchTask.status == args.status).all()
            if not replay_tasks:
                print(f"no '{args.status}' tasks to prune")
                return
            task_ids = [task.id for task in replay_tasks]
            print(f"pruning {len(task_ids)} '{args.status}' task(s)")

        question_ids = [row[0] for row in session.query(ResearchQuestion.id).filter(
            ResearchQuestion.task_id.in_(task_ids)).all()]
        evidence_ids = [row[0] for row in session.query(EvidenceUnit.id).filter(
            EvidenceUnit.task_id.in_(task_ids)).all()]
        gap_ids = [row[0] for row in session.query(GapCandidate.id).filter(
            GapCandidate.task_id.in_(task_ids)).all()]
        print(f"children: {len(question_ids)} questions, "
              f"{len(evidence_ids)} evidence units, {len(gap_ids)} gaps")

        connection = session.connection()

        def _delete(table, where):
            result = connection.execute(table.delete().where(where))
            print(f"{table.name}: -{result.rowcount}")

        # Association tables first (keyed by child ids, not task_id).
        if question_ids or evidence_ids:
            clauses = []
            if question_ids:
                clauses.append(QuestionEvidenceLink.__table__.c.question_id.in_(question_ids))
            if evidence_ids:
                clauses.append(QuestionEvidenceLink.__table__.c.evidence_id.in_(evidence_ids))
            _delete(QuestionEvidenceLink.__table__, or_(*clauses))
        if gap_ids:
            for name in ("gap_evidence_links", "gap_audits"):
                table = Base.metadata.tables.get(name)
                if table is None or "gap_id" not in table.columns:
                    continue
                _delete(table, table.c.gap_id.in_(gap_ids))

        # Then every table with a task_id column.
        for table in Base.metadata.sorted_tables:
            if "task_id" not in table.columns:
                continue
            _delete(table, table.c.task_id.in_(task_ids))

        session.commit()
        print("prune committed")
    finally:
        session.close()


if __name__ == "__main__":
    # Rebind stdout to UTF-8 only when run as a script: doing it at import
    # time replaces whatever stream the host holds (pytest capture, an IDE
    # runner, a notebook), which makes the host fail at teardown with
    # "I/O operation on closed file".
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()

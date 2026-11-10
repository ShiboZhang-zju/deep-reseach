"""P5 (probe df744606): extraction_status must be explicit for non-candidates.

130 low-priority papers carried NULL extraction_status after the probe run —
indistinguishable from pre-0034 historical rows. After extraction, papers
outside the high/medium candidate set are now stamped
not_assessed / below_priority_floor.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


@pytest.mark.asyncio
async def test_low_priority_papers_get_explicit_not_assessed(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps import extract_evidence as module
    from app.db.models import Paper, ResearchContract, ResearchTask, TaskPaper

    db = temp_db()
    task = ResearchTask(user_input="probe", status="extracting_evidence")
    db.add(task)
    db.flush()
    low_paper = Paper(title="Low Paper", abstract="a")
    high_paper = Paper(title="High Paper", abstract="b")
    db.add_all([low_paper, high_paper])
    db.flush()
    db.add_all([
        TaskPaper(task_id=task.id, paper_id=low_paper.id, discovered_round=1,
                  priority="low", final_score=0.3),
        TaskPaper(task_id=task.id, paper_id=high_paper.id, discovered_round=1,
                  priority="high", final_score=0.9),
    ])
    db.commit()

    # No PDFs and a failing LLM: extraction yields nothing but must still run
    # the not_assessed backfill for the non-candidate papers.
    class FailingLLM:
        async def chat_json(self, *args, **kwargs):
            raise RuntimeError("no endpoint")

    async def no_pdfs(papers, task_id, concurrency=3):
        return {}

    async def no_prewarm(papers, task_id):
        return {"retrieved": 0, "total": len(papers), "unavailable": len(papers),
                "reasons": {"pdf_not_retrieved": len(papers)}}

    monkeypatch.setattr(module, "_download_pdfs", no_pdfs)
    monkeypatch.setattr(module, "prewarm_pdfs", no_prewarm)
    monkeypatch.setattr(module, "_classify_paper_roles_safe",
                        lambda db_, tid, papers: _noop())

    state = ResearchState(task_id=task.id)
    await module.extract_evidence_units(db, state, FailingLLM(), task.id, round_number=1)

    low_tp = db.query(TaskPaper).filter(TaskPaper.paper_id == low_paper.id).one()
    assert low_tp.extraction_status == "not_assessed"
    assert low_tp.extraction_gap_reason == "below_priority_floor"
    # The high-priority paper is NOT stamped by the backfill (its status is
    # set by the extraction path itself — here abstract too short => nothing,
    # but never the below_priority_floor reason).
    high_tp = db.query(TaskPaper).filter(TaskPaper.paper_id == high_paper.id).one()
    assert high_tp.extraction_gap_reason != "below_priority_floor"
    db.close()


async def _noop():
    return None

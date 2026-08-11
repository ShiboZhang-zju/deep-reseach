"""Coverage must reflect corroboration breadth, not evidence-unit count."""

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


def _seed(db, *, paper_count, units_per_paper):
    """Create one question plus supporting evidence spread over N papers."""
    from app.db.models import (EvidenceUnit, Paper, ResearchContract, ResearchQuestion,
                               ResearchTask)

    task = ResearchTask(user_input="long horizon tool use", status="updating_coverage")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Tool Use", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="长任务中的错误累积如何被量化评估？",
        question_type="evaluation", importance=0.9, status="open")
    db.add(question)
    db.flush()
    for paper_index in range(paper_count):
        paper = Paper(title=f"Paper {paper_index}", abstract="abstract")
        db.add(paper)
        db.flush()
        for unit_index in range(units_per_paper):
            db.add(EvidenceUnit(
                task_id=task.id, paper_id=paper.id, evidence_type="limitation",
                normalized_claim=f"claim {paper_index}-{unit_index}",
                extraction_method="pdf_fulltext", verification_status="verified"))
    db.commit()
    return task, question


async def _run_with_all_supporting(db, monkeypatch, task_id):
    """Run the coverage update with every evidence unit counted as supporting."""
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module

    async def _all_supports(llm, question, evidence_index):
        return [(index, "supports", 0.9) for index in range(len(evidence_index))]

    monkeypatch.setattr(module, "_llm_match_evidence", _all_supports)
    state = ResearchState(task_id=task_id, current_round=1)
    return await module.update_coverage_matrix(db, state, object(), task_id, round_number=1)


@pytest.mark.asyncio
async def test_one_paper_cannot_saturate_a_question(temp_db, monkeypatch):
    """12 units from a single paper is weak corroboration, not full coverage.

    With the old unit-count formula (0.4 + supporting * 0.1) this scored a
    perfect 1.00, and a real round had 6 of 10 questions at 1.00 after 15 papers.
    """
    db = temp_db()
    task, question = _seed(db, paper_count=1, units_per_paper=12)

    deltas = await _run_with_all_supporting(db, monkeypatch, task.id)

    assert len(deltas) == 1
    assert deltas[0]["distinct_supporting_papers"] == 1
    assert deltas[0]["supporting"] == 12
    assert deltas[0]["new_coverage"] < 0.7, "a single paper must not mark a question covered"
    db.close()


@pytest.mark.asyncio
async def test_many_papers_do_reach_full_coverage(temp_db, monkeypatch):
    db = temp_db()
    task, question = _seed(db, paper_count=8, units_per_paper=1)

    deltas = await _run_with_all_supporting(db, monkeypatch, task.id)

    assert deltas[0]["distinct_supporting_papers"] == 8
    assert deltas[0]["new_coverage"] == 1.0
    assert deltas[0]["status"] == "covered"
    db.close()


@pytest.mark.asyncio
async def test_same_unit_count_scores_higher_when_spread_across_papers(temp_db, monkeypatch):
    """The score must separate 12-from-1 from 12-from-6, which used to tie at 1.00."""
    db_narrow = temp_db()
    task_narrow, _ = _seed(db_narrow, paper_count=1, units_per_paper=12)
    narrow = await _run_with_all_supporting(db_narrow, monkeypatch, task_narrow.id)

    db_broad = temp_db()
    task_broad, _ = _seed(db_broad, paper_count=6, units_per_paper=2)
    broad = await _run_with_all_supporting(db_broad, monkeypatch, task_broad.id)

    assert narrow[0]["supporting"] == broad[0]["supporting"] == 12
    assert broad[0]["new_coverage"] > narrow[0]["new_coverage"]
    db_narrow.close()
    db_broad.close()

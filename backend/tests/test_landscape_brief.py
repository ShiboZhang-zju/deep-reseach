"""Tests for the Research Landscape Brief (O9).

The brief must ALWAYS be produced — even when the pipeline yielded no gaps —
and must never raise on missing data.
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
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _make_task_with_contract(db):
    from app.db.models import ResearchTask, ResearchContract, ResearchQuestion
    task = ResearchTask(user_input="agent memory")
    db.add(task)
    db.commit()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory",
                                status="active", version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    q = ResearchQuestion(task_id=task.id, contract_id=contract.id,
                         question="Does memory fail after state changes?",
                         question_type="failure", status="open", importance=0.9)
    db.add(q)
    db.commit()
    return task, contract, q


def test_landscape_brief_generated_even_with_no_gaps(temp_db):
    """A brief must be produced when the run ends with more_research_required
    and there are no gaps — it should still contain the question tree and
    honest next-step guidance."""
    from app.agent.steps.generate_landscape_brief import build_landscape_brief_markdown

    db = temp_db()
    task, contract, q = _make_task_with_contract(db)

    md = build_landscape_brief_markdown(
        db, task.id, contract.id,
        terminal_status="more_research_required",
        terminal_reason="no_evidence_backed_gap_candidates",
    )

    assert "研究态势简报" in md
    assert "Agent Memory" in md
    assert "本次运行结果" in md
    assert "建议的下一步" in md
    # No gaps section should say so honestly, not crash
    assert "候选研究空白" in md
    db.close()


def test_landscape_brief_handles_missing_contract(temp_db):
    """Brief must not raise even when there is no contract at all."""
    from app.agent.steps.generate_landscape_brief import build_landscape_brief_markdown
    from app.db.models import ResearchTask

    db = temp_db()
    task = ResearchTask(user_input="orphan task")
    db.add(task)
    db.commit()

    md = build_landscape_brief_markdown(db, task.id, None,
                                        terminal_status="failed",
                                        terminal_reason="no_active_contract")
    assert "研究态势简报" in md
    assert "运行" in md
    db.close()


def test_landscape_brief_includes_gap_tiers(temp_db):
    """When gaps exist, the brief lists them with A/B tier from provenance."""
    from app.agent.steps.generate_landscape_brief import build_landscape_brief_markdown
    from app.db.repositories import gap_repo

    db = temp_db()
    task, contract, q = _make_task_with_contract(db)

    gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="State-change boundary is unmeasured.",
        target_setting="Agent memory", observed_problem="p",
        existing_coverage="c", missing_capability="m", claimed_delta="d",
        testable_hypothesis="h", falsification_condition="f",
        provenance_status="partial", question_ids=[q.id], mining_round=1,
        mining_policy_version="evidence-admission-v1",
    )
    db.commit()

    md = build_landscape_brief_markdown(
        db, task.id, contract.id,
        terminal_status="more_research_required",
        terminal_reason="no_surviving_gap_after_audit",
    )
    assert "候选 Gap 共 1 个" in md
    assert "B(摘要级)" in md  # partial provenance -> B tier
    db.close()

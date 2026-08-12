"""Failure handling in the opportunity pipeline.

From a real run (task 3286cf05): four search rounds, 540 evidence units and 154
papers were collected, then gap mining hit the model's context window and the
whole task was marked `failed` — discarding two hours of retrieval, evidence and
coverage for a recoverable input-size problem.
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


def _seed_contract(db, status="mining_gaps"):
    from app.db.models import ResearchContract, ResearchTask

    task = ResearchTask(user_input="long-horizon tool use", status=status)
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Long-horizon tool use",
                                status="active", version=1, input_hash="v1")
    db.add(contract)
    db.commit()
    return task, contract


@pytest.mark.asyncio
async def test_gap_mining_overflow_degrades_instead_of_failing_the_task(temp_db, monkeypatch):
    from app.agent import runner
    from app.agent.state import ResearchState
    from app.db.models import ResearchTask
    from app.db.repositories import task_repo
    from app.llm.base import LLMContextOverflow

    db = temp_db()
    task, contract = _seed_contract(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=4)
    task_repo.save_state(db, task.id, state)
    db.commit()

    async def _overflow(*args, **kwargs):
        raise LLMContextOverflow("prompt is ~41000 tokens but only 36864 are available")

    briefs = []

    async def _fake_brief(db, state, task_id, status, reason):
        briefs.append((status, reason))

    monkeypatch.setattr(runner, "mine_gap_candidates", _overflow)
    monkeypatch.setattr(runner, "generate_landscape_brief", _fake_brief)

    await runner._run_opportunity_pipeline(db, state, object(), task.id)

    refreshed = db.get(ResearchTask, task.id)
    assert refreshed.status == "more_research_required"
    assert refreshed.stop_reason == "gap_mining_prompt_too_large"
    # The run still reports what it found instead of vanishing.
    assert briefs == [("more_research_required", "gap_mining_prompt_too_large")]
    db.close()


@pytest.mark.asyncio
async def test_gap_mining_error_is_named_in_the_stop_reason(temp_db, monkeypatch):
    from app.agent import runner
    from app.agent.state import ResearchState
    from app.db.models import ResearchTask
    from app.db.repositories import task_repo

    db = temp_db()
    task, contract = _seed_contract(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id,
                          pipeline_version=2, current_round=2)
    task_repo.save_state(db, task.id, state)
    db.commit()

    async def _boom(*args, **kwargs):
        raise RuntimeError("LLM call failed: 503 - upstream unavailable")

    async def _fake_brief(db, state, task_id, status, reason):
        return None

    monkeypatch.setattr(runner, "mine_gap_candidates", _boom)
    monkeypatch.setattr(runner, "generate_landscape_brief", _fake_brief)

    await runner._run_opportunity_pipeline(db, state, object(), task.id)

    refreshed = db.get(ResearchTask, task.id)
    assert refreshed.status == "more_research_required"
    assert refreshed.stop_reason.startswith("gap_mining_failed: ")
    assert "503" in refreshed.stop_reason
    db.close()

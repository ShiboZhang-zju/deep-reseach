"""Regression tests: resuming a task must not invalidate its Research Contract.

Root cause covered here (observed in production on 2026-08-11): every state
reload used to copy the decomposed *research* questions into
`clarification_questions`. That silently changed the contract input version
after the contract had been built, so resuming a task superseded its own active
contract, created new questions, orphaned every coverage snapshot bound to the
old ones, and the readiness gate then failed with
`no_high_importance_question_has_coverage_snapshot`.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig()
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    cfg.set_main_option("script_location",
                        os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    alembic_command.upgrade(cfg, "head")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


def test_state_reload_does_not_turn_research_questions_into_clarification_questions():
    from app.agent.state import ResearchState

    state = ResearchState(user_input="topic", pipeline_version=2,
                          contract_id="contract-1", active_question_ids=["q1"])
    state.research_questions = ["Decomposed research question 1?",
                                "Decomposed research question 2?"]

    reloaded = ResearchState.from_json(state.to_json())

    assert reloaded.clarification_questions == []
    assert reloaded.research_questions == state.research_questions


def test_legacy_state_blob_still_migrates_research_questions():
    from app.agent.state import ResearchState

    legacy_blob = json.dumps({
        "user_input": "topic",
        "research_questions": ["Legacy question?"],
    })

    reloaded = ResearchState.from_json(legacy_blob)

    assert reloaded.clarification_questions == ["Legacy question?"]


class _ContractLLM:
    """Returns a fresh contract each call so contract rebuilds are detectable."""

    def __init__(self):
        self.calls = 0

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import ResearchContractSchema

        self.calls += 1
        return ResearchContractSchema(
            topic=f"Topic v{self.calls}", target_problem="p", target_setting="s",
            desired_output="method", novelty_bar="conference",
            preferred_directions=[], excluded_directions=[],
            gpu_available=False, max_gpu_hours=0, max_api_budget=0,
            max_runtime_minutes=60, allow_large_benchmark=False,
            allow_model_training=False, experiment_preferences={},
            key_terms=["memory"], time_scope_start=None, time_scope_end=None,
            confidence=0.8,
        )


@pytest.mark.asyncio
async def test_resume_after_decomposition_reuses_contract_and_questions(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.db.models import ResearchContract, ResearchQuestion, ResearchTask
    from app.db.repositories import task_repo

    db = temp_db()
    task = ResearchTask(user_input="agent memory under state changes", status="pending")
    task.state_json = ResearchState(user_input="agent memory under state changes").to_json()
    db.add(task)
    db.commit()

    llm = _ContractLLM()
    state = task_repo.get_state(db, task.id)
    contract_v1 = await build_research_contract(db, state, llm, task.id)

    question = ResearchQuestion(task_id=task.id, contract_id=contract_v1.id,
                               question="Does memory fail after state changes?",
                               question_type="failure", importance=0.9, status="open")
    db.add(question)
    db.flush()
    # What decompose_research_space persists after building questions.
    state = task_repo.get_state(db, task.id)
    state.active_question_ids = [question.id]
    state.research_questions = [question.question]
    task_repo.save_state(db, task.id, state)
    db.commit()

    # --- resume: same inputs, so the contract must be reused as-is ---
    resumed_state = task_repo.get_state(db, task.id)
    contract_after_resume = await build_research_contract(db, resumed_state, llm, task.id)

    assert contract_after_resume.id == contract_v1.id
    assert contract_after_resume.version == 1
    assert llm.calls == 1, "resume must not rebuild the contract"
    assert db.query(ResearchContract).filter(
        ResearchContract.task_id == task.id).count() == 1
    db.refresh(question)
    assert question.status == "open", "questions must survive a resume"
    assert resumed_state.active_question_ids == [question.id]
    db.close()


@pytest.mark.asyncio
async def test_contract_stored_with_legacy_hash_scheme_is_restamped_not_superseded(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import (
        _compute_legacy_input_version_v1,
        build_research_contract,
        compute_contract_input_version,
    )
    from app.db.models import ResearchContract, ResearchTask
    from app.db.repositories import task_repo

    db = temp_db()
    task = ResearchTask(user_input="agent memory", status="pending")
    task.state_json = ResearchState(user_input="agent memory").to_json()
    db.add(task)
    db.commit()

    state = task_repo.get_state(db, task.id)
    legacy_hash = _compute_legacy_input_version_v1(db, task, state)
    current_hash = compute_contract_input_version(db, task, state)
    assert legacy_hash != current_hash

    existing = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash=legacy_hash)
    db.add(existing)
    db.commit()

    llm = _ContractLLM()
    reused = await build_research_contract(db, task_repo.get_state(db, task.id), llm, task.id)

    assert reused.id == existing.id
    assert llm.calls == 0
    assert reused.input_hash == current_hash, "hash scheme should be re-stamped on reuse"
    assert db.query(ResearchContract).filter(
        ResearchContract.task_id == task.id).count() == 1
    db.close()

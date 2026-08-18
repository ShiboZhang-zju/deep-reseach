"""Tests for the phenomenon-validation plan step."""

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


class PhenomenonLLM:
    async def chat_json(self, messages, schema):
        from app.agent.steps.generate_phenomenon_plans import PhenomenonPlanSchema
        return PhenomenonPlanSchema(
            phenomenon="models pass sparse tests but fail dense tests on the same program",
            critical_unknown="whether sparse-pass/dense-fail is common enough to matter",
            oracle_experiment="measure sparse-pass vs dense-pass rate on existing self-correction benchmarks",
            kill_criterion="if fewer than 5% of sparse-pass samples fail dense tests, abandon the direction",
            measurement="sparse-pass -> dense-fail rate",
        )


def _seed_surviving_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import GapCandidate, ResearchContract, ResearchTask

    task = ResearchTask(user_input="code gen", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Code Gen", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="missing_evaluation",
        description="self-correction eval", target_setting="code gen",
        observed_problem="accidental correctness", existing_coverage="generic benchmarks",
        missing_capability="dense hidden tests", claimed_delta="sparse-to-dense eval",
        testable_hypothesis="sparse pass overestimates reliability",
        falsification_condition="dense tests show the same pass rate",
        status="surviving", mining_policy_version=GAP_MINING_POLICY_VERSION,
        nearest_prior_art_paper_id="p1", nearest_prior_art_title="Prior Art",
        residual_gap="no sparse->dense protocol", search_confidence="high",
    )
    db.add(gap)
    db.commit()
    return task, contract, gap


@pytest.mark.asyncio
async def test_surviving_gap_gets_a_phenomenon_plan(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.generate_phenomenon_plans import generate_phenomenon_plans
    from app.db.models import GapPhenomenonPlan

    db = temp_db()
    task, contract, gap = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    ids = await generate_phenomenon_plans(db, state, PhenomenonLLM(), task.id)

    assert len(ids) == 1
    plan = db.query(GapPhenomenonPlan).filter(GapPhenomenonPlan.gap_id == gap.id).one()
    assert plan.phenomenon
    assert plan.kill_criterion, "every plan must carry a quantitative kill criterion"
    assert plan.oracle_experiment
    db.close()


@pytest.mark.asyncio
async def test_phenomenon_plan_is_idempotent(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.generate_phenomenon_plans import generate_phenomenon_plans
    from app.db.models import GapPhenomenonPlan

    db = temp_db()
    task, contract, gap = _seed_surviving_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    await generate_phenomenon_plans(db, state, PhenomenonLLM(), task.id)
    # A second call must reuse the existing plan, not insert a duplicate.
    ids2 = await generate_phenomenon_plans(db, state, PhenomenonLLM(), task.id)

    assert len(ids2) == 1
    assert db.query(GapPhenomenonPlan).filter(
        GapPhenomenonPlan.gap_id == gap.id).count() == 1
    db.close()

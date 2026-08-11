"""Tests for narrowing gaps that the neighbour audit only partially closed."""

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


def _seed_audited_gap(db, *, audit_result="partially_closed",
                      remaining_delta="固定预算下状态变化边界的量化评估仍未被任何近邻覆盖。",
                      recommended_action="narrow", narrow_history=0):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import ResearchContract, ResearchTask
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="agent memory", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    gap = gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="状态变化边界未被评估", target_setting="固定预算 Agent Memory",
        observed_problem="状态变化导致证据丢失", existing_coverage="一般问答准确率",
        claimed_delta="固定预算下的状态变化边界评测", missing_capability="状态变化边界评测",
        testable_hypothesis="状态变化性能较低", falsification_condition="已有同设置边界评测",
        provenance_status="complete", question_ids=[], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    gap.status = "audited"
    for index in range(narrow_history):
        gap_repo.create_gap_audit(
            db, gap_id=gap.id, task_id=task.id, adversarial_queries=[],
            audit_result="partially_closed", recommended_action="narrow",
            remaining_delta=f"历史收窄 {index}", audit_round=index + 1)
    gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=task.id, adversarial_queries=[],
        audit_result=audit_result, recommended_action=recommended_action,
        remaining_delta=remaining_delta,
        nearest_neighbor_summary="近邻已覆盖静态问答下的记忆压缩评测。",
        audit_round=narrow_history + 1)
    db.commit()
    return task, contract, gap


def test_partially_closed_gap_is_narrowed_to_its_remaining_delta(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    narrowed = narrow_audited_gaps(db, state, task.id)

    db.refresh(gap)
    assert narrowed == [gap.id]
    assert gap.claimed_delta == "固定预算下状态变化边界的量化评估仍未被任何近邻覆盖。"
    assert "近邻已覆盖静态问答下的记忆压缩评测。" in gap.existing_coverage
    assert "一般问答准确率" in gap.existing_coverage, "原有覆盖不能被丢弃"
    assert gap.status == "auditing", "收窄后的 claim 必须重新经过审计"
    db.close()


def test_confirmed_or_rejected_gaps_are_not_narrowed(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, audit_result="confirmed",
                                            recommended_action="continue")
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert narrow_audited_gaps(db, state, task.id) == []
    db.refresh(gap)
    assert gap.claimed_delta == "固定预算下的状态变化边界评测"
    db.close()


def test_gap_without_a_concrete_remaining_delta_is_not_narrowed(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, remaining_delta="  ")
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert narrow_audited_gaps(db, state, task.id) == []
    db.refresh(gap)
    assert gap.status == "audited", "没有可收窄的内容时不得凭空发明 claim"
    db.close()


def test_narrowing_is_capped(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import MAX_NARROW_ATTEMPTS, narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, narrow_history=MAX_NARROW_ATTEMPTS)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert narrow_audited_gaps(db, state, task.id) == []
    db.refresh(gap)
    assert gap.status == "audited"
    db.close()

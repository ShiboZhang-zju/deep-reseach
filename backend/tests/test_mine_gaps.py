"""Tests for lightweight evidence-backed gap mining."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


class FakeGapLLM:
    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        question_id = next(line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Question ID:" in line)
        evidence_id = next(line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Evidence ID:" in line)
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type="boundary_gap",
            description="现有固定预算记忆方法在状态变化场景下缺少明确边界评测。",
            target_setting="固定 token 预算的 Agent Memory",
            observed_problem="状态变化发生后准确率下降，现有证据没有报告该边界。",
            existing_coverage="已有工作报告一般问答准确率和压缩结果。",
            missing_capability="对状态变化与冲突查询的边界评测。",
            claimed_delta="明确评估固定预算下状态变化证据保留能力。",
            testable_hypothesis="状态变化场景的性能低于稳定事实场景。",
            falsification_condition="近邻论文已在相同预算和场景下完成该评测。",
            question_ids=[question_id],
            supporting_evidence_ids=[evidence_id],
        )])


@pytest.mark.asyncio
async def test_mine_gap_creates_traceable_candidate(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, ResearchContract, ResearchQuestion, ResearchTask
    from app.db.repositories import gap_repo

    db = temp_db()
    task = ResearchTask(user_input="agent memory", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active", version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id,
        contract_id=contract.id,
        question="固定预算下状态变化会如何影响记忆系统？",
        question_type="failure",
        importance=0.9,
        searchability=0.8,
        status="partially_covered",
    )
    db.add(question)
    db.flush()
    evidence = EvidenceUnit(
        task_id=task.id,
        paper_id="paper-1",
        evidence_type="limitation",
        normalized_claim="在状态变化后，固定预算记忆会遗漏关键历史证据。",
        verification_status="abstract_only",
        extraction_confidence=0.8,
    )
    db.add(evidence)
    db.commit()

    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    gaps = await mine_gap_candidates(db, state, FakeGapLLM(), task.id)

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.status == "candidate"
    assert gap.gap_type == "boundary_gap"
    assert gap.provenance_status == "complete"
    assert state.active_gap_ids == [gap.id]
    links = gap_repo.list_gap_evidence(db, gap.id)
    assert len(links) == 1
    assert links[0].evidence_id == evidence.id
    db.close()


@pytest.mark.asyncio
async def test_mine_gap_rejects_hallucinated_ids(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, ResearchContract, ResearchQuestion, ResearchTask
    from app.schemas.schemas import GapCandidateList, GapCandidateSchema

    class BadLLM:
        async def chat_json(self, messages, schema):
            return GapCandidateList(gaps=[GapCandidateSchema(
                gap_type="missing_evaluation",
                description="无效 ID 不得被写入缺口控制面。",
                target_setting="test",
                observed_problem="test",
                existing_coverage="test",
                missing_capability="test",
                claimed_delta="test",
                testable_hypothesis="test",
                falsification_condition="test",
                question_ids=["unknown-question"],
                supporting_evidence_ids=["unknown-evidence"],
            )])

    db = temp_db()
    task = ResearchTask(user_input="test", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="test", status="active", version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    db.add(ResearchQuestion(task_id=task.id, contract_id=contract.id, question="test question", question_type="evaluation"))
    db.add(EvidenceUnit(task_id=task.id, paper_id="paper-1", evidence_type="metric", normalized_claim="test metric"))
    db.commit()

    gaps = await mine_gap_candidates(db, ResearchState(task_id=task.id, contract_id=contract.id), BadLLM(), task.id)
    assert gaps == []
    db.close()

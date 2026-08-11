"""Tests for the lightweight adversarial gap audit."""

import ast
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
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


class ConfirmingAuditLLM:
    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapAuditDecisionSchema, NeighborAuditSchema

        text = messages[1]["content"]
        evidence_id = ast.literal_eval(next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Supporting evidence IDs:")))[0]
        paper_id = next(line.split(": ", 1)[1] for line in text.splitlines() if "Paper ID:" in line)
        return GapAuditDecisionSchema(
            audit_result="confirmed",
            recommended_action="continue",
            remaining_delta="近邻论文未评估固定预算下的状态变化边界。",
            nearest_neighbor_summary="近邻只报告通用问答性能。",
            differentiation_summary="候选 Gap 关注状态变化边界评测。",
            evidence_for_gap_ids=[evidence_id],
            novelty_confidence=0.8,
            audit_confidence=0.8,
            comparisons=[NeighborAuditSchema(
                paper_id=paper_id,
                similarity_score=0.7,
                shared_problem="都研究 Agent Memory。",
                shared_mechanism="都使用记忆压缩。",
                shared_evaluation="都报告问答准确率。",
                covered_claims=["一般准确率评估"],
                uncovered_claims=["状态变化边界评测"],
                overlap_ratio=0.4,
                overlap_risk=0.3,
            )],
        )


class MistypedEvidenceIdAuditLLM:
    """Returns a sound verdict but corrupts one evidence UUID while copying it."""

    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapAuditDecisionSchema

        return GapAuditDecisionSchema(
            audit_result="partially_closed",
            recommended_action="narrow",
            remaining_delta="近邻覆盖了长程评测，但没有定义固定预算下的状态变化边界指标。",
            nearest_neighbor_summary="近邻覆盖长程评测与多轴指标。",
            evidence_for_gap_ids=["e3024371-a441-44a2-b4-a40c-f879938efa05"],
            novelty_confidence=0.7,
            audit_confidence=0.7,
        )


class MalformedAuditLLM:
    """Returns a decision whose recommended_action is outside the contract."""

    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapAuditDecisionSchema

        return GapAuditDecisionSchema(
            audit_result="confirmed",
            recommended_action="proceed_to_experiment",
            novelty_confidence=0.8,
            audit_confidence=0.8,
        )


class UncertainAuditLLM:
    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapAuditDecisionSchema

        return GapAuditDecisionSchema(
            audit_result="uncertain",
            recommended_action="more_search",
            novelty_confidence=0.3,
            audit_confidence=0.3,
        )


def _seed_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import EvidenceUnit, GapCandidate, Paper, ResearchContract, ResearchTask, TaskPaper
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="agent memory", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active", version=1, input_hash="v1")
    paper = Paper(title="Neighbor", abstract="Memory compression evaluates generic question answering.", citation_count=10)
    db.add_all([contract, paper])
    db.flush()
    db.add(TaskPaper(task_id=task.id, paper_id=paper.id, discovered_round=1, final_score=0.9, priority="high"))
    evidence = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation", normalized_claim="No state change evaluation")
    db.add(evidence)
    db.flush()
    gap = GapCandidate(
        task_id=task.id,
        contract_id=contract.id,
        gap_type="boundary_gap",
        description="状态变化边界未被评估",
        target_setting="固定预算 Agent Memory",
        observed_problem="状态变化导致证据丢失",
        existing_coverage="一般问答准确率",
        missing_capability="状态变化边界评测",
        claimed_delta="固定预算下状态变化边界",
        testable_hypothesis="状态变化性能较低",
        falsification_condition="已有同设置边界评测",
        status="candidate",
        # The audit only considers gaps mined under the *current* policy, so this
        # must track the constant rather than pin an outdated literal.
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.flush()
    gap_repo.create_gap_evidence_link(db, gap.id, evidence.id, "suggests", 0.9)
    db.commit()
    return task, gap, evidence


@pytest.mark.asyncio
async def test_confirmed_audit_creates_comparison_and_survives(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, ConfirmingAuditLLM(), task.id, perform_search=True)

    assert results[0].audit_result == "confirmed"
    assert state.surviving_gap_ids == [gap.id]
    assert gap_repo.get_gap(db, gap.id).status == "surviving"
    assert len(gap_repo.list_neighbor_comparisons(db, gap.id)) == 1
    assert gap_repo.list_gap_audits(db, gap.id)[0].recommended_action == "continue"
    db.close()


@pytest.mark.asyncio
async def test_mistyped_evidence_id_does_not_discard_a_sound_verdict(temp_db):
    """A miscopied evidence UUID is an annotation defect, not a bad verdict.

    Production case (task 3286cf05): the model wrote
    ...-44a2-b4-... instead of ...-42b4-..., and a complete partially_closed
    verdict with a concrete remaining delta was thrown away as uncertain.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.models import AgentTrace
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)

    results = await audit_gap_candidates(db, state, MistypedEvidenceIdAuditLLM(), task.id,
                                         perform_search=True)

    assert [item.audit_result for item in results] == ["partially_closed"]
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert audit.recommended_action == "narrow"
    assert audit.remaining_delta, "the usable part of the verdict must be kept"
    assert not (audit.rejection_reason or ""), "this is not a contract violation"
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_audit_dropped_evidence_ids").one()
    assert "e3024371-a441-44a2-b4-a40c-f879938efa05" in trace.output_json
    db.close()


@pytest.mark.asyncio
async def test_malformed_audit_decision_degrades_one_gap_instead_of_failing_the_run(temp_db):
    """A decision outside the audit contract must not abort the whole task.

    Production case (task 5e040ad5): the first run that ever produced gap
    candidates died with ValueError("Invalid recommended action"), discarding
    259 evidence units and every downstream artefact.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.models import AgentTrace
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)

    results = await audit_gap_candidates(db, state, MalformedAuditLLM(), task.id,
                                         perform_search=True)

    assert [item.audit_result for item in results] == ["uncertain"]
    assert state.surviving_gap_ids == []
    assert gap_repo.get_gap(db, gap.id).status == "auditing"
    audit = gap_repo.list_gap_audits(db, gap.id)[0]
    assert audit.recommended_action == "more_search"
    assert "proceed_to_experiment" in (audit.rejection_reason or ""), (
        "the offending value must be recorded for diagnosis"
    )
    assert db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_audit_invalid_decision").count() == 1
    db.close()


@pytest.mark.asyncio
async def test_uncertain_audit_never_survives(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    await audit_gap_candidates(db, state, UncertainAuditLLM(), task.id, perform_search=False)

    assert state.surviving_gap_ids == []
    assert gap_repo.get_gap(db, gap.id).status == "auditing"
    assert gap_repo.list_gap_audits(db, gap.id)[0].recommended_action == "more_search"
    db.close()

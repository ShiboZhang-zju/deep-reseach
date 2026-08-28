"""Tests for narrowing gaps that the neighbour audit only partially closed."""

import asyncio
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _run(coro):
    return asyncio.run(coro)


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
    from app.db.models import GapCandidate

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    narrowed = _run(narrow_audited_gaps(db, state, task.id))

    # Narrowing versions the gap instead of overwriting it: the parent keeps its
    # original claim and is retired, while a child carries the narrowed claim.
    assert len(narrowed) == 1
    child_id = narrowed[0]
    assert child_id != gap.id
    child = db.get(GapCandidate, child_id)
    assert child.claimed_delta == "固定预算下状态变化边界的量化评估仍未被任何近邻覆盖。"
    assert "近邻已覆盖静态问答下的记忆压缩评测。" in child.existing_coverage
    assert "一般问答准确率" in child.existing_coverage, "原有覆盖不能被丢弃"
    assert child.status == "auditing", "收窄后的 claim 必须重新经过审计"
    assert child.version == 2
    assert child.parent_gap_id == gap.id
    assert child.canonical_gap_id == gap.id
    db.refresh(gap)
    assert gap.status == "superseded", "收窄后父版本必须退役，不能作为并列 Gap 出现"
    assert gap.claimed_delta == "固定预算下的状态变化边界评测", "父版本原始 claim 不能被覆盖"
    db.close()


def test_narrowed_gap_surfaces_as_a_single_canonical_head(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps
    from app.db.repositories import gap_repo

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    child_id = _run(narrow_audited_gaps(db, state, task.id))[0]

    # The report must show exactly one row per canonical gap (its latest head),
    # never the superseded v1 next to its v2.
    heads = gap_repo.list_canonical_gap_heads(db, task.id, contract.id)
    assert [h.id for h in heads] == [child_id]
    db.close()


def test_confirmed_or_rejected_gaps_are_not_narrowed(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, audit_result="confirmed",
                                            recommended_action="continue")
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert _run(narrow_audited_gaps(db, state, task.id)) == []
    db.refresh(gap)
    assert gap.claimed_delta == "固定预算下的状态变化边界评测"
    db.close()


def test_gap_without_a_concrete_remaining_delta_is_not_narrowed(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, remaining_delta="  ")
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert _run(narrow_audited_gaps(db, state, task.id)) == []
    db.refresh(gap)
    assert gap.status == "audited", "没有可收窄的内容时不得凭空发明 claim"
    db.close()


def test_re_auditing_a_gap_updates_the_neighbour_comparison_in_place(temp_db):
    """A second audit of the same gap/neighbour pair must not violate uniqueness.

    Narrowing makes re-audits actually happen; before this, the second audit of
    any gap crashed the whole run with an IntegrityError on
    neighbor_comparisons(gap_id, paper_id).
    """
    from app.db.models import Paper
    from app.db.repositories import gap_repo

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db)
    paper = Paper(title="Neighbour", abstract="a")
    db.add(paper)
    db.flush()

    gap_repo.create_neighbor_comparison(
        db, gap_id=gap.id, paper_id=paper.id, task_id=task.id,
        similarity_score=0.4, overlap_ratio=0.3, covered_claims=["generic QA"])
    gap_repo.create_neighbor_comparison(
        db, gap_id=gap.id, paper_id=paper.id, task_id=task.id,
        similarity_score=0.8, overlap_ratio=0.6, covered_claims=["state change"])
    db.commit()

    comparisons = gap_repo.list_neighbor_comparisons(db, gap.id)
    assert len(comparisons) == 1
    assert comparisons[0].similarity_score == 0.8
    assert "state change" in comparisons[0].covered_claims_json
    db.close()


def test_narrowing_is_capped(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import MAX_NARROW_ATTEMPTS, narrow_audited_gaps

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, narrow_history=MAX_NARROW_ATTEMPTS)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    assert _run(narrow_audited_gaps(db, state, task.id)) == []
    db.refresh(gap)
    assert gap.status == "audited"
    db.close()


# --- P0-2 (task d6f64087): hedged remaining_delta must not become claimed_delta ---

_HEDGE_DELTA = ("The audit cannot confirm novelty because the provided neighbor "
                "papers do not explicitly address the intersection of 'noisy labels' "
                "and 'small-scale (<3B) reward models' with 'real-time online "
                "evaluation'. The evidence is insufficient to rule out prior work "
                "in this specific niche.")


class _DistillingLLM:
    """Distills the hedge into a positive delta claim."""

    async def chat(self, messages, temperature=0.7):
        return ("The intersection of noisy labels, small-scale (<3B) reward models "
                "and real-time online evaluation remains unverified in prior work.")


class _EchoingLLM:
    """Echoes the hedge back — distillation must be rejected, raw text kept."""

    async def chat(self, messages, temperature=0.7):
        return "The audit cannot confirm novelty for this niche."


def test_hedged_remaining_delta_is_distilled_into_positive_claim(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps
    from app.db.models import AgentTrace, GapCandidate

    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, remaining_delta=_HEDGE_DELTA)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    narrowed = _run(narrow_audited_gaps(db, state, task.id, llm=_DistillingLLM()))

    assert len(narrowed) == 1
    child = db.get(GapCandidate, narrowed[0])
    assert "cannot confirm" not in child.claimed_delta
    assert "remains unverified" in child.claimed_delta
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "narrow_gaps").one()
    payload = json.loads(trace.output_json)
    assert payload["hedge_distillations"][0]["distilled"] is True
    db.close()


def test_hedged_remaining_delta_degrades_to_raw_text_when_distillation_unavailable(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps
    from app.db.models import AgentTrace, GapCandidate

    # No LLM at all: degradation keeps the raw hedge (a polluted claim beats a
    # dropped gap) and records the fallback in the trace.
    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, remaining_delta=_HEDGE_DELTA)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    narrowed = _run(narrow_audited_gaps(db, state, task.id))
    assert len(narrowed) == 1
    child = db.get(GapCandidate, narrowed[0])
    assert child.claimed_delta == _HEDGE_DELTA
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "narrow_gaps").one()
    payload = json.loads(trace.output_json)
    assert payload["hedge_distillations"][0]["distilled"] is False
    assert payload["hedge_distillations"][0]["fallback"] == "raw_hedge_kept"
    db.close()


def test_hedged_remaining_delta_degrades_when_distillation_echoes_hedge(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.narrow_gaps import narrow_audited_gaps
    from app.db.models import GapCandidate

    # An LLM that echoes the hedge back gains nothing: the distilled claim is
    # rejected and the raw hedge text is kept.
    db = temp_db()
    task, contract, gap = _seed_audited_gap(db, remaining_delta=_HEDGE_DELTA)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    narrowed = _run(narrow_audited_gaps(db, state, task.id, llm=_EchoingLLM()))
    assert len(narrowed) == 1
    child = db.get(GapCandidate, narrowed[0])
    assert child.claimed_delta == _HEDGE_DELTA
    db.close()

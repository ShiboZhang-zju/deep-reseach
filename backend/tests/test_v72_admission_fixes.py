"""v7.2 admission fixes (pilot adjudication 2026-10-29).

Two admission-tightening rules driven by the blind-review false-open breakdown
of the v7 replay pilot:

1. boundary_gap rejects PERFORMANCE_TARGET_DELTA — both human-true pilot
   survivors were "achieves >X% ..." outcome targets, not boundary conditions
   (reviewer flag: target_only, "NONE does not mean a credible gap").
2. scaling_regime admission gains a claim-level precedent retrieval
   (SCALING_RANGE_ALREADY_MEASURED) — 4/7 pilot scaling_regime opportunities
   were false-open; the claimed "untested range" was already measured by work
   outside the evidence pool. Infra failures fail open (degraded -> accept).
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
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
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


# === Pure helper tests ======================================================

def test_performance_target_delta_detects_pilot_examples():
    from app.agent.steps.mine_gaps import _is_performance_target_delta

    # The two human-true pilot survivors (both reviewer-flagged target_only).
    assert _is_performance_target_delta(
        "Achieves >10% higher accuracy than truncated gradient methods "
        "(<=4 steps) while maintaining <5% latency overhead compared to "
        "zero-shot baselines.")
    assert _is_performance_target_delta(
        "Improves the ability to distinguish expert-specific semantic "
        "contributions by >25% in imbalanced settings compared to standard "
        "activation patching.")


def test_performance_target_delta_spares_boundary_statements():
    from app.agent.steps.mine_gaps import _is_performance_target_delta

    # A boundary condition phrased around a measurement.
    assert not _is_performance_target_delta(
        "Methods degrade when label noise exceeds 10% in fixed-budget memory.")
    # Untested-setting phrasing.
    assert not _is_performance_target_delta(
        "No prior work evaluates state-change retention beyond 32k token budgets.")
    assert not _is_performance_target_delta(
        "Retrieval augmentation remains untested below 1B parameters.")
    # Boundary language exempts even a performance-verb lead.
    assert not _is_performance_target_delta(
        "Improves nothing: the method fails to generalize and its evaluation "
        "was never reported under distribution shift.")
    # The pre-v7.2 test-fixture shape (Chinese) is untouched.
    assert not _is_performance_target_delta("明确评估固定预算下状态变化证据保留能力。")
    # Empty / whitespace safe.
    assert not _is_performance_target_delta("")
    assert not _is_performance_target_delta("   ")


def test_policy_version_bumped_to_v72():
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    assert GAP_MINING_POLICY_VERSION == "opportunity-ontology-v7.2"


def test_boundary_gap_prompt_discloses_target_rejection():
    """Prompt-side guard: the boundary_gap directive names the rejection."""
    from app.agent.steps.mine_gaps import _CONTRACTS_BY_TYPE

    directive = _CONTRACTS_BY_TYPE["boundary_gap"].discovery_directive
    assert "PERFORMANCE_TARGET_DELTA" in directive
    assert "outcome target" in directive


# === Admission-chain integration ===========================================

def _seed_pool(Session):
    """Task + contract + question + two papers + admitted evidence."""
    from app.db.models import (
        EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract,
        ResearchQuestion, ResearchTask,
    )

    db = Session()
    task = ResearchTask(user_input="retrieval augmentation", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(
        task_id=task.id, topic="Retrieval Augmentation", status="active",
        version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="How does model scale affect retrieval augmentation?",
        question_type="failure", importance=0.9, searchability=0.8,
        status="partially_covered",
    )
    db.add(question)
    db.flush()
    first_paper = Paper(title="Evidence Paper A")
    second_paper = Paper(title="Evidence Paper B")
    db.add_all([first_paper, second_paper])
    db.flush()
    first_evidence = EvidenceUnit(
        task_id=task.id, paper_id=first_paper.id, evidence_type="limitation",
        normalized_claim="Evaluation only covers models above 1B parameters.",
        original_span="only covers models above 1B", section="Limitations",
        page_number=8, span_start=10, span_end=42, source_chunk_hash="fulltext-a",
        verification_status="verified", extraction_confidence=0.8,
    )
    second_evidence = EvidenceUnit(
        task_id=task.id, paper_id=second_paper.id, evidence_type="comparison",
        normalized_claim="Sub-1B behaviour is not reported.",
        verification_status="abstract_only",
    )
    db.add_all([first_evidence, second_evidence])
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=first_evidence.id,
                             relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id,
                             relation_type="supports", relevance_score=0.9),
    ])
    db.commit()
    return db, task, contract, question, first_evidence, second_evidence


def _parse_prompt_ids(messages):
    user_content = messages[1]["content"]
    question_id = next(line.split(": ", 1)[1] for line in user_content.splitlines()
                       if "Question ID:" in line)
    evidence_ids = [line.split(": ", 1)[1] for line in user_content.splitlines()
                    if "Evidence ID:" in line]
    return question_id, evidence_ids


class _LensFakeLLM:
    """Returns one candidate of `lens_type` for that lens, nothing elsewhere."""

    def __init__(self, lens_type, claimed_delta):
        self.lens_type = lens_type
        self.claimed_delta = claimed_delta

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        if schema is not GapCandidateList:
            raise AssertionError(f"unexpected schema in lens call: {schema}")
        if f"Opportunity type for this call: {self.lens_type}" not in messages[0]["content"]:
            return GapCandidateList(gaps=[])
        question_id, evidence_ids = _parse_prompt_ids(messages)
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type=self.lens_type,
            description="Sub-1B retrieval augmentation behaviour is undocumented.",
            target_setting="Retrieval augmentation below 1B parameters",
            observed_problem="Evaluations only cover models above 1B parameters.",
            existing_coverage="Results reported for 1B-70B models.",
            missing_capability="Measurements in the sub-1B regime.",
            claimed_delta=self.claimed_delta,
            testable_hypothesis="Retrieval augmentation changes sign below 1B parameters.",
            falsification_condition="A paper reports sub-1B retrieval augmentation results.",
            question_ids=[question_id],
            supporting_evidence_ids=evidence_ids,
        )])


def _rejected_reasons(db, task_id):
    from app.db.models import AgentTrace

    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id,
        AgentTrace.step_name == "gap_mining_candidates",
    ).order_by(AgentTrace.created_at.desc()).first()
    payload = json.loads(trace.output_json)
    return [r["reason"] for r in payload.get("rejected_candidates", [])]


@pytest.mark.asyncio
async def test_boundary_gap_performance_target_rejected(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = _LensFakeLLM(
        "boundary_gap",
        "Achieves >10% higher accuracy than fixed-budget baselines while "
        "maintaining <5% latency overhead.",
    )
    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert gaps == []
    reasons = _rejected_reasons(db, task.id)
    assert "PERFORMANCE_TARGET_DELTA" in reasons
    db.close()


@pytest.mark.asyncio
async def test_boundary_gap_boundary_statement_still_accepted(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    llm = _LensFakeLLM(
        "boundary_gap",
        "Fixed-budget memory fails to retain state-change evidence; no prior "
        "work evaluates this boundary below 1B parameters.",
    )
    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1
    assert gaps[0].gap_type == "boundary_gap"
    assert "PERFORMANCE_TARGET_DELTA" not in _rejected_reasons(db, task.id)
    db.close()


@pytest.mark.asyncio
async def test_scaling_precedent_covered_rejects_candidate(temp_db, monkeypatch):
    """A retrieved paper measuring in the claimed range rejects the candidate."""
    from app.agent.state import ResearchState
    from app.agent.steps import mine_gaps
    from app.agent.steps.mine_gaps import (
        ScalingPrecedentQueriesSchema, ScalingPrecedentVerdictSchema,
        mine_gap_candidates,
    )
    from app.db.models import Paper, SearchQueryPaper

    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    holder = {}

    async def fake_search(db_, state_, executions, task_id_, round_num):
        paper = Paper(
            title="Sub-1B Retrieval Augmentation Study",
            abstract="We evaluate retrieval augmentation from 350M to 3B "
                     "parameters and report accuracy across the range.",
        )
        db_.add(paper)
        db_.flush()
        holder["paper_id"] = paper.id
        for execution in executions:
            db_.add(SearchQueryPaper(query_id=execution.query_id,
                                     paper_id=paper.id, rank=1))
        db_.commit()
        return (1, [paper.id], [paper.id])

    monkeypatch.setattr("app.agent.steps.search_papers.search_and_save_papers",
                        fake_search)

    class ScalingLLM(_LensFakeLLM):
        async def chat_json(self, messages, schema):
            if schema is ScalingPrecedentQueriesSchema:
                return ScalingPrecedentQueriesSchema(
                    queries=["sub-1B parameter retrieval augmentation"])
            if schema is ScalingPrecedentVerdictSchema:
                return ScalingPrecedentVerdictSchema(
                    range_already_measured=True,
                    evidence_paper_id=holder.get("paper_id"),
                )
            return await super().chat_json(messages, schema)

    llm = ScalingLLM(
        "scaling_regime",
        "Retrieval augmentation behaviour below 1B parameters is untested.",
    )
    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert gaps == []
    reasons = _rejected_reasons(db, task.id)
    assert "SCALING_RANGE_ALREADY_MEASURED" in reasons
    # The precedent check left a trace with the covering paper.
    from app.db.models import AgentTrace
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "scaling_precedent_check",
    ).first()
    assert trace is not None
    payload = json.loads(trace.output_json)
    assert payload["verdict"] == "covered"
    assert payload["covering_paper_id"] == holder["paper_id"]
    db.close()


@pytest.mark.asyncio
async def test_scaling_precedent_infra_failure_fails_open(temp_db, monkeypatch):
    """A retrieval outage is not evidence about the claim — candidate accepted."""
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import (
        ScalingPrecedentQueriesSchema, mine_gap_candidates,
    )

    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    async def broken_search(db_, state_, executions, task_id_, round_num):
        raise RuntimeError("all sources 429")

    monkeypatch.setattr("app.agent.steps.search_papers.search_and_save_papers",
                        broken_search)

    class DegradedLLM(_LensFakeLLM):
        async def chat_json(self, messages, schema):
            if schema is ScalingPrecedentQueriesSchema:
                return ScalingPrecedentQueriesSchema(
                    queries=["sub-1B retrieval augmentation"])
            return await super().chat_json(messages, schema)

    llm = DegradedLLM(
        "scaling_regime",
        "Retrieval augmentation behaviour below 1B parameters is untested.",
    )
    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1
    assert gaps[0].gap_type == "scaling_regime"
    from app.db.models import AgentTrace
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "scaling_precedent_check",
    ).first()
    assert json.loads(trace.output_json)["verdict"] == "degraded"
    db.close()


@pytest.mark.asyncio
async def test_scaling_precedent_disabled_by_settings(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import (
        ScalingPrecedentQueriesSchema, mine_gap_candidates,
    )
    from app.config import settings

    db, task, contract, question, ev_a, ev_b = _seed_pool(temp_db)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    monkeypatch.setattr(settings, "gap_mining_precedent_check_enabled", False)

    class NoPrecedentLLM(_LensFakeLLM):
        async def chat_json(self, messages, schema):
            assert schema is not ScalingPrecedentQueriesSchema, (
                "precedent check must not run when disabled")
            return await super().chat_json(messages, schema)

    llm = NoPrecedentLLM(
        "scaling_regime",
        "Retrieval augmentation behaviour below 1B parameters is untested.",
    )
    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1
    db.close()

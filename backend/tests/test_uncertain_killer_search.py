"""P1 (probe df744606, 2026-10-30): uncertain verdicts get a killer-directed search.

The probe's two uncertain audits both recorded "the direct-hit prior work was
not retrieved" while their own killer descriptions named exactly what to look
for — and each bought a ~20-minute O2 remediation round that ended uncertain
again. Now an uncertain/more_search verdict runs ONE killer-directed search
before it may enter the remediation loop: a hit closes the gap
(UNCERTAIN_KILLER_FOUND, closed/reject), a miss leaves the verdict untouched.
"""

import ast
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
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


def _seed_gap(db):
    # Same fixture shape as tests/test_audit_gaps.py (kept local so this file
    # stays self-contained).
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import EvidenceUnit, GapCandidate, Paper, ResearchContract, ResearchTask, TaskPaper

    task = ResearchTask(user_input="agent memory", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1")
    paper = Paper(title="Neighbor", abstract="Memory compression evaluates generic QA.",
                  citation_count=10)
    db.add_all([contract, paper])
    db.flush()
    db.add(TaskPaper(task_id=task.id, paper_id=paper.id, discovered_round=1,
                     final_score=0.9, priority="high"))
    evidence = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation",
                            normalized_claim="No state change evaluation")
    db.add(evidence)
    db.flush()
    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="状态变化边界未被评估", target_setting="固定预算 Agent Memory",
        observed_problem="状态变化导致证据丢失", existing_coverage="一般问答准确率",
        missing_capability="状态变化边界评测", claimed_delta="固定预算下状态变化边界",
        testable_hypothesis="状态变化性能较低", falsification_condition="已有同设置边界评测",
        status="candidate", mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.commit()
    return task, gap


def _pin_admission_and_neighbors(monkeypatch, db, gap):
    from app.agent.steps import audit_gaps as module
    from app.db.models import EvidenceUnit, Paper, TaskPaper

    neighbor = Paper(title="Neighbor B", abstract="Evaluates generic long-context recall.",
                     citation_count=20)
    db.add(neighbor)
    db.flush()
    db.add(TaskPaper(task_id=gap.task_id, paper_id=neighbor.id, discovered_round=1,
                     final_score=0.8, priority="high"))
    db.add(EvidenceUnit(task_id=gap.task_id, paper_id=neighbor.id,
                        evidence_type="limitation",
                        normalized_claim="No boundary evaluation under fixed budgets",
                        verification_status="verified"))
    db.commit()

    admission = module.GapSearchAdmission(
        gap.id, "PASS", [], ["q1"], ["q1"], [], ["overlap"], 1, [neighbor.id], [neighbor.id])
    monkeypatch.setattr(module, "evaluate_gap_search_admission",
                        lambda db, gap, query_ids: admission)
    monkeypatch.setattr(module, "select_gap_specific_neighbors",
                        lambda db, gap, query_ids, limit=5, **kwargs: [neighbor])
    return neighbor


class UncertainKillerLLM:
    """Uncertain verdict that names its own killer (the probe's exact shape:
    'the direct-hit prior work was not retrieved' + a killer description)."""

    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import GapAuditDecisionSchema

        if schema.__name__ != "GapAuditDecisionSchema":
            raise AssertionError(f"unexpected schema: {schema.__name__}")
        return GapAuditDecisionSchema(
            audit_result="uncertain",
            recommended_action="more_search",
            novelty_confidence=0.3,
            audit_confidence=0.3,
            closest_killer_work=(
                "A benchmark study directly comparing a judge's tolerance of "
                "its own vs another model's factually identical hallucinations"),
            killer_query_terms=["judge self hallucination tolerance"],
            killer_found=False,
            residual_uncertainty="the direct-hit prior work was not retrieved",
        )


@pytest.mark.asyncio
async def test_uncertain_killer_hit_closes_gap(temp_db, monkeypatch):
    """A recalled strongly-matching killer closes the uncertain gap instead of
    sending it into the O2 remediation loop."""
    from app.agent.state import ResearchState
    from app.agent.steps import audit_gaps as module
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.models import Paper
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap = _seed_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap)

    killer_paper = Paper(
        title="Judges tolerate their own identical hallucinations",
        abstract="A benchmark comparing a judge's tolerance of its own vs "
                 "another model's factually identical hallucinations.",
        citation_count=5)
    db.add(killer_paper)
    db.commit()

    async def fake_search(db_, state_, executions, task_id_, round_):
        return 1, 1, [killer_paper.id]

    monkeypatch.setattr(module, "search_and_save_papers", fake_search)
    monkeypatch.setattr(module.settings, "audit_score_retrieved_papers", False)
    import app.services.embedding_service as emb
    monkeypatch.setattr(emb, "embed_texts", lambda texts: [[1.0, 0.0]] * len(texts))
    monkeypatch.setattr(emb, "cosine_similarity", lambda a, b: 1.0)

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, UncertainKillerLLM(), task.id,
                                         perform_search=True)

    assert results[0].audit_result == "closed"
    assert results[0].recommended_action == "reject"
    assert gap_repo.get_gap(db, gap.id).status == "rejected"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    codes = json.loads(audit.failure_reason_codes_json)
    assert "UNCERTAIN_KILLER_FOUND" in codes
    db.close()


@pytest.mark.asyncio
async def test_uncertain_killer_miss_keeps_more_search(temp_db, monkeypatch):
    """No recalled match: the verdict stays uncertain/more_search — the search
    is a resolution attempt, not a new rejection source."""
    from app.agent.state import ResearchState
    from app.agent.steps import audit_gaps as module
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap = _seed_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap)

    async def empty_search(db_, state_, executions, task_id_, round_):
        return 0, 0, []

    monkeypatch.setattr(module, "search_and_save_papers", empty_search)
    monkeypatch.setattr(module.settings, "audit_score_retrieved_papers", False)

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, UncertainKillerLLM(), task.id,
                                         perform_search=True)

    assert results[0].audit_result == "uncertain"
    assert results[0].recommended_action == "more_search"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    codes = json.loads(audit.failure_reason_codes_json)
    assert "UNCERTAIN_KILLER_FOUND" not in codes

    # A miss must still be observable: without this record a run cannot tell
    # "guard skipped the search" from "search ran and found nothing".
    from app.db.models import AgentTrace
    trace = db.query(AgentTrace).filter(
        AgentTrace.task_id == task.id,
        AgentTrace.step_name == "gap_audit_uncertain_killer_search",
    ).first()
    assert trace is not None
    payload = json.loads(trace.output_json)
    assert payload["ran"] is True
    assert payload["skip_reason"] is None
    assert payload["hit_count"] == 0
    db.close()


@pytest.mark.asyncio
async def test_uncertain_killer_skipped_when_search_disabled(temp_db, monkeypatch):
    """perform_search=False (replay/constrained mode): no killer search, the
    uncertain verdict is returned untouched."""
    from app.agent.state import ResearchState
    from app.agent.steps import audit_gaps as module
    from app.agent.steps.audit_gaps import audit_gap_candidates

    db = temp_db()
    task, gap = _seed_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap)

    async def must_not_search(db_, state_, executions, task_id_, round_):
        raise AssertionError("search must not run with perform_search=False")

    monkeypatch.setattr(module, "search_and_save_papers", must_not_search)

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, UncertainKillerLLM(), task.id,
                                         perform_search=False)

    assert results[0].audit_result == "uncertain"
    assert results[0].recommended_action == "more_search"
    db.close()

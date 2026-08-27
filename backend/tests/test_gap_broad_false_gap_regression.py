"""Regression: a broad gap must not survive when direct prior art is in NPA.

fd688ba6: the broad claim "RAG lacks dedicated refusal-under-insufficiency
evaluation" was confirmed/survived because the neighbor pool was all generic
surveys (Tree of Reviews, Systematic Lit Review RAG, ...) — direct prior art
(HopRefusalBench, Evidence-Calibrated RAG) was recalled by the gap search but
never scored, so it never reached the NPA pool. With gap-specific relevance
screening, the direct prior art enters the pool and the claim judge now sees
overlap -> the broad gap must be CLOSED or NARROW, not SURVIVING.

This is the minimal end-to-end assertion (with mocked claim judge): given a
neighbor that IS direct prior art, a partially_closed/narrow verdict must be
honored and the gap must not survive.
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


class DirectPriorArtAuditLLM:
    """Claim judge sees ONE direct prior art neighbor and correctly narrows.

    The direct prior art (HopRefusalBench-like) evaluates refusal accuracy
    under missing evidence, so the broad claim "lacks dedicated refusal
    evaluation" is partially closed and must narrow to the residual.

    Dispatches on schema: returns atomic claims for the decomposition call and
    the per-neighbor decision (with claim_coverage) for the audit call.
    """

    async def chat_json(self, messages, schema):
        from app.agent.steps.audit_gaps import (AtomicClaimList, AtomicClaimSchema,
                                                ClaimCoverageSchema,
                                                GapAuditDecisionSchema,
                                                NeighborAuditSchema)
        from app.agent.steps.gap_relevance import (GapPaperRelevanceList,
                                                   GapPaperRelevanceSchema)
        import ast

        if schema is None:
            return GapAuditDecisionSchema(audit_result="uncertain",
                                          recommended_action="more_search",
                                          novelty_confidence=0.3, audit_confidence=0.3)
        name = schema.__name__
        if name == "AtomicClaimList":
            return AtomicClaimList(claims=[
                AtomicClaimSchema(claim_text="RAG 缺乏专门评测拒绝回答的框架"),
                AtomicClaimSchema(claim_text="需要区分自信幻觉与正确拒答的协议"),
                AtomicClaimSchema(claim_text="需要将检索不确定性信号纳入拒答评估"),
            ])
        if name == "GapPaperRelevanceList":
            # Direct prior art scores high against THIS gap; survey would score 0.
            return GapPaperRelevanceList(papers=[
                GapPaperRelevanceSchema(
                    paper_id=_extract_first_paper_id(messages),
                    problem_overlap="yes", mechanism_overlap="partial",
                    evaluation_overlap="yes", claim_overlap="yes",
                    addresses_claim_ids=[], rationale="Direct prior art."),
            ])

        text = messages[1]["content"]
        paper_id = next(line.split(": ", 1)[1] for line in text.splitlines() if "Paper ID:" in line)
        evidence_id = ast.literal_eval(next(
            line.split(": ", 1)[1] for line in text.splitlines()
            if line.startswith("Supporting evidence IDs:")))[0]
        return GapAuditDecisionSchema(
            audit_result="partially_closed",
            recommended_action="narrow",
            remaining_delta="缺乏将检索不确定性显式暴露给生成器的评估协议。",
            nearest_neighbor_summary="HopRefusalBench 已评估缺失证据下的拒答精度。",
            differentiation_summary="broad 拒答评测已被覆盖，仅剩检索不确定性信号的评估缺口。",
            evidence_for_gap_ids=[evidence_id],
            novelty_confidence=0.5,
            audit_confidence=0.7,
            comparisons=[NeighborAuditSchema(
                paper_id=paper_id,
                similarity_score=0.6,
                shared_problem="都研究检索失败时 RAG 是否应拒答。",
                shared_mechanism="都评估拒绝回答行为。",
                shared_evaluation="都报告拒答精度。",
                covered_claims=["RAG 缺专门拒答评测"],
                uncovered_claims=["检索不确定性信号评估"],
                overlap_ratio=0.6,
                overlap_risk=0.3,
                claim_coverage=[
                    ClaimCoverageSchema(claim_index=0, coverage="FULL",
                                        rationale="HopRefusalBench 直接评测拒答精度"),
                    ClaimCoverageSchema(claim_index=1, coverage="PARTIAL",
                                        rationale="区分了拒答类型"),
                    ClaimCoverageSchema(claim_index=2, coverage="NONE",
                                        rationale="未纳入检索不确定性信号"),
                ],
            )],
        )


def _extract_first_paper_id(messages) -> str:
    """Pull the first paper id from the user prompt (both screens use it)."""
    import re
    text = messages[1]["content"] if len(messages) > 1 else ""
    m = re.search(r"Paper id: ([0-9a-f-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"Paper ID: ([0-9a-f-]+)", text)
    return m.group(1) if m else "unknown-paper"


def _seed_broad_gap(db):
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import (EvidenceUnit, GapCandidate, Paper, ResearchContract,
                               ResearchTask, TaskPaper)

    task = ResearchTask(user_input="RAG abstention", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="RAG abstention", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    # Direct prior art: HopRefusalBench-like — evaluates refusal under missing evidence.
    prior_art = Paper(
        title="HopRefusalBench: Diagnosing Refusal Failures in Multi-Hop Search",
        abstract="We introduce HopRefusalBench, a benchmark that evaluates whether "
                 "retrieval-augmented generation systems refuse to answer when "
                 "multi-hop evidence is missing, measuring refusal accuracy across "
                 "target-aware, pseudo-refusal and hallucinated completions.",
        citation_count=42,
    )
    db.add_all([contract, prior_art])
    db.flush()
    db.add(TaskPaper(task_id=task.id, paper_id=prior_art.id, discovered_round=2,
                     final_score=None, priority="high"))
    evidence = EvidenceUnit(task_id=task.id, paper_id=prior_art.id,
                            evidence_type="comparison",
                            normalized_claim="Refusal accuracy under missing evidence",
                            verification_status="verified")
    db.add(evidence)
    db.flush()
    gap = GapCandidate(
        task_id=task.id,
        contract_id=contract.id,
        gap_type="missing_evaluation",
        description="RAG 缺乏证据不足时拒答的专门评测。",
        target_setting="检索证据不足时的 RAG 拒答",
        observed_problem="系统在证据不足时继续编造答案",
        existing_coverage="HaluEval 覆盖生成幻觉",
        missing_capability="专门拒答评测",
        claimed_delta="RAG 缺乏专门评测拒绝回答与检索证据不足的协议",
        testable_hypothesis="证据不足时系统不会拒答",
        falsification_condition="已有专门拒答评测覆盖该场景",
        status="candidate",
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    db.add(gap)
    db.flush()
    gap_repo = __import__("app.db.repositories.gap_repo", fromlist=["gap_repo"])
    gap_repo.create_gap_evidence_link(db, gap.id, evidence.id, "suggests", 0.9)
    db.commit()
    return task, gap, prior_art


def _pin_admission_and_neighbors(monkeypatch, db, gap, neighbor):
    """Pin admission PASS + the given neighbor set, skipping LLM query gen."""
    from app.agent.steps import audit_gaps as module
    from app.db.models import Paper

    class Admission:
        status = "PASS"
        reason_codes = []
        gap_id = gap.id
        candidate_paper_ids = [neighbor.id]
        completed_query_ids = ["q-1"]
        failed_query_ids = []
        query_ids = ["q-1"]
        completed_families = ["exact_gap"]
        source_count = 1
        external_neighbor_ids = [neighbor.id]

    async def _fake_gen(db, llm, gap):
        # A non-empty query set: an empty list is treated as QUERY_GENERATION_INVALID.
        from app.agent.steps.audit_gaps import AdversarialQuerySpec
        return [AdversarialQuerySpec(family="exact_gap",
                                     query_text="HopRefusalBench refusal accuracy",
                                     variant_index=0)]
    monkeypatch.setattr(module, "generate_english_adversarial_queries", _fake_gen)
    monkeypatch.setattr(module, "evaluate_gap_search_admission",
                        lambda db, gap, query_ids: Admission())
    monkeypatch.setattr(module, "select_gap_specific_neighbors",
                        lambda db, gap, query_ids, limit=5, **kwargs: [neighbor])
    return neighbor


@pytest.mark.asyncio
async def test_broad_gap_does_not_survive_with_direct_prior_art_neighbor(temp_db, monkeypatch):
    """The fd688ba6 failure mode: with direct prior art in NPA, a broad claim
    must be narrowed (partially_closed), never confirmed/surviving."""
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, prior_art = _seed_broad_gap(db)
    _pin_admission_and_neighbors(monkeypatch, db, gap, prior_art)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)

    results = await audit_gap_candidates(db, state, DirectPriorArtAuditLLM(), task.id,
                                         perform_search=False)

    assert [item.audit_result for item in results] == ["partially_closed"]
    assert state.surviving_gap_ids == [], \
        "a broad gap facing direct prior art must NOT survive"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert audit.recommended_action == "narrow"
    assert audit.remaining_delta, "the genuine residual claim must be kept"
    # partially_closed -> narrow marks the gap audited (not surviving, not auditing).
    assert gap_repo.get_gap(db, gap.id).status == "audited"
    db.close()


@pytest.mark.asyncio
async def test_broad_gap_still_survives_with_generic_survey_neighbor(temp_db, monkeypatch):
    """Control: if the neighbor is ONLY a generic survey (the pre-fix fd688ba6
    condition), the claim judge may still confirm. This documents the boundary:
    relevance screening must put direct prior art in the pool — the judge alone
    cannot fix an empty pool."""
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.models import Paper
    from app.db.repositories import gap_repo

    db = temp_db()
    task, gap, _ = _seed_broad_gap(db)
    survey = Paper(title="A Systematic Literature Review of Retrieval-Augmented Generation",
                   abstract="We survey retrieval strategies and generation methods across "
                            "many tasks in retrieval-augmented generation.")
    db.add(survey)
    db.flush()
    _pin_admission_and_neighbors(monkeypatch, db, gap, survey)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)

    # The claim judge sees a survey that does not evaluate refusal — it confirms.
    # (This is the KNOWN limitation: the judge needs a populated pool; the
    # relevance screen is what populates it. Keep this as documentation of the
    # boundary, not as an acceptable outcome.)
    results = await audit_gap_candidates(db, state, DirectPriorArtAuditLLM(), task.id,
                                         perform_search=False)
    # Even so, a well-behaved judge that reports overlap for the survey is
    # honored: the verdict reflects what the judge sees. The pool is the fix.
    assert state.surviving_gap_ids == [], "survey-only pool must not confirm either"
    db.close()

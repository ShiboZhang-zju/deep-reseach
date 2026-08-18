"""Semantic-dedup tests: lock the four cases that keep dedup correct.

Without these, dedup can drift toward either "only compare against active gaps"
(missing a reworded broad claim after narrowing) or "kill on surface keywords"
(false-positive dedup). The four cases:

1. exact duplicate      — identical fingerprint collapses to one gap.
2. strong paraphrase    — same gap reworded collapses to one gap.
3. reworded broad claim vs a superseded (narrowed) family — still blocked.
4. keyword overlap but different mechanism — NOT killed.
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
    from alembic.config import Config
    from alembic import command
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


def _seed_mining_context(db):
    """Task + contract + one covered question backed by limitation evidence."""
    from app.db.models import (
        EvidenceUnit, Paper, QuestionEvidenceLink,
        ResearchContract, ResearchQuestion, ResearchTask,
    )

    task = ResearchTask(user_input="agent memory", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="固定预算下状态变化会如何影响记忆系统？", question_type="failure",
        importance=0.9, searchability=0.8, status="partially_covered",
    )
    db.add(question)
    db.flush()
    p1 = Paper(title="Evidence Paper A")
    p2 = Paper(title="Evidence Paper B")
    db.add_all([p1, p2])
    db.flush()
    ev1 = EvidenceUnit(
        task_id=task.id, paper_id=p1.id, evidence_type="limitation",
        normalized_claim="状态变化后固定预算记忆会遗漏关键历史证据。",
        original_span="遗漏关键证据", section="Limitations", page_number=8,
        span_start=10, span_end=22, source_chunk_hash="test-fulltext",
        verification_status="verified", extraction_confidence=0.8,
    )
    ev2 = EvidenceUnit(
        task_id=task.id, paper_id=p2.id, evidence_type="comparison",
        normalized_claim="现有评测未覆盖状态变化边界。", verification_status="abstract_only",
    )
    db.add_all([ev1, ev2])
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=ev1.id,
                             relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=ev2.id,
                             relation_type="supports", relevance_score=0.9),
    ])
    db.commit()
    return task, contract, question, [ev1.id, ev2.id]


def _gap(observed, missing, claimed, qid, ev_ids):
    from app.schemas.schemas import GapCandidateSchema
    return GapCandidateSchema(
        gap_type="boundary_gap", description="a research gap in agent memory",
        target_setting="agent memory", observed_problem=observed,
        existing_coverage="generic QA", missing_capability=missing,
        claimed_delta=claimed, testable_hypothesis="hypothesis",
        falsification_condition="falsification", question_ids=[qid],
        supporting_evidence_ids=ev_ids,
    )


class MultiGapLLM:
    def __init__(self, gaps):
        self.gaps = gaps

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList
        return GapCandidateList(gaps=self.gaps)


def _patch_embed(monkeypatch, vectors):
    """Replace the embedding call with fixed vectors so cosine is controlled."""
    import app.agent.steps.mine_gaps as mg

    async def _embed(texts):
        # vectors length must equal len(existing_fps) + len(candidate_fps),
        # otherwise mine disables dedup (dedup_enabled=False).
        return vectors

    monkeypatch.setattr(mg, "_embed_fingerprints", _embed)


async def _mine(db, monkeypatch, llm, vectors):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    task, contract, question, ev_ids = _seed_mining_context(db)
    _patch_embed(monkeypatch, vectors)
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    gaps = await mine_gap_candidates(db, state, llm, task.id)
    return task, contract, question, ev_ids, gaps


@pytest.mark.asyncio
async def test_exact_duplicate_collapses(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, question, ev_ids = _seed_mining_context(db)
    dup = _gap("state change drops evidence", "state-change boundary eval",
               "evaluate state-change boundary under fixed budget", question.id, ev_ids)
    llm = MultiGapLLM([dup, dup])
    _patch_embed(monkeypatch, [[1.0, 0.0], [1.0, 0.0]])  # identical fingerprint
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1, "exact duplicate must collapse to a single gap"
    db.close()


@pytest.mark.asyncio
async def test_strong_paraphrase_collapses(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, question, ev_ids = _seed_mining_context(db)
    a = _gap("state change drops evidence", "state-change boundary eval",
             "evaluate the boundary of state change under a fixed budget",
             question.id, ev_ids)
    b = _gap("after the state changes, prior evidence is lost", "a boundary evaluation",
             "measure how state-change degrades retention under a fixed token budget",
             question.id, ev_ids)
    llm = MultiGapLLM([a, b])
    # Cosine of [1,0] vs [0.995,0.1] ~ 0.995 > 0.85 -> duplicate.
    _patch_embed(monkeypatch, [[1.0, 0.0], [0.995, 0.1]])
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 1, "strong paraphrase must collapse to a single gap"
    db.close()


@pytest.mark.asyncio
async def test_reworded_broad_claim_is_blocked_by_superseded_family(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION, mine_gap_candidates
    from app.db.models import GapCandidate
    from app.db.repositories import gap_repo

    db = temp_db()
    task, contract, question, ev_ids = _seed_mining_context(db)

    # A narrowed family: v1 (broad) was superseded, v2 is the active narrow head.
    v1 = gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="broad", target_setting="agent memory",
        observed_problem="state change drops evidence",
        missing_capability="state-change boundary eval",
        claimed_delta="evaluate state-change boundary under fixed budget",
        testable_hypothesis="h", falsification_condition="f",
        provenance_status="complete", question_ids=[question.id], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    v1.status = "superseded"
    db.flush()
    gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="narrow", target_setting="agent memory",
        observed_problem="state change drops evidence",
        missing_capability="state-change boundary eval",
        claimed_delta="quantify retention drop at the exact state-change boundary",
        testable_hypothesis="h", falsification_condition="f",
        provenance_status="complete", question_ids=[question.id], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
        version=2, canonical_gap_id=v1.id, parent_gap_id=v1.id,
    )
    db.commit()

    # New candidate rewords v1's ORIGINAL broad claim (not the narrow v2).
    broad = _gap("state change drops evidence", "state-change boundary eval",
                 "evaluate state-change boundary under fixed budget",
                 question.id, ev_ids)
    llm = MultiGapLLM([broad])
    # existing_fps has [v1, v2]; candidate_fps has [broad].
    # v1 == broad exactly, so the broad reword must be blocked even though v1 is
    # superseded and only v2 (a different, narrower claim) is active.
    _patch_embed(monkeypatch, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 0, "reworded broad claim must be blocked by its superseded family"
    db.close()


@pytest.mark.asyncio
async def test_keyword_overlap_with_different_mechanism_is_not_killed(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, question, ev_ids = _seed_mining_context(db)
    # Same surface words ("state change", "boundary"), different mechanisms:
    # one is a retention/evaluation gap, the other a retrieval-latency gap.
    a = _gap("state change drops evidence", "state-change boundary eval",
             "evaluate retention across the state-change boundary", question.id, ev_ids)
    b = _gap("state change raises retrieval latency", "state-change boundary latency",
             "measure the latency spike across the state-change boundary", question.id, ev_ids)
    llm = MultiGapLLM([a, b])
    _patch_embed(monkeypatch, [[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> cosine 0
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert len(gaps) == 2, "distinct mechanisms with shared keywords must both survive"
    db.close()

"""Phase 2.2A Final Closure: Strict integration tests.

Tests:
1. test_retry_does_not_pollute_round_termination — 3 coverage attempts, 1 round
2. test_search_phase_exact_reuse — search_call_count == queries_per_round
3. test_round_search_result_serialization_roundtrip — to/from payload exact match
4. test_phase_run_output_json_migration — output_json column exists after migration
5. test_readiness_gate_missing_snapshot — status == failed
6. test_readiness_gate_insufficient_coverage — status == more_research_required
7. test_readiness_gate_pass — status == ready
8. test_superseded_contract_isolation — v1 coverage doesn't bleed into v2
9. test_bootstrap_rejects_missing_manifest_column — papers missing title
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Shared fixtures (reuse from test_search_loop_integration) ===

from tests.test_search_loop_integration import FakeLLM, FakeSearchService, temp_db


# === Test 1: Retry does not pollute round-level termination state ===

@pytest.mark.asyncio
async def test_retry_does_not_pollute_round_termination(temp_db):
    """3 coverage attempts (fail, fail, success) → only 1 completed round,
    no_new_high_priority_count updated at most once.
    """
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.steps.decompose_research_space import decompose_research_space, select_target_questions

    task = ResearchTask(user_input="agent memory token budget", status="pending")
    state = ResearchState(task_id=task.id, user_input=task.user_input, pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id

    llm = FakeLLM()
    state = task_repo.get_state(db, task_id)
    state.normalized_topic = "agent memory token budget"
    state.keywords = ["memory", "token"]
    task_repo.save_state(db, task_id, state)
    db.commit()

    state = task_repo.get_state(db, task_id)
    await build_research_contract(db, state, llm, task_id)
    state = task_repo.get_state(db, task_id)
    await decompose_research_space(db, state, llm, task_id)
    db.commit()

    state = task_repo.get_state(db, task_id)
    target_qs = select_target_questions(db, task_id, limit=3)
    llm.set_question_ids([q.id for q in target_qs])

    fake_search = FakeSearchService()

    # Patch evidence extraction to succeed
    from tests.test_search_loop_integration import _make_evidence_patch
    evidence_patch = _make_evidence_patch(db)

    # Patch coverage to fail twice then succeed
    coverage_call_count = [0]
    async def _fail_twice_then_succeed(db, state, llm, task_id, round_num):
        coverage_call_count[0] += 1
        if coverage_call_count[0] <= 2:
            raise RuntimeError(f"Coverage attempt {coverage_call_count[0]} fails")
        from app.agent.steps.update_coverage import update_coverage_matrix
        return await update_coverage_matrix(db, state, llm, task_id, round_num)

    # Patch settings to limit max_rounds=1 (prevents loop from continuing)
    with patch("app.agent.policy.settings.max_rounds", 1), \
         patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.runner.extract_evidence_units", side_effect=evidence_patch), \
         patch("app.agent.runner.update_coverage_matrix", _fail_twice_then_succeed):

        # Spy on early_termination_check
        from app.agent import policy
        original_check = policy.early_termination_check
        spy_calls = []

        def _spy_check(state, no_new_high_count, dup_rate):
            spy_calls.append({
                "current_round": state.current_round,
                "no_new_high_priority_count": no_new_high_count,
                "duplicate_rate": dup_rate,
            })
            return original_check(state, no_new_high_count, dup_rate)

        with patch("app.agent.policy.early_termination_check", _spy_check):
            from app.agent.runner import _run_search_loop
            state = task_repo.get_state(db, task_id)
            result = await _run_search_loop(db, state, llm, task_id)

    # Assert: only 1 completed round
    assert result.completed_rounds == 1, \
        f"Expected 1 completed round, got {result.completed_rounds}"

    # Assert: status is stopped_normally or completed (not failed)
    assert result.status in ("stopped_normally", "completed"), \
        f"Expected stopped_normally or completed, got {result.status}"

    # Assert: early_termination_check was called with no_new_high_priority_count <= 1
    # (not 2 or 3 which would have been the case if retries polluted the counter)
    for call in spy_calls:
        assert call["no_new_high_priority_count"] <= 1, \
            f"no_new_high_priority_count should be <= 1, but was {call['no_new_high_priority_count']}"

    # Assert: search was called 3 times (3 queries per round, not re-run on retry)
    assert fake_search.search_call_count == 3, \
        f"Expected 3 search calls (3 queries), got {fake_search.search_call_count}"

    db.close()


# === Test 2: Search phase exact reuse ===

@pytest.mark.asyncio
async def test_search_phase_exact_reuse(temp_db):
    """Coverage fails once, succeeds on retry → search_call_count == 3 (not 6)."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, SearchQueryRecord, ResearchRound, PhaseRun
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.steps.decompose_research_space import decompose_research_space, select_target_questions

    task = ResearchTask(user_input="agent memory token budget", status="pending")
    state = ResearchState(task_id=task.id, user_input=task.user_input, pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id

    llm = FakeLLM()
    state = task_repo.get_state(db, task_id)
    state.normalized_topic = "agent memory token budget"
    state.keywords = ["memory", "token"]
    task_repo.save_state(db, task_id, state)
    db.commit()

    state = task_repo.get_state(db, task_id)
    await build_research_contract(db, state, llm, task_id)
    state = task_repo.get_state(db, task_id)
    await decompose_research_space(db, state, llm, task_id)
    db.commit()

    state = task_repo.get_state(db, task_id)
    target_qs = select_target_questions(db, task_id, limit=3)
    llm.set_question_ids([q.id for q in target_qs])

    fake_search = FakeSearchService()
    from tests.test_search_loop_integration import _make_evidence_patch
    evidence_patch = _make_evidence_patch(db)

    coverage_call_count = [0]
    async def _fail_once_then_succeed(db, state, llm, task_id, round_num):
        coverage_call_count[0] += 1
        if coverage_call_count[0] == 1:
            raise RuntimeError("Coverage first attempt fails")
        from app.agent.steps.update_coverage import update_coverage_matrix
        return await update_coverage_matrix(db, state, llm, task_id, round_num)

    # Patch settings to limit max_rounds=1
    with patch("app.agent.policy.settings.max_rounds", 1), \
         patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.runner.extract_evidence_units", side_effect=evidence_patch), \
         patch("app.agent.runner.update_coverage_matrix", _fail_once_then_succeed):

        from app.agent.runner import _run_search_loop
        state = task_repo.get_state(db, task_id)
        result = await _run_search_loop(db, state, llm, task_id)

    # Assert: search was called exactly 3 times (3 queries, not re-run on coverage retry)
    assert fake_search.search_call_count == 3, \
        f"Expected 3 search calls (search phase reused on retry), got {fake_search.search_call_count}"

    # Assert: SearchQueryRecord count == 3
    sqr_count = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task_id,
    ).count()
    assert sqr_count == 3, f"Expected 3 SearchQueryRecords, got {sqr_count}"

    # Assert: ResearchRound count == 1
    round_count = db.query(ResearchRound).filter(
        ResearchRound.task_id == task_id,
        ResearchRound.round_number == 1,
    ).count()
    assert round_count == 1, f"Expected 1 ResearchRound for round 1, got {round_count}"

    # Assert: search_round_1 completed PhaseRun == 1
    search_phases = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == "search_round_1",
        PhaseRun.status == "completed",
    ).all()
    assert len(search_phases) == 1, \
        f"Expected 1 completed search_round_1 PhaseRun, got {len(search_phases)}"

    # Assert: update_coverage_round_1 has 1 failed + 1 completed
    cov_phases = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == "update_coverage_round_1",
    ).all()
    failed_cov = [p for p in cov_phases if p.status == "failed"]
    completed_cov = [p for p in cov_phases if p.status == "completed"]
    assert len(failed_cov) == 1, f"Expected 1 failed coverage attempt, got {len(failed_cov)}"
    assert len(completed_cov) == 1, f"Expected 1 completed coverage attempt, got {len(completed_cov)}"

    # Assert: completed_rounds == 1
    assert result.completed_rounds == 1

    db.close()


# === Test 3: RoundSearchResult serialization roundtrip ===

def test_round_search_result_serialization_roundtrip():
    """RoundSearchResult → to_phase_payload → from_phase_payload == original."""
    from app.agent.runner import RoundSearchResult

    original = RoundSearchResult(
        query_ids=["q1", "q2", "q3"],
        query_texts=["query one", "query two", "query three"],
        papers_found=42,
        deduped_count=35,
        new_paper_ids=["p1", "p2", "p3", "p4", "p5"],
        duplicate_rate=0.166,
        high_priority_before=5,
        high_priority_after=8,
        new_high_priority_count=3,
        round_summary="Round 1 found 42 papers with 3 high-priority",
        knowledge_gaps=["gap1", "gap2"],
    )

    payload = original.to_phase_payload()

    # Verify payload is a plain dict with all fields
    expected_fields = {
        "query_ids", "query_texts", "papers_found", "deduped_count",
        "new_paper_ids", "duplicate_rate", "high_priority_before",
        "high_priority_after", "new_high_priority_count",
        "round_summary", "knowledge_gaps",
    }
    assert set(payload.keys()) == expected_fields

    # Serialize to JSON and back (simulates PhaseRun.output_json storage)
    json_str = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    restored_payload = json.loads(json_str)

    restored = RoundSearchResult.from_phase_payload(restored_payload)

    # Assert exact equality on all fields
    assert restored.query_ids == original.query_ids
    assert restored.query_texts == original.query_texts
    assert restored.papers_found == original.papers_found
    assert restored.deduped_count == original.deduped_count
    assert restored.new_paper_ids == original.new_paper_ids
    assert restored.duplicate_rate == original.duplicate_rate
    assert restored.high_priority_before == original.high_priority_before
    assert restored.high_priority_after == original.high_priority_after
    assert restored.new_high_priority_count == original.new_high_priority_count
    assert restored.round_summary == original.round_summary
    assert restored.knowledge_gaps == original.knowledge_gaps

    # Test missing field raises TypeError
    incomplete = {k: v for k, v in payload.items() if k != "new_high_priority_count"}
    with pytest.raises(TypeError, match="missing fields"):
        RoundSearchResult.from_phase_payload(incomplete)


# === Test 4: PhaseRun output_json migration ===

def test_phase_run_output_json_migration(temp_db):
    """Fresh DB has phase_runs.output_json column."""
    engine, Session = temp_db
    from sqlalchemy import inspect
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("phase_runs")}
    assert "output_json" in cols, f"output_json column missing from phase_runs: {cols}"


# === Test 5-8: Readiness Gate tests ===

def _create_readiness_test_data(db, task_id, contract_id, question_count=3,
                                 high_importance_count=2,
                                 add_coverage=True, coverage_score=0.5,
                                 missing_coverage_for_one_high=False,
                                 add_evidence=True):
    """Helper to set up test data for readiness gate tests."""
    from app.db.models import (
        ResearchContract, ResearchQuestion, EvidenceUnit, CoverageRecord,
    )
    import uuid

    # Create active contract
    contract = ResearchContract(
        id=contract_id,
        task_id=task_id,
        topic="test topic",
        status="active",
        version=1,
        input_hash="test_hash_123",
    )
    db.add(contract)
    db.flush()

    # Create questions
    questions = []
    for i in range(question_count):
        importance = 0.9 if i < high_importance_count else 0.5
        q = ResearchQuestion(
            id=str(uuid.uuid4()),
            task_id=task_id,
            contract_id=contract_id,
            question=f"Question {i}?",
            question_type="method",
            importance=importance,
            status="open",
        )
        db.add(q)
        questions.append(q)
    db.flush()

    # Add evidence
    if add_evidence:
        for i in range(3):
            eu = EvidenceUnit(
                id=str(uuid.uuid4()),
                task_id=task_id,
                paper_id=str(uuid.uuid4()),
                evidence_type="method",
                normalized_claim=f"evidence claim {i}",
                verification_status="unverified",
            )
            db.add(eu)
        db.flush()

    # Add coverage records
    if add_coverage:
        high_qs = questions[:high_importance_count]
        for i, q in enumerate(high_qs):
            if missing_coverage_for_one_high and i == 0:
                continue  # Skip first high-importance question
            cr = CoverageRecord(
                id=str(uuid.uuid4()),
                task_id=task_id,
                question_id=q.id,
                coverage_score=coverage_score,
                confidence=0.8,
                round_number=1,
            )
            db.add(cr)

        # Add coverage for non-high questions too
        for q in questions[high_importance_count:]:
            cr = CoverageRecord(
                id=str(uuid.uuid4()),
                task_id=task_id,
                question_id=q.id,
                coverage_score=0.3,
                confidence=0.5,
                round_number=1,
            )
            db.add(cr)

    db.commit()
    return contract, questions


def test_readiness_gate_missing_snapshot(temp_db):
    """O3: Active Contract, 3 active Questions, 1 of 2 high-importance questions
    missing CoverageRecord (snapshot_ratio = 0.5 < 0.6) → downgraded to
    more_research_required (NOT a hard failure). This is the O3 fix: a single
    missing snapshot must not kill the whole task.
    """
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask
    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()

    contract, questions = _create_readiness_test_data(
        db, task.id, "contract-test-1",
        question_count=3, high_importance_count=2,
        add_coverage=True, coverage_score=0.5,
        missing_coverage_for_one_high=True,
    )

    from app.agent.steps.readiness_gate import evaluate_phase2_readiness
    result = evaluate_phase2_readiness(db, task.id, contract.id)

    assert result.status == "more_research_required", \
        f"Expected 'more_research_required' (partial snapshot below ratio), got '{result.status}'"
    assert "coverage_below_ratio" in result.reason
    assert len(result.missing_question_ids) == 1

    db.close()


def test_readiness_gate_insufficient_coverage(temp_db):
    """All questions have latest snapshot, but high-importance coverage = 0
    → status == more_research_required.
    """
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask
    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()

    contract, questions = _create_readiness_test_data(
        db, task.id, "contract-test-2",
        question_count=3, high_importance_count=2,
        add_coverage=True, coverage_score=0.0,  # All coverage = 0
        missing_coverage_for_one_high=False,
    )

    from app.agent.steps.readiness_gate import evaluate_phase2_readiness
    result = evaluate_phase2_readiness(db, task.id, contract.id)

    assert result.status == "more_research_required", \
        f"Expected 'more_research_required', got '{result.status}'"
    assert result.high_importance_question_count == 2
    assert result.high_importance_covered_count == 0

    db.close()


def test_readiness_gate_pass(temp_db):
    """All conditions met → status == ready."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask
    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()

    contract, questions = _create_readiness_test_data(
        db, task.id, "contract-test-3",
        question_count=3, high_importance_count=2,
        add_coverage=True, coverage_score=0.5,
        missing_coverage_for_one_high=False,
    )

    from app.agent.steps.readiness_gate import evaluate_phase2_readiness
    result = evaluate_phase2_readiness(db, task.id, contract.id)

    assert result.status == "ready", \
        f"Expected 'ready', got '{result.status}' (reason: {result.reason})"
    assert result.ready is True
    assert result.high_importance_covered_count == 2

    db.close()


def test_superseded_contract_isolation(temp_db):
    """Contract v1 has high coverage, v2 lacks coverage → Gate uses v2."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, ResearchContract, ResearchQuestion
    from app.agent.steps.readiness_gate import evaluate_phase2_readiness
    import uuid

    task = ResearchTask(user_input="test superseded", status="searching")
    db.add(task)
    db.commit()

    # Contract v1 (superseded) with high coverage
    contract_v1 = ResearchContract(
        id="contract-v1",
        task_id=task.id,
        topic="v1 topic",
        status="superseded",
        version=1,
        input_hash="v1_hash",
    )
    db.add(contract_v1)
    db.flush()

    # v1 questions (superseded)
    for i in range(3):
        q = ResearchQuestion(
            id=str(uuid.uuid4()),
            task_id=task.id,
            contract_id="contract-v1",
            question=f"V1 Q{i}?",
            question_type="method",
            importance=0.9,
            status="superseded",
        )
        db.add(q)

    # Contract v2 (active) with NO coverage
    contract_v2 = ResearchContract(
        id="contract-v2",
        task_id=task.id,
        topic="v2 topic",
        status="active",
        version=2,
        input_hash="v2_hash",
    )
    db.add(contract_v2)
    db.flush()

    # v2 questions (active)
    for i in range(3):
        q = ResearchQuestion(
            id=str(uuid.uuid4()),
            task_id=task.id,
            contract_id="contract-v2",
            question=f"V2 Q{i}?",
            question_type="method",
            importance=0.9,
            status="open",
        )
        db.add(q)

    # Add evidence (belongs to task, not contract)
    from app.db.models import EvidenceUnit
    for i in range(3):
        eu = EvidenceUnit(
            id=str(uuid.uuid4()),
            task_id=task.id,
            paper_id=str(uuid.uuid4()),
            evidence_type="method",
            normalized_claim=f"evidence {i}",
            verification_status="unverified",
        )
        db.add(eu)

    db.commit()

    result = evaluate_phase2_readiness(db, task.id, "contract-v2")

    # Should be "failed" because ALL of v2's high-importance questions have no
    # coverage snapshot (snapshot_ratio == 0.0 → total control-plane failure).
    assert result.status == "failed", \
        f"Expected 'failed' (v2 missing all coverage), got '{result.status}'"
    assert "no_high_importance_question_has_coverage_snapshot" in result.reason

    # Verify it's using v2's questions, not v1's
    assert result.active_question_count == 3
    assert result.high_importance_question_count == 3

    db.close()


# === Test 9: Bootstrap rejects missing manifest column ===

def test_bootstrap_rejects_missing_manifest_column():
    """Three base tables exist but papers is missing 'title' column →
    bootstrap returns False and doesn't create alembic_version.
    """
    from sqlalchemy import create_engine, text, inspect
    import gc

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{db_path}"

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE research_tasks (
                    id TEXT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    status TEXT DEFAULT 'pending'
                )
            """))
            # papers MISSING 'title' column
            conn.execute(text("""
                CREATE TABLE papers (
                    id TEXT PRIMARY KEY,
                    abstract TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE task_papers (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    paper_id TEXT,
                    discovered_round INTEGER
                )
            """))
            conn.commit()
        engine.dispose()

        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        sys.path.insert(0, scripts_dir)
        from bootstrap_db import bootstrap
        success = bootstrap(db_url)

        assert success is False, "Bootstrap should reject missing manifest column"

        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert 'alembic_version' not in tables, \
            "alembic_version should NOT be created for missing manifest"
        engine.dispose()

    finally:
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

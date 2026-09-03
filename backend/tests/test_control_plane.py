"""Phase 2.2A Closure: Control-plane integration tests.

Uses real Alembic temporary SQLite — no MagicMock for DB.
"""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database via Alembic migrations."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command

    cfg = AlembicConfig()
    cfg.set_main_option('sqlalchemy.url', f'sqlite:///{db_path}')
    script_loc = os.path.join(os.path.dirname(__file__), "..", "alembic_migrations")
    cfg.set_main_option('script_location', script_loc)
    alembic_command.upgrade(cfg, 'head')

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)

    yield engine, Session

    engine.dispose()
    import gc
    gc.collect()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


# === #2: Contract invalidation integration test ===

@pytest.mark.asyncio
async def test_contract_invalidation_on_clarification_change(temp_db):
    """Contract v1 → v2 when clarification answer changes.

    Full test:
    - Contract v1 with decomposed Questions
    - Clarification answer changes → Contract v2
    - Decompose v2 Questions
    - Assert: v1 Contract superseded, v1 Questions all superseded,
      v2 Contract active, v2 Questions active,
      state.active_question_ids only contains v2 Question IDs,
      build_contract PhaseRun input_version == v2.input_hash
    """
    engine, Session = temp_db

    db = Session()
    from app.db.models import (
        ResearchTask, ResearchContract, ResearchQuestion,
        PhaseRun as PhaseRunModel,
    )
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract, compute_contract_input_version
    from app.agent.steps.decompose_research_space import decompose_research_space
    from app.schemas.schemas import ResearchContractSchema, ResearchDecompositionSchema
    from app.db.repositories.paper_repo import save_feedback
    import json as _json

    # Create long input (> 300 chars)
    long_input = "Agent memory under fixed token budgets, focusing on temporal state changes. " * 5
    long_input += " Exclude large benchmark construction."

    task = ResearchTask(user_input=long_input, status="pending")
    state = ResearchState(task_id=task.id, user_input=long_input, pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id

    # FakeLLM for Contract + Decomposition
    class ContractDecomposeLLM:
        def __init__(self):
            self._decompose_count = 0

        async def chat_json(self, messages, schema, **kwargs):
            schema_name = schema.__name__ if hasattr(schema, '__name__') else str(schema)
            if schema_name == "ResearchContractSchema":
                return ResearchContractSchema(
                    topic="agent memory token budget",
                    target_problem="智能体记忆在固定token预算下的状态变化",
                    target_setting="LLM Agent系统",
                    desired_output="method",
                    novelty_bar="conference",
                    preferred_directions=["memory-efficient architectures"],
                    excluded_directions=["large-scale model training"],
                    key_terms=["agent memory", "token budget"],
                    experiment_preferences={"prefer_gpu": False},
                    confidence=0.8,
                )
            elif schema_name == "ResearchDecompositionSchema":
                from app.schemas.schemas import (
                    ResearchDecompositionSchema as RDS,
                    ResearchQuestionSchema, ResearchAxisSchema,
                )
                self._decompose_count += 1
                return RDS(
                    axes=[
                        ResearchAxisSchema(axis_name="problem", values=["memory overflow"]),
                        ResearchAxisSchema(axis_name="method", values=["compression", "retrieval"]),
                        ResearchAxisSchema(axis_name="evaluation", values=["accuracy", "latency"]),
                    ],
                    questions=[
                        ResearchQuestionSchema(
                            question=f"V{self._decompose_count} Q1: How to handle memory budget?",
                            question_type="method", importance=0.9, searchability=0.8,
                            axis_name="method",
                        ),
                        ResearchQuestionSchema(
                            question=f"V{self._decompose_count} Q2: What datasets exist?",
                            question_type="dataset", importance=0.7, searchability=0.6,
                            axis_name="dataset",
                        ),
                        ResearchQuestionSchema(
                            question=f"V{self._decompose_count} Q3: What are failure modes?",
                            question_type="failure", importance=0.6, searchability=0.5,
                            axis_name="failure",
                        ),
                        ResearchQuestionSchema(
                            question=f"V{self._decompose_count} Q4: How does budget affect retrieval?",
                            question_type="problem", importance=0.85, searchability=0.75,
                            axis_name="problem",
                        ),
                        ResearchQuestionSchema(
                            question=f"V{self._decompose_count} Q5: How to evaluate temporal state?",
                            question_type="evaluation", importance=0.8, searchability=0.7,
                            axis_name="evaluation",
                        ),
                    ],
                )
            return None

        async def chat(self, messages, **kwargs):
            return "ok"

    llm = ContractDecomposeLLM()

    # --- Contract v1 (via execute_phase for PhaseRun tracking) ---
    state = task_repo.get_state(db, task_id)
    state.clarification_questions = ["What GPU budget?", "What dataset?"]
    state.user_input = long_input + "\nClarifications:\nAnswer A"
    task_repo.save_state(db, task_id, state)
    db.commit()

    # Save clarification_answer feedback v1
    save_feedback(db, task_id, "clarification_answer",
                  _json.dumps({"questions": ["What GPU budget?"], "answers": ["Answer A"]},
                              ensure_ascii=False),
                  None, False)
    db.commit()

    # Build Contract v1 via execute_phase (creates PhaseRun)
    from app.services import phase_service
    from app.agent.steps.build_contract import compute_contract_input_version

    task_obj = db.get(ResearchTask, task_id)
    state = task_repo.get_state(db, task_id)
    v1_input_version = compute_contract_input_version(db, task_obj, state)

    async def _build_v1_op(db):
        return await build_research_contract(db, state, llm, task_id)

    contract_v1 = await phase_service.execute_phase(
        db, task_id, "build_contract", _build_v1_op,
        input_version=v1_input_version,
    )
    db.commit()
    assert contract_v1 is not None
    assert contract_v1.version == 1
    assert contract_v1.status == "active"

    # Decompose v1 Questions
    state = task_repo.get_state(db, task_id)
    v1_questions = await decompose_research_space(db, state, llm, task_id)
    db.commit()

    assert len(v1_questions) == 5
    v1_question_ids = [q.id for q in v1_questions]
    for q in v1_questions:
        assert q.status == "open"
        assert q.contract_id == contract_v1.id

    # Verify state.active_question_ids populated
    state = task_repo.get_state(db, task_id)
    assert set(state.active_question_ids) == set(v1_question_ids)

    # Compute v1 hash
    v1_hash = v1_input_version

    # --- Simulate clarification answer change ---
    state = task_repo.get_state(db, task_id)
    state.user_input = long_input + "\nClarifications:\nAnswer B"
    task_repo.save_state(db, task_id, state)
    db.commit()

    # Save new clarification_answer feedback
    save_feedback(db, task_id, "clarification_answer",
                  _json.dumps({"questions": ["What GPU budget?"], "answers": ["Answer B"]},
                              ensure_ascii=False),
                  None, False)
    db.commit()

    # Compute v2 hash — should be different
    state = task_repo.get_state(db, task_id)
    task_obj = db.get(ResearchTask, task_id)
    v2_input_version = compute_contract_input_version(db, task_obj, state)
    assert v1_hash != v2_input_version, "Hash should differ when clarification answer changes"

    # Build Contract v2 via execute_phase (creates PhaseRun)
    async def _build_v2_op(db):
        return await build_research_contract(db, state, llm, task_id)

    contract_v2 = await phase_service.execute_phase(
        db, task_id, "build_contract", _build_v2_op,
        input_version=v2_input_version,
    )
    db.commit()

    assert contract_v2 is not None
    assert contract_v2.version == 2
    assert contract_v2.status == "active"

    # v1 should be superseded
    db.refresh(contract_v1)
    assert contract_v1.status == "superseded"
    assert contract_v1.input_hash != contract_v2.input_hash

    # v1 Questions should all be superseded
    v1_qs_after = db.query(ResearchQuestion).filter(
        ResearchQuestion.contract_id == contract_v1.id,
    ).all()
    assert len(v1_qs_after) == 5
    for q in v1_qs_after:
        assert q.status == "superseded", f"V1 Q {q.id[:8]} should be superseded, got {q.status}"

    # state.contract_id should point to v2
    state = task_repo.get_state(db, task_id)
    assert state.contract_id == contract_v2.id

    # Decompose v2 Questions
    v2_questions = await decompose_research_space(db, state, llm, task_id)
    db.commit()

    assert len(v2_questions) == 5
    v2_question_ids = [q.id for q in v2_questions]
    for q in v2_questions:
        assert q.status == "open"
        assert q.contract_id == contract_v2.id

    # state.active_question_ids should only contain v2 Question IDs
    state = task_repo.get_state(db, task_id)
    assert set(state.active_question_ids) == set(v2_question_ids)

    # Verify no v1 question IDs in active_question_ids
    v1_id_set = set(v1_question_ids)
    v2_id_set = set(v2_question_ids)
    assert v1_id_set != v2_id_set, "v1 and v2 question IDs must differ"
    assert v1_id_set.isdisjoint(v2_id_set), "v1 and v2 question IDs should not overlap"

    # Verify PhaseRun for build_contract has input_version == v2.input_hash
    build_contract_phases = db.query(PhaseRunModel).filter(
        PhaseRunModel.task_id == task_id,
        PhaseRunModel.phase_name == "build_contract",
        PhaseRunModel.status == "completed",
    ).all()
    assert len(build_contract_phases) >= 2  # v1 and v2

    # Latest build_contract PhaseRun should have v2 input_hash
    latest_bc = sorted(build_contract_phases, key=lambda p: p.created_at)[-1]
    assert latest_bc.input_version == v2_input_version, \
        f"PhaseRun input_version {latest_bc.input_version[:16]} != v2.input_hash {v2_input_version[:16]}"

    db.close()


# === P0-2: /start must fail fast instead of silently accepting ===

def test_start_returns_429_when_capacity_full(temp_db, monkeypatch):
    """A full registry must yield an honest 429. A silent 200 left the task
    pending forever and eval drivers burned their whole wall clock polling a
    task that never started."""
    engine, Session = temp_db

    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_db_session
    from app.api.routes import tasks as tasks_routes
    from app.db.models import ResearchTask

    db = Session()
    task = ResearchTask(user_input="capacity probe", status="pending")
    db.add(task)
    db.commit()
    task_id = task.id
    db.close()

    def _override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = _override_db
    monkeypatch.setattr(tasks_routes, "capacity_has_room", lambda: False)
    try:
        client = TestClient(app)
        response = client.post(f"/api/tasks/{task_id}/start")
        assert response.status_code == 429, (
            f"Expected 429 when capacity is full, got {response.status_code}: {response.text}"
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_deferred_start_marks_task_failed_when_rejected(temp_db, monkeypatch):
    """When the deferred atomic check rejects (capacity full), the task must
    land in a terminal state with a stop reason instead of hanging pending."""
    engine, Session = temp_db

    from app.db.models import ResearchTask
    from app.api.routes import tasks as tasks_routes

    db = Session()
    task = ResearchTask(user_input="deferred reject probe", status="pending")
    db.add(task)
    db.commit()
    task_id = task.id
    db.close()

    def _reject(_task_id):
        # start_agent is a synchronous function; a fake must be too.
        return False

    monkeypatch.setattr(tasks_routes, "start_agent", _reject)
    # _deferred_start_agent opens its own session; point it at the temp DB.
    monkeypatch.setattr(tasks_routes, "SessionLocal", Session)

    await tasks_routes._deferred_start_agent(task_id)

    db = Session()
    try:
        row = db.query(ResearchTask).filter(ResearchTask.id == task_id).one()
        assert row.status == "failed"
        assert row.stop_reason == "max_concurrent_agents_reached"
    finally:
        db.close()


# === #1: Clarification feedback preserves questions ===

@pytest.mark.asyncio
async def test_clarification_feedback_preserves_questions(temp_db):
    """Feedback JSON contains questions, answers, and submitted_at.

    Uses real FastAPI TestClient to call the actual /clarify endpoint.
    Only _deferred_start_agent is mocked.
    """
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask, UserFeedback
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState

    task = ResearchTask(user_input="test input", status="waiting_for_clarification")
    state = ResearchState(task_id=task.id, user_input="test input", pipeline_version=2)
    state.clarification_questions = ["Question 1?", "Question 2?"]
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id
    db.close()

    # Use real FastAPI TestClient
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_db_session

    # Override DB dependency to use our temp DB
    def _override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db_session] = _override_db

    # Mock only _deferred_start_agent (so it doesn't try to start the real agent)
    with patch("app.api.routes.tasks._deferred_start_agent", new_callable=AsyncMock):
        client = TestClient(app)
        response = client.post(
            f"/api/tasks/{task_id}/clarify",
            json={"answers": ["Answer 1", "Answer 2"]},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.json()["status"] == "restarted"

    # Verify feedback was saved correctly
    db = Session()
    fb = db.query(UserFeedback).filter(
        UserFeedback.task_id == task_id,
        UserFeedback.feedback_type == "clarification_answer",
    ).first()
    assert fb is not None, "UserFeedback should exist"

    payload = json.loads(fb.content)
    assert payload["questions"] == ["Question 1?", "Question 2?"]
    assert payload["answers"] == ["Answer 1", "Answer 2"]
    assert "submitted_at" in payload
    assert payload["submitted_at"] != ""

    # State should have clarification_questions cleared
    state = task_repo.get_state(db, task_id)
    assert state.clarification_questions == []

    db.close()


# === #3: SearchQuery lifecycle ===

@pytest.mark.asyncio
async def test_searchquery_lifecycle_pending_to_completed(temp_db):
    """SearchQueryRecord goes from pending to completed."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask, SearchQueryRecord
    from app.db.repositories.search_query_repo import save_search_query, update_query_results

    task = ResearchTask(user_input="test", status="pending")
    db.add(task)
    db.commit()

    # Save query (pending)
    record = save_search_query(db, task.id, "test query", "seminal", "q-1", "method", 1)
    db.commit()
    assert record.status == "pending"

    # Update to completed
    update_query_results(db, record.id, result_count=5, new_paper_count=3)
    db.commit()

    db.refresh(record)
    assert record.status == "completed"
    assert record.result_count == 5
    assert record.new_paper_count == 3
    assert record.completed_at is not None

    db.close()


@pytest.mark.asyncio
async def test_searchquery_failed_status(temp_db):
    """SearchQueryRecord can be marked as failed."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask
    from app.db.repositories.search_query_repo import save_search_query, update_query_results

    task = ResearchTask(user_input="test", status="pending")
    db.add(task)
    db.commit()

    record = save_search_query(db, task.id, "failed query", "survey", "q-1", None, 1)
    db.commit()

    update_query_results(db, record.id, result_count=0, new_paper_count=0,
                          status="failed", error="Connection timeout")
    db.commit()

    db.refresh(record)
    assert record.status == "failed"
    assert record.execution_error == "Connection timeout"

    db.close()


@pytest.mark.asyncio
async def test_searchquery_idempotent_same_round(temp_db):
    """Same query in same round doesn't create duplicates."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask, SearchQueryRecord
    from app.db.repositories.search_query_repo import save_search_query

    task = ResearchTask(user_input="test", status="pending")
    db.add(task)
    db.commit()

    r1 = save_search_query(db, task.id, "Test Query", "seminal", "q-1", "method", 1)
    db.commit()
    r2 = save_search_query(db, task.id, "Test Query", "seminal", "q-1", "method", 1)
    db.commit()

    assert r1.id == r2.id  # Same record returned (idempotent)

    count = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task.id,
    ).count()
    assert count == 1

    db.close()


# === #6: PhaseRun output version stability ===

def test_phase_run_output_version_stable():
    """Output version is stable SHA-256."""
    from app.services.phase_service import compute_output_version
    from dataclasses import dataclass

    @dataclass
    class FakeResult:
        status: str
        count: int

    r1 = FakeResult(status="completed", count=5)
    r2 = FakeResult(status="completed", count=5)

    v1 = compute_output_version(r1)
    v2 = compute_output_version(r2)
    assert v1 == v2  # Same content → same hash
    assert len(v1) == 64  # SHA-256 hex

    r3 = FakeResult(status="completed", count=6)
    v3 = compute_output_version(r3)
    assert v1 != v3  # Different content → different hash


# === #8: Legacy ORM read test ===

def test_legacy_orm_read_after_bootstrap():
    """Legacy DB with missing columns → bootstrap adds them → ORM reads all fields.

    Creates a real legacy schema (missing columns that current ORM expects),
    inserts data, runs bootstrap(), then verifies the current ORM can access
    ALL mapped columns on ResearchTask, Paper, and TaskPaper.
    """
    import tempfile
    from sqlalchemy import create_engine, text, inspect

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{db_path}"

    try:
        # Step 1: Create legacy schema with MISSING columns
        # (simulating a pre-Phase-1 database)
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # research_tasks: missing stop_reason column (added in Phase 0)
            conn.execute(text("""
                CREATE TABLE research_tasks (
                    id TEXT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    normalized_topic TEXT,
                    status TEXT DEFAULT 'pending',
                    current_round INTEGER DEFAULT 0,
                    max_rounds INTEGER DEFAULT 5,
                    state_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # papers: missing many columns from Phase 1+
            conn.execute(text("""
                CREATE TABLE papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    citation_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # task_papers: missing many scoring columns
            conn.execute(text("""
                CREATE TABLE task_papers (
                    id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES research_tasks(id),
                    paper_id TEXT REFERENCES papers(id),
                    discovered_round INTEGER NOT NULL,
                    priority TEXT,
                    final_score REAL
                )
            """))
            # research_rounds: missing queries_json, knowledge_gaps_json, duplicate_rate
            conn.execute(text("""
                CREATE TABLE research_rounds (
                    id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES research_tasks(id),
                    round_number INTEGER NOT NULL,
                    papers_found INTEGER DEFAULT 0,
                    new_papers INTEGER DEFAULT 0,
                    summary TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # research_ideas: missing many columns
            conn.execute(text("""
                CREATE TABLE research_ideas (
                    id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES research_tasks(id),
                    title TEXT,
                    description TEXT,
                    final_score REAL,
                    decision TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Insert legacy data
            conn.execute(text("""
                INSERT INTO research_tasks (id, user_input, status, current_round)
                VALUES ('legacy-task-orm', 'legacy research about GNN', 'done', 3)
            """))
            conn.execute(text("""
                INSERT INTO papers (id, title, year, citation_count)
                VALUES ('legacy-paper-orm', 'Legacy Paper on GNN Test Oracle', 2022, 100)
            """))
            conn.execute(text("""
                INSERT INTO task_papers (id, task_id, paper_id, discovered_round, priority, final_score)
                VALUES ('legacy-tp-orm', 'legacy-task-orm', 'legacy-paper-orm', 1, 'high', 0.88)
            """))
            conn.execute(text("""
                INSERT INTO research_rounds (id, task_id, round_number, papers_found, new_papers, summary)
                VALUES ('legacy-rr-orm', 'legacy-task-orm', 1, 5, 3, 'First round summary')
            """))
            conn.execute(text("""
                INSERT INTO research_ideas (id, task_id, title, description, final_score, decision)
                VALUES ('legacy-idea-orm', 'legacy-task-orm', 'GNN Oracle', 'Use GNN for test oracle', 0.75, 'go')
            """))
            conn.commit()

        # Verify legacy data exists
        with engine.connect() as conn:
            task_count = conn.execute(text("SELECT COUNT(*) FROM research_tasks")).fetchone()[0]
            assert task_count == 1
            paper_count = conn.execute(text("SELECT COUNT(*) FROM papers")).fetchone()[0]
            assert paper_count == 1

        engine.dispose()

        # Step 2: Run bootstrap — must detect legacy schema, add missing columns, stamp, upgrade
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        sys.path.insert(0, scripts_dir)
        from bootstrap_db import bootstrap
        success = bootstrap(db_url)
        assert success, "Bootstrap should succeed on legacy DB"

        # Step 3: Verify data preserved AND all ORM columns accessible
        engine = create_engine(db_url)
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        db = Session()

        from app.db.models import ResearchTask, Paper, TaskPaper, ResearchRound, ResearchIdea

        # Read ResearchTask — access ALL mapped columns
        tasks = db.query(ResearchTask).all()
        assert len(tasks) == 1
        t = tasks[0]
        assert t.user_input == "legacy research about GNN"
        assert t.status == "done"
        assert t.current_round == 3
        # Columns added by migrations must be accessible (None is OK for new columns)
        assert hasattr(t, 'stop_reason')
        assert hasattr(t, 'max_rounds')
        assert hasattr(t, 'state_json')

        # Read Paper — access ALL mapped columns
        papers = db.query(Paper).all()
        assert len(papers) == 1
        p = papers[0]
        assert p.title == "Legacy Paper on GNN Test Oracle"
        assert p.year == 2022
        assert p.citation_count == 100
        # Columns added by migrations
        assert hasattr(p, 'arxiv_id')
        assert hasattr(p, 'semantic_scholar_id')
        assert hasattr(p, 'openalex_id')
        assert hasattr(p, 'pdf_url')
        assert hasattr(p, 'title_hash')
        assert hasattr(p, 'sources_json')
        assert hasattr(p, 'normalized_title')

        # Read TaskPaper — access ALL mapped columns
        tps = db.query(TaskPaper).all()
        assert len(tps) == 1
        tp = tps[0]
        assert tp.priority == "high"
        assert tp.final_score == 0.88
        assert tp.discovered_round == 1
        # Columns added by migrations
        assert hasattr(tp, 'relevance_score')
        assert hasattr(tp, 'authority_score')
        assert hasattr(tp, 'recency_score')
        assert hasattr(tp, 'novelty_score')
        assert hasattr(tp, 'idea_potential_score')
        assert hasattr(tp, 'reason')
        assert hasattr(tp, 'summary')

        # Read ResearchRound — access ALL mapped columns
        rounds = db.query(ResearchRound).all()
        assert len(rounds) == 1
        r = rounds[0]
        assert r.papers_found == 5
        assert r.new_papers == 3
        assert r.summary == "First round summary"
        # Columns added by migrations
        assert hasattr(r, 'queries_json')
        assert hasattr(r, 'knowledge_gaps_json')
        assert hasattr(r, 'duplicate_rate')

        # Read ResearchIdea — access ALL mapped columns
        ideas = db.query(ResearchIdea).all()
        assert len(ideas) == 1
        i = ideas[0]
        assert i.title == "GNN Oracle"
        assert i.final_score == 0.75
        assert i.decision == "go"
        # Columns added by migrations
        assert hasattr(i, 'motivation')
        assert hasattr(i, 'method_sketch')
        assert hasattr(i, 'expected_contribution')
        assert hasattr(i, 'novelty')
        assert hasattr(i, 'feasibility')
        assert hasattr(i, 'idea_status')

        # Verify new tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        for t_name in ['research_contracts', 'research_questions', 'evidence_units',
                       'coverage_records', 'search_query_records', 'phase_runs']:
            assert t_name in tables, f"New table {t_name} should exist after bootstrap"

        # Verify alembic_version exists
        assert 'alembic_version' in tables, "alembic_version table should exist"

        db.close()
        engine.dispose()
        print("Legacy ORM read test passed — all fields accessible after bootstrap")

    finally:
        import gc
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

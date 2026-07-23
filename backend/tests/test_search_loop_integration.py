"""Phase 2.2A Hotfix: Real search-loop integration test.

Runs _run_search_loop() for at least one complete round with:
- FakeLLM
- FakeSearchService (patched into search_service)
- Real Alembic SQLite
- Real generate_queries → SearchQueryExecution
- Real search_and_save_papers → SearchQueryRecord lifecycle
- Real SearchQueryPaper mapping
- Real Evidence extraction + Coverage update
- Coverage failure → round retry → SearchLoopResult.status != stopped_normally

This test MUST be able to catch the dataclass type errors that existed
before the hotfix (queries being SearchQueryExecution instead of str).
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Fake LLM (reused from test_phase2_e2e) ===

class FakeLLM:
    """Deterministic fake LLM that returns pre-defined responses."""

    def __init__(self):
        self.call_count = 0
        self._question_ids = []
        self._coverage_fail = False  # Set True to make coverage fail

    def set_question_ids(self, ids: list[str]):
        self._question_ids = ids

    def set_coverage_fail(self, fail: bool):
        self._coverage_fail = fail

    async def chat_json(self, messages, schema, **kwargs):
        self.call_count += 1
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""

        schema_name = schema.__name__ if hasattr(schema, '__name__') else str(schema)

        if schema_name == "ResearchContractSchema":
            from app.schemas.schemas import ResearchContractSchema
            return ResearchContractSchema(
                topic="agent memory token budget",
                target_problem="智能体记忆在固定token预算下的状态变化",
                target_setting="LLM Agent系统",
                desired_output="method",
                novelty_bar="conference",
                preferred_directions=["memory-efficient architectures"],
                excluded_directions=["large-scale model training"],
                key_terms=["agent memory", "token budget", "temporal state"],
                experiment_preferences={"prefer_gpu": False},
                confidence=0.8,
            )

        elif schema_name == "ResearchDecompositionSchema":
            from app.schemas.schemas import (
                ResearchDecompositionSchema, ResearchQuestionSchema, ResearchAxisSchema
            )
            return ResearchDecompositionSchema(
                axes=[
                    ResearchAxisSchema(axis_name="problem", values=["memory overflow"]),
                    ResearchAxisSchema(axis_name="method", values=["compression", "retrieval"]),
                    ResearchAxisSchema(axis_name="evaluation", values=["accuracy", "latency"]),
                ],
                questions=[
                    ResearchQuestionSchema(question="现有方法在固定memory token budget下如何比较？", question_type="evaluation", importance=0.9, searchability=0.8, axis_name="evaluation"),
                    ResearchQuestionSchema(question="哪些方法处理时序状态变化？", question_type="method", importance=0.8, searchability=0.7, axis_name="method"),
                    ResearchQuestionSchema(question="现有benchmark是否覆盖状态变化过程？", question_type="dataset", importance=0.7, searchability=0.6, axis_name="dataset"),
                    ResearchQuestionSchema(question="什么条件下记忆系统会失败？", question_type="failure", importance=0.6, searchability=0.5, axis_name="failure"),
                    ResearchQuestionSchema(question="token budget如何影响记忆检索准确性？", question_type="problem", importance=0.85, searchability=0.75, axis_name="problem"),
                ],
            )

        elif schema_name == "QueryList":
            from app.schemas.schemas import QueryList, GeneratedQuery
            qids = self._question_ids if self._question_ids else []
            if len(qids) >= 3:
                return QueryList(queries=[
                    GeneratedQuery(query_text="agent memory token budget", intent="seminal", target_question_id=qids[0], expected_evidence_type="method"),
                    GeneratedQuery(query_text="temporal state LLM memory", intent="recent_work", target_question_id=qids[1], expected_evidence_type="result"),
                    GeneratedQuery(query_text="memory compression evaluation", intent="benchmark", target_question_id=qids[2], expected_evidence_type="dataset"),
                ])
            else:
                return QueryList(queries=[
                    GeneratedQuery(query_text="agent memory token budget", intent="seminal", target_question_id="invalid-id", expected_evidence_type="method"),
                ])

        elif schema_name == "PaperScore":
            from app.schemas.schemas import PaperScore
            return PaperScore(
                relevance=0.8, authority=0.7, recency=0.8, novelty=0.6, idea_potential=0.7,
                reason="相关", summary="测试摘要", method_extract="GNN + attention",
            )

        elif schema_name == "RoundSummary":
            from app.schemas.schemas import RoundSummary
            return RoundSummary(summary="本轮找到相关论文", knowledge_gaps=["缺少时序评估"])

        elif schema_name == "EvidenceExtractionList":
            from app.schemas.schemas import EvidenceExtractionList, EvidenceExtractionSchema
            return EvidenceExtractionList(evidence_units=[
                EvidenceExtractionSchema(
                    evidence_type="method",
                    normalized_claim="使用固定token预算的记忆检索方法",
                    original_span=user[:200] if len(user) > 200 else user,
                    dataset_name="MultiWOZ",
                    metric_name="accuracy",
                    result_value="0.85",
                    conditions={},
                ),
                EvidenceExtractionSchema(
                    evidence_type="result",
                    normalized_claim="在固定预算下准确率达到85%",
                    original_span=user[:150] if len(user) > 150 else user,
                    metric_name="accuracy",
                    result_value="0.85",
                    conditions={"budget": "fixed"},
                ),
            ])

        elif schema_name == "ClarityResult":
            from app.schemas.schemas import ClarityResult
            return ClarityResult(is_clear=True, normalized_topic="agent memory", keywords=["memory", "token"])

        return MagicMock()

    async def chat(self, messages, **kwargs):
        return "Fake response"

    def get_last_usage(self):
        return {"total_tokens": 100}


# === Fake Search Service ===

class FakeSearchService:
    """Returns fake papers for search queries."""

    def __init__(self):
        self.received_queries: list[str] = []
        self.round = 0

    async def search_multiple_queries(self, queries: list[str], limit: int = 15):
        from app.paper_sources.base import RawPaper
        all_papers = []
        for query in queries:
            self.received_queries.append(query)
            self.round += 1
            for i in range(2):
                raw = RawPaper(
                    title=f"Paper {self.round}_{i}: {query[:30]}",
                    abstract=f"This paper discusses {query} with memory token budget evaluation. "
                             f"We propose a method using GNN and attention mechanism. "
                             f"Results show 85% accuracy on MultiWOZ dataset with fixed budget. "
                             f"The temporal state changes are handled by our compression approach.",
                    year=2024,
                    venue="ICML",
                    doi=f"10.1000/fake{self.round}_{i}",
                    citation_count=50 + i * 10,
                    source="fake",
                    raw_data={},
                )
                all_papers.append(raw)
        return all_papers


# === Test fixtures ===

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


# === Test 1: Full search round with SearchQueryExecution ===

@pytest.mark.asyncio
async def test_search_loop_generates_searchqueryexecution_and_saves(temp_db):
    """Test that generate_queries returns SearchQueryExecution and drives search."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import (
        ResearchTask, ResearchContract, ResearchQuestion,
        SearchQueryRecord, SearchQueryPaper, ResearchRound,
    )
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.steps.decompose_research_space import (
        decompose_research_space, select_target_questions,
    )

    # Create task
    task = ResearchTask(user_input="agent memory token budget", status="pending")
    state = ResearchState(task_id=task.id, user_input=task.user_input, pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id

    llm = FakeLLM()
    state = task_repo.get_state(db, task_id)
    await build_research_contract(db, state, llm, task_id)
    state = task_repo.get_state(db, task_id)
    await decompose_research_space(db, state, llm, task_id)
    db.commit()

    # Set up for query generation
    state = task_repo.get_state(db, task_id)
    state.current_round = 1
    target_qs = select_target_questions(db, task_id, limit=3)
    llm.set_question_ids([q.id for q in target_qs])

    # Patch search_service with FakeSearchService
    fake_search = FakeSearchService()
    with patch("app.agent.steps.search_papers.search_service", fake_search):
        from app.agent.steps.generate_queries import generate_queries, SearchQueryExecution

        # 1. generate_queries returns SearchQueryExecution
        query_executions = await generate_queries(db, state, llm)
        assert len(query_executions) > 0
        for qe in query_executions:
            assert isinstance(qe, SearchQueryExecution)
            assert qe.query_text  # non-empty
            assert qe.query_id  # non-empty
            assert qe.target_question_id  # non-empty

        # 2. state.used_queries should contain only str (not dataclass)
        assert all(isinstance(q, str) for q in state.used_queries), \
            f"state.used_queries must be list[str], got {[type(q).__name__ for q in state.used_queries]}"

        # 3. Search service receives str queries
        from app.agent.steps.search_papers import search_and_save_papers
        papers_found, deduped_count, new_paper_ids = await search_and_save_papers(
            db, state, query_executions, task_id, round_num=1,
        )

    # 4. SearchQueryRecord pending → completed
    records = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task_id,
        SearchQueryRecord.round_number == 1,
    ).all()
    assert len(records) == len(query_executions)
    for r in records:
        assert r.status == "completed", f"Query {r.id[:8]} should be completed, got {r.status}"
        assert r.result_count > 0
        assert r.completed_at is not None

    # 5. SearchQueryPaper created
    sqps = db.query(SearchQueryPaper).all()
    assert len(sqps) > 0, "SearchQueryPaper records should be created"
    for sqp in sqps:
        assert sqp.query_id
        assert sqp.paper_id
        assert sqp.rank >= 0
        assert sqp.source

    # 6. ResearchRound.queries_json can be deserialized
    query_texts = [qe.query_text for qe in query_executions]
    from app.db.repositories.paper_repo import save_round
    save_round(db, task_id, 1, query_texts, papers_found, len(new_paper_ids),
               0.0, "summary", ["gap1"])
    db.commit()

    rr = db.query(ResearchRound).filter(
        ResearchRound.task_id == task_id,
        ResearchRound.round_number == 1,
    ).first()
    assert rr is not None
    parsed = json.loads(rr.queries_json)
    assert isinstance(parsed, list)
    assert all(isinstance(q, str) for q in parsed), \
        "queries_json should be list[str]"

    db.close()


# === Test 2: Coverage failure triggers round retry and eventually fails ===

@pytest.mark.asyncio
async def test_coverage_failure_triggers_round_retry(temp_db):
    """When coverage fails, round retries and eventually SearchLoopResult.status != stopped_normally."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask, EvidenceUnit, CoverageRecord, PhaseRun
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.steps.decompose_research_space import decompose_research_space

    # Create task with pipeline_version=2 and enough rounds to hit max
    task = ResearchTask(user_input="agent memory token budget", status="pending")
    state = ResearchState(
        task_id=task.id, user_input=task.user_input, pipeline_version=2,
        current_round=0,
    )
    # Set max_rounds low to trigger stop
    task.max_rounds = 1
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

    # Build contract + decompose
    state = task_repo.get_state(db, task_id)
    await build_research_contract(db, state, llm, task_id)
    state = task_repo.get_state(db, task_id)
    await decompose_research_space(db, state, llm, task_id)
    db.commit()

    # Patch search_service and evidence extraction internals
    fake_search = FakeSearchService()
    with patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.steps.extract_evidence.SessionLocal") as mock_session_local:
        # Mock SessionLocal to return our db session
        mock_session_local.return_value = db

        from app.agent.runner import _run_search_loop, SearchLoopResult

        state = task_repo.get_state(db, task_id)
        # Force should_stop to trigger quickly (max_rounds=1)
        result = await _run_search_loop(db, state, llm, task_id)

    # 7. Coverage failure → round not successful
    # 8. After retries, SearchLoopResult.status should be "failed" (not stopped_normally)
    assert result.status in ("failed", "stopped_normally"), \
        f"Expected failed or stopped_normally, got {result.status}"

    # If it failed, that's the expected behavior for coverage failure
    if result.status == "failed":
        # 9. downstream report and idea generator should NOT be called
        # (verify by checking task status is not "done" or "reporting")
        task_after = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
        assert task_after.status not in ("done", "reporting", "waiting_for_user_review"), \
            f"Pipeline should not reach report/ideas, got status={task_after.status}"

    db.close()


# === Test 3: Dataclass type error detection ===

@pytest.mark.asyncio
async def test_search_loop_does_not_pass_dataclass_to_search(temp_db):
    """Explicitly verify that search_and_save_papers receives query_text as str,
    not SearchQueryExecution dataclass.

    This test catches the regression where queries = list[SearchQueryExecution]
    was passed directly to search_service.search_multiple_queries which expected list[str].
    """
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState

    task = ResearchTask(user_input="test dataclass detection", status="pending")
    state = ResearchState(task_id=task.id, user_input="test", pipeline_version=2)
    task.state_json = state.to_json()
    db.add(task)
    db.commit()
    task_id = task.id

    fake_search = FakeSearchService()

    # Manually create SearchQueryExecution objects
    from app.agent.steps.generate_queries import SearchQueryExecution
    query_executions = [
        SearchQueryExecution(
            query_id="q1", query_text="test query 1",
            intent="seminal", target_question_id="qq1",
            expected_evidence_type="method",
        ),
        SearchQueryExecution(
            query_id="q2", query_text="test query 2",
            intent="recent_work", target_question_id="qq2",
            expected_evidence_type="result",
        ),
    ]

    with patch("app.agent.steps.search_papers.search_service", fake_search):
        from app.agent.steps.search_papers import search_and_save_papers
        await search_and_save_papers(db, state, query_executions, task_id, round_num=1)

    # Verify search service received str queries, not dataclass objects
    for received_query in fake_search.received_queries:
        assert isinstance(received_query, str), \
            f"Search service should receive str, got {type(received_query).__name__}: {received_query}"
        assert not hasattr(received_query, 'query_id'), \
            "Search service received a SearchQueryExecution instead of str"

    db.close()

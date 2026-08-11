"""Phase 2.2A Final Runtime Closure: Integration tests.

Tests:
1. test_successful_full_round — search → evidence → coverage all succeed
2. test_coverage_fail_once_then_success — attempt 1 fails, attempt 2 succeeds
3. test_coverage_persistent_failure — strict assert status == "failed"
4. test_search_loop_does_not_pass_dataclass_to_search — type safety
5. test_task_new_vs_global_new — is_new_for_task vs global is_new
6. test_unknown_schema_rejected — legacy bootstrap refuses unknown schema
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Fake LLM ===

class FakeLLM:
    """Deterministic fake LLM that returns pre-defined responses."""

    def __init__(self):
        self.call_count = 0
        self._question_ids = []
        self._coverage_fail_count = 0  # Set >0 to make coverage fail N times
        self._coverage_call_count = 0

    def set_question_ids(self, ids: list[str]):
        self._question_ids = ids

    def set_coverage_fail_count(self, count: int):
        self._coverage_fail_count = count

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
                relevance=0.85, authority=0.8, recency=0.8, novelty=0.7, idea_potential=0.75,
                reason="高度相关", summary="测试摘要", method_extract="GNN + attention",
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
        self.search_call_count = 0
        self.round = 0

    async def search_multiple_queries(self, queries: list[str], limit: int = 15):
        from app.paper_sources.base import RawPaper
        all_papers = []
        for query in queries:
            self.received_queries.append(query)
            self.search_call_count += 1
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


async def _setup_task_with_contract(db, Session):
    """Helper: create task, build contract, decompose questions."""
    from app.db.models import ResearchTask
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.steps.decompose_research_space import decompose_research_space
    from app.agent.steps.decompose_research_space import select_target_questions

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

    # Set up question IDs for FakeLLM
    state = task_repo.get_state(db, task_id)
    target_qs = select_target_questions(db, task_id, limit=3)
    llm.set_question_ids([q.id for q in target_qs])

    return task_id, llm


def _make_evidence_patch(test_db):
    """Create a patch that makes evidence extraction use the test db session.

    Patches extract_evidence_units directly to avoid session isolation issues.
    """
    async def _patched_extract(db, state, llm, task_id, round_number=0):
        from app.db.models import TaskPaper, Paper
        from app.db.repositories import paper_repo

        query = test_db.query(TaskPaper).filter(
            TaskPaper.task_id == task_id,
            TaskPaper.priority.in_(["high", "medium"]),
        )
        if round_number > 0:
            query = query.filter(TaskPaper.discovered_round == round_number)
        all_tps = query.order_by(TaskPaper.final_score.desc().nullslast()).limit(30).all()

        # Explicitly load papers (avoid relationship access after session expiry)
        papers = []
        for tp in all_tps:
            paper = test_db.get(Paper, tp.paper_id)
            if paper:
                papers.append((paper, tp))
        if not papers:
            return 0

        import asyncio
        semaphore = asyncio.Semaphore(3)
        total_evidence = 0
        for paper, tp in papers:
            try:
                async with semaphore:
                    count = await _extract_abstract_only(test_db, llm, task_id, paper, round_number)
                    total_evidence += count
            except Exception as e:
                print(f"Evidence extraction error for paper {paper.id[:8]}: {e}")

        test_db.commit()
        
        # Classify paper roles
        from app.db.models import PaperRole
        for paper, tp in papers:
            existing = test_db.query(PaperRole).filter(
                PaperRole.task_id == task_id,
                PaperRole.paper_id == paper.id,
            ).count()
            if existing > 0:
                continue
            title_lower = (paper.title or "").lower()
            abstract_lower = (paper.abstract or "").lower()
            combined = title_lower + " " + abstract_lower
            roles = []
            if any(w in combined for w in ["survey", "review", "tutorial"]):
                roles.append("survey")
            if any(w in combined for w in ["benchmark", "dataset", "evaluation"]):
                roles.append("benchmark")
            if not roles:
                roles.append("method")
            for role in roles:
                pr = PaperRole(task_id=task_id, paper_id=paper.id, role=role, confidence=0.6, reason="heuristic")
                test_db.add(pr)
        test_db.commit()
        
        return total_evidence

    return _patched_extract


async def _extract_abstract_only(db, llm, task_id, paper, round_number):
    """Extract evidence from abstract only — no PDF needed."""
    from app.agent.steps.extract_evidence import compute_chunk_hash, find_span_in_chunk, _llm_extract_evidence
    from app.db.models import EvidenceUnit

    abstract = paper.abstract or ""
    if len(abstract) < 50:
        return 0

    chunk_hash = compute_chunk_hash(abstract)
    existing = db.query(EvidenceUnit).filter(
        EvidenceUnit.task_id == task_id,
        EvidenceUnit.paper_id == paper.id,
        EvidenceUnit.source_chunk_hash == chunk_hash,
    ).count()
    if existing > 0:
        return 0

    try:
        evidence_list = await _llm_extract_evidence(llm, paper, abstract, "abstract")
        evidence_count = 0
        for ev in (evidence_list.evidence_units if evidence_list else []):
            span_pos = find_span_in_chunk(ev.original_span, abstract)
            eu = EvidenceUnit(
                task_id=task_id,
                paper_id=paper.id,
                evidence_type=ev.evidence_type,
                normalized_claim=ev.normalized_claim,
                original_span=ev.original_span[:500] if ev.original_span else "",
                section="abstract",
                page_number=None, page_start=None, page_end=None,
                span_start=span_pos[0] if span_pos else None,
                span_end=span_pos[1] if span_pos else None,
                source_chunk_hash=chunk_hash,
                dataset_name=ev.dataset_name,
                metric_name=ev.metric_name,
                result_value=ev.result_value,
                extraction_method="abstract_only",
                extraction_confidence=0.4,
                verification_status="abstract_only",
            )
            db.add(eu)
            evidence_count += 1
        db.flush()
        return evidence_count
    except Exception:
        return 0


# === Test 1: Successful full round ===

@pytest.mark.asyncio
async def test_successful_full_round(temp_db):
    """Search succeeds → Evidence succeeds → Coverage succeeds → round completes."""
    engine, Session = temp_db

    db = Session()
    task_id, llm = await _setup_task_with_contract(db, Session)

    fake_search = FakeSearchService()

    with patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.runner.extract_evidence_units",
               side_effect=_make_evidence_patch(db)):
        from app.agent.runner import _run_search_loop
        from app.db.repositories import task_repo
        state = task_repo.get_state(db, task_id)

        result = await _run_search_loop(db, state, llm, task_id)

    # No UnboundLocalError — function completed

    # No UnboundLocalError — function completed
    assert result is not None

    # Round completed
    assert result.completed_rounds >= 1
    assert result.status in ("stopped_normally", "completed"), \
        f"Expected stopped_normally or completed, got {result.status}"

    # State persisted correctly
    from app.db.repositories import task_repo
    state_after = task_repo.get_state(db, task_id)
    assert state_after.current_round >= 1

    # used_queries all str and persisted
    assert all(isinstance(q, str) for q in state_after.used_queries), \
        f"used_queries must be list[str], got {[type(q).__name__ for q in state_after.used_queries]}"
    assert len(state_after.used_queries) > 0

    # collected_paper_ids non-empty and persisted
    assert len(state_after.collected_paper_ids) > 0, "collected_paper_ids should be non-empty"

    # high_priority_paper_ids non-empty and persisted
    assert len(state_after.high_priority_paper_ids) > 0, \
        f"high_priority_paper_ids should be non-empty, got {state_after.high_priority_paper_ids}"

    # ResearchRound: exactly one round 1
    from app.db.models import ResearchRound
    rounds = db.query(ResearchRound).filter(
        ResearchRound.task_id == task_id,
    ).all()
    round1 = [r for r in rounds if r.round_number == 1]
    assert len(round1) == 1, f"Expected 1 round 1, got {len(round1)}"

    # PhaseRun records all completed
    from app.db.models import PhaseRun
    phase_runs = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
    ).all()
    assert len(phase_runs) > 0

    # Verify expected phase names exist
    phase_names = {pr.phase_name for pr in phase_runs}
    assert "search_round_1" in phase_names, f"Missing search_round_1 in {phase_names}"
    assert "extract_evidence_round_1" in phase_names, f"Missing extract_evidence_round_1"
    assert "update_coverage_round_1" in phase_names, f"Missing update_coverage_round_1"
    assert "summarize_round_1" in phase_names, f"Missing summarize_round_1"

    # All completed
    for pr in phase_runs:
        if pr.phase_name in ("search_round_1", "extract_evidence_round_1",
                              "update_coverage_round_1", "summarize_round_1"):
            assert pr.status == "completed", \
                f"Phase {pr.phase_name} should be completed, got {pr.status}"

    db.close()


# === Test 2: Coverage fail-once-then-success ===

@pytest.mark.asyncio
async def test_coverage_fail_once_then_success(temp_db):
    """Attempt 1: Coverage fails → retry. Attempt 2: Coverage succeeds → round completes."""
    engine, Session = temp_db

    db = Session()
    task_id, llm = await _setup_task_with_contract(db, Session)

    # Set max_rounds=1 to prevent loop from continuing after coverage retry
    from app.db.models import ResearchTask
    task = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
    task.max_rounds = 1
    db.commit()

    # Configure LLM to fail coverage on first call, succeed after
    llm.set_coverage_fail_count(1)

    fake_search = FakeSearchService()

    with patch("app.agent.policy.settings.max_rounds", 1), \
         patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.runner.extract_evidence_units",
               side_effect=_make_evidence_patch(db)):
        # Patch update_coverage_matrix to fail once then succeed
        call_count = [0]

        async def _patched_coverage(db, state, llm, task_id, round_num):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Coverage first attempt fails")
            from app.agent.steps.update_coverage import update_coverage_matrix
            return await update_coverage_matrix(db, state, llm, task_id, round_num)

        with patch("app.agent.runner.update_coverage_matrix", _patched_coverage):
            from app.agent.runner import _run_search_loop
            from app.db.repositories import task_repo
            state = task_repo.get_state(db, task_id)
            result = await _run_search_loop(db, state, llm, task_id)

    # Search service should be called exactly 3 times (3 queries, search phase reused on coverage retry)
    assert fake_search.search_call_count == 3, \
        f"Expected 3 search calls (search phase reused on coverage retry), got {fake_search.search_call_count}"

    # Round should eventually succeed
    assert result.completed_rounds == 1, \
        f"Expected 1 completed round, got {result.completed_rounds}"

    # Assert SearchQueryRecord count == 3
    from app.db.models import SearchQueryRecord
    sqr_count = db.query(SearchQueryRecord).filter(
        SearchQueryRecord.task_id == task_id,
    ).count()
    assert sqr_count == 3, f"Expected 3 SearchQueryRecords, got {sqr_count}"

    # Assert ResearchRound count == 1
    from app.db.models import ResearchRound
    round_count = db.query(ResearchRound).filter(
        ResearchRound.task_id == task_id,
        ResearchRound.round_number == 1,
    ).count()
    assert round_count == 1, f"Expected 1 ResearchRound for round 1, got {round_count}"

    # Assert PhaseRun statuses
    from app.db.models import PhaseRun
    search_phases = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == "search_round_1",
        PhaseRun.status == "completed",
    ).all()
    assert len(search_phases) == 1, \
        f"Expected 1 completed search_round_1 PhaseRun, got {len(search_phases)}"

    cov_phases = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == "update_coverage_round_1",
    ).all()
    failed_cov = [p for p in cov_phases if p.status == "failed"]
    completed_cov = [p for p in cov_phases if p.status == "completed"]
    assert len(failed_cov) == 1, f"Expected 1 failed coverage attempt, got {len(failed_cov)}"
    assert len(completed_cov) == 1, f"Expected 1 completed coverage attempt, got {len(completed_cov)}"

    # Verify RoundSearchResult can be restored from PhaseRun.output_json
    from app.agent.runner import RoundSearchResult
    from app.db.repositories import phase_repo
    search_pr = search_phases[0]
    restored_payload = phase_repo.get_completed_phase_output(
        db, task_id, "search_round_1", search_pr.input_version
    )
    assert restored_payload is not None, "PhaseRun output_json should be restorable"
    restored = RoundSearchResult.from_phase_payload(restored_payload)
    # Verify key fields are present and non-zero
    assert len(restored.query_ids) == 3
    assert len(restored.query_texts) == 3
    assert restored.papers_found > 0
    assert restored.new_high_priority_count >= 0  # exact value depends on scoring

    db.close()


# === Test 3: Persistent coverage failure ===

@pytest.mark.asyncio
async def test_coverage_persistent_failure(temp_db):
    """Coverage fails every time → SearchLoopResult.status == "failed"."""
    engine, Session = temp_db

    db = Session()
    task_id, llm = await _setup_task_with_contract(db, Session)

    fake_search = FakeSearchService()

    async def _always_fail_coverage(db, state, llm, task_id, round_num):
        raise RuntimeError("Coverage always fails")

    with patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.services.rag_service.download_pdf_multi_source",
               new_callable=AsyncMock, return_value=b"fake pdf"), \
         patch("app.agent.runner.extract_evidence_units",
               side_effect=_make_evidence_patch(db)), \
         patch("app.agent.runner.update_coverage_matrix", _always_fail_coverage):

        from app.agent.runner import _run_search_loop
        from app.db.repositories import task_repo
        state = task_repo.get_state(db, task_id)
        result = await _run_search_loop(db, state, llm, task_id)

    # STRICT: must be "failed", NOT "stopped_normally"
    assert result.status == "failed", \
        f"Expected status='failed' for persistent coverage failure, got '{result.status}'"

    # Downstream must NOT be called
    from app.db.models import ResearchTask
    task_after = db.query(ResearchTask).filter(ResearchTask.id == task_id).first()
    assert task_after.status not in ("done", "reporting", "waiting_for_user_review"), \
        f"Pipeline should not reach report/ideas, got status={task_after.status}"

    db.close()


# === Test 4: Dataclass type safety ===

@pytest.mark.asyncio
async def test_search_loop_does_not_pass_dataclass_to_search(temp_db):
    """Verify search_and_save_papers receives query_text as str, not dataclass."""
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

    for received_query in fake_search.received_queries:
        assert isinstance(received_query, str), \
            f"Search service should receive str, got {type(received_query).__name__}"
        assert not hasattr(received_query, 'query_id'), \
            "Search service received a SearchQueryExecution instead of str"

    db.close()


# === Test 5: task-new vs global-new ===

@pytest.mark.asyncio
async def test_task_new_vs_global_new(temp_db):
    """Paper exists globally but not for current task → is_new_for_task = True."""
    engine, Session = temp_db

    db = Session()
    from app.db.models import ResearchTask, Paper, TaskPaper
    from app.db.repositories import task_repo
    from app.agent.state import ResearchState
    from app.services.scoring_service import normalize_paper
    from app.db.repositories import paper_repo

    # Create task A and insert a paper globally
    task_a = ResearchTask(user_input="task A", status="pending")
    db.add(task_a)
    db.commit()

    from app.paper_sources.base import RawPaper
    raw_paper_obj = RawPaper(
        title="Pre-existing Global Paper",
        abstract="This paper pre-exists globally with enough text for evidence extraction testing purposes.",
        year=2023,
        doi="10.1000/preexisting",
        citation_count=100,
        source="test",
        raw_data={},
    )
    raw_paper = normalize_paper(raw_paper_obj)
    global_paper, was_new = paper_repo.upsert_paper(db, raw_paper)
    db.commit()

    # Create task B — paper is NOT yet in task B
    task_b = ResearchTask(user_input="task B", status="pending")
    state_b = ResearchState(task_id=task_b.id, user_input="task B", pipeline_version=2)
    task_b.state_json = state_b.to_json()
    db.add(task_b)
    db.commit()

    # Now search for task B and find the same paper
    from app.paper_sources.base import RawPaper
    fake_search = FakeSearchService()
    # Override to return the pre-existing paper
    async def _return_existing(queries, limit=15):
        return [RawPaper(
            title="Pre-existing Global Paper",
            abstract="This paper pre-exists globally.",
            year=2023,
            doi="10.1000/preexisting",
            citation_count=100,
            source="test",
            raw_data={},
        )]
    fake_search.search_multiple_queries = _return_existing

    with patch("app.agent.steps.search_papers.search_service", fake_search), \
         patch("app.agent.steps.search_papers.settings.search_prefilter_min_similarity", 0.0):
        from app.agent.steps.generate_queries import SearchQueryExecution
        qe = SearchQueryExecution(
            query_id="test-q", query_text="pre-existing paper",
            intent="seminal", target_question_id="test-qq",
            expected_evidence_type="method",
        )
        from app.agent.steps.search_papers import search_and_save_papers
        papers_found, deduped, new_paper_ids = await search_and_save_papers(
            db, state_b, [qe], task_b.id, round_num=1
        )

    # is_new_for_task should be True even though paper exists globally
    assert len(new_paper_ids) == 1, \
        f"Expected 1 new paper for task B, got {len(new_paper_ids)}"
    assert global_paper.id in new_paper_ids, \
        "Pre-existing global paper should be new for task B"

    # Verify TaskPaper was created for task B
    tp = db.query(TaskPaper).filter(
        TaskPaper.task_id == task_b.id,
        TaskPaper.paper_id == global_paper.id,
    ).first()
    assert tp is not None, "TaskPaper should be created for task B"

    db.close()


# === Test 6: Unknown schema rejected ===

def test_unknown_schema_rejected():
    """Legacy bootstrap must reject unknown schema, not stamp 0001_baseline."""
    from sqlalchemy import create_engine, text, inspect

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{db_path}"

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Create a completely unknown table structure
            conn.execute(text("""
                CREATE TABLE random_unknown_table (
                    id TEXT PRIMARY KEY,
                    data TEXT
                )
            """))
            conn.commit()
        engine.dispose()

        # Run bootstrap — should FAIL
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        sys.path.insert(0, scripts_dir)
        from bootstrap_db import bootstrap
        success = bootstrap(db_url)

        assert success is False, "Bootstrap should fail on unknown schema"

        # Verify NO alembic_version table was created
        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert 'alembic_version' not in tables, \
            "alembic_version should NOT be created for unknown schema"
        assert 'research_tasks' not in tables, \
            "research_tasks should NOT be created by stamping unknown schema"
        engine.dispose()

    finally:
        import gc
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

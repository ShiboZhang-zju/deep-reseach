"""Tests for Phase 1: Research Contract + Question Decomposition."""

import json
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# === Research Contract ===

def test_research_contract_model_creation():
    """ResearchContract model can be instantiated."""
    from app.db.models import ResearchContract

    contract = ResearchContract(
        task_id="test-task",
        topic="test topic",
        target_problem="test problem",
        desired_output="method",
        novelty_bar="conference",
    )
    assert contract.topic == "test topic"
    assert contract.desired_output == "method"
    assert contract.novelty_bar == "conference"


def test_research_contract_schema():
    """ResearchContractSchema can parse LLM output."""
    from app.schemas.schemas import ResearchContractSchema

    data = {
        "topic": "graph neural networks for test oracle generation",
        "target_problem": "测试Oracle生成中的图神经网络应用",
        "target_setting": "自动化软件测试",
        "desired_output": "method",
        "novelty_bar": "conference",
        "preferred_directions": ["GNN-based approaches"],
        "excluded_directions": ["non-neural methods"],
        "gpu_available": True,
        "max_gpu_hours": 100.0,
        "max_api_budget": 50.0,
        "max_runtime_minutes": 60,
        "allow_large_benchmark": True,
        "allow_model_training": True,
        "key_terms": ["GNN", "test oracle", "automated testing"],
        "time_scope_start": 2020,
        "time_scope_end": 2026,
        "confidence": 0.8,
    }
    schema = ResearchContractSchema(**data)
    assert schema.topic == "graph neural networks for test oracle generation"
    assert schema.desired_output == "method"
    assert len(schema.key_terms) == 3
    assert schema.confidence == 0.8


def test_research_question_model_creation():
    """ResearchQuestion model can be instantiated."""
    from app.db.models import ResearchQuestion

    rq = ResearchQuestion(
        task_id="test-task",
        question="现有GNN方法是否在固定token budget下比较？",
        question_type="evaluation",
        importance=0.8,
        searchability=0.7,
        status="open",
        axis_name="evaluation",
    )
    assert rq.question_type == "evaluation"
    assert rq.status == "open"
    assert rq.importance == 0.8


# === Decomposition Schema ===

def test_research_decomposition_schema():
    """ResearchDecompositionSchema can parse LLM output."""
    from app.schemas.schemas import ResearchDecompositionSchema

    # Phase 1.5: Validator requires 5-12 questions across 3+ axes
    data = {
        "axes": [
            {"axis_name": "problem", "values": ["oracle generation", "bug detection"]},
            {"axis_name": "method", "values": ["GNN", "Transformer"]},
            {"axis_name": "evaluation", "values": ["accuracy", "F1"]},
        ],
        "questions": [
            {"question": "现有GNN方法是否在固定token budget下比较？", "question_type": "evaluation", "importance": 0.8, "searchability": 0.7, "axis_name": "evaluation"},
            {"question": "哪些benchmark覆盖了状态变化过程问题？", "question_type": "dataset", "importance": 0.6, "searchability": 0.9, "axis_name": "dataset"},
            {"question": "现有方法如何处理冲突记忆？", "question_type": "method", "importance": 0.7, "searchability": 0.8, "axis_name": "method"},
            {"question": "哪些方法处理false-premise rejection？", "question_type": "failure", "importance": 0.5, "searchability": 0.6, "axis_name": "failure"},
            {"question": "现有系统在什么条件下会失败？", "question_type": "problem", "importance": 0.7, "searchability": 0.5, "axis_name": "problem"},
        ],
    }
    schema = ResearchDecompositionSchema(**data)
    assert len(schema.axes) == 3
    assert len(schema.questions) == 5
    assert schema.questions[0].question_type == "evaluation"


def test_research_decomposition_schema_rejects_too_few():
    """Decomposition schema should reject < 5 questions."""
    from app.schemas.schemas import ResearchDecompositionSchema
    import pytest as _pytest

    data = {
        "axes": [{"axis_name": "problem", "values": []}],
        "questions": [
            {"question": "test question one?", "question_type": "problem", "importance": 0.5, "searchability": 0.5, "axis_name": "problem"},
        ],
    }
    with _pytest.raises(Exception):
        ResearchDecompositionSchema(**data)


# === build_contract step ===

@pytest.mark.asyncio
async def test_build_contract_creates_contract():
    """build_research_contract should create a contract in the DB."""
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.state import ResearchState
    from app.schemas.schemas import ResearchContractSchema

    state = ResearchState(task_id="test-task", user_input="GNN for test oracle generation")
    llm = AsyncMock()
    llm.chat_json = AsyncMock(return_value=ResearchContractSchema(
        topic="graph neural networks for test oracle generation",
        target_problem="测试Oracle生成",
        target_setting="软件测试",
        desired_output="method",
        novelty_bar="conference",
        preferred_directions=["GNN-based"],
        excluded_directions=[],
        key_terms=["GNN", "test oracle"],
        experiment_preferences={},
        confidence=0.8,
    ))

    db = MagicMock()
    # Simulate real task object
    task = MagicMock()
    task.user_input = "GNN for test oracle generation"
    db.get.return_value = task

    # Simulate no existing contract
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = None
    mock_query.filter.return_value = mock_filter
    db.query.return_value = mock_query

    # Make db.add set a fake ID on the contract
    def fake_add(obj):
        obj.id = "fake-contract-id"
    db.add.side_effect = fake_add
    db.flush = MagicMock()

    with patch("app.agent.steps.build_contract.paper_repo"), \
         patch("app.agent.steps.build_contract.task_repo") as mock_task_repo:
        mock_task_repo.save_state = MagicMock()
        mock_task_repo.update_normalized_topic = MagicMock()
        contract = await build_research_contract(db, state, llm, "test-task")

    assert contract is not None
    assert contract.topic == "graph neural networks for test oracle generation"
    assert state.normalized_topic == "graph neural networks for test oracle generation"
    assert "GNN" in state.keywords
    assert state.contract_id is not None


@pytest.mark.asyncio
async def test_build_contract_skips_if_exists():
    """build_research_contract should skip if contract already exists with same hash."""
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.state import ResearchState

    state = ResearchState(task_id="test-task", user_input="test input")
    llm = AsyncMock()

    # Simulate real task
    task = MagicMock()
    task.user_input = "test input"
    db = MagicMock()
    db.get.return_value = task

    # Compute the expected hash
    from app.agent.steps.build_contract import compute_input_hash
    expected_hash = compute_input_hash(task, state)

    existing_contract = MagicMock()
    existing_contract.id = "existing-id"
    existing_contract.input_hash = expected_hash  # Match so it reuses
    existing_contract.version = 1
    existing_contract.topic = "test topic"
    existing_contract.key_terms_json = '["term1"]'

    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = existing_contract
    mock_query.filter.return_value = mock_filter
    db.query.return_value = mock_query

    with patch("app.agent.steps.build_contract.task_repo") as mock_task_repo:
        mock_task_repo.save_state = MagicMock()
        mock_task_repo.update_normalized_topic = MagicMock()
        contract = await build_research_contract(db, state, llm, "test-task")

    # Should return existing contract without calling LLM
    assert contract is existing_contract
    llm.chat_json.assert_not_called()


# === decompose_research_space step ===

@pytest.mark.asyncio
async def test_decompose_creates_questions():
    """decompose_research_space should create research questions."""
    from app.agent.steps.decompose_research_space import decompose_research_space
    from app.agent.state import ResearchState
    from app.schemas.schemas import ResearchDecompositionSchema, ResearchQuestionSchema, ResearchAxisSchema
    from app.db.models import ResearchContract

    state = ResearchState(task_id="test-task", user_input="test", contract_id="contract-id")
    llm = AsyncMock()
    llm.chat_json = AsyncMock(return_value=ResearchDecompositionSchema(
        axes=[
            ResearchAxisSchema(axis_name="problem", values=["oracle generation"]),
            ResearchAxisSchema(axis_name="method", values=["GNN"]),
            ResearchAxisSchema(axis_name="evaluation", values=["accuracy"]),
        ],
        questions=[
            ResearchQuestionSchema(question="现有方法是否在固定token budget下比较？", question_type="evaluation", importance=0.8, searchability=0.7, axis_name="evaluation"),
            ResearchQuestionSchema(question="哪些benchmark覆盖了状态变化过程问题？", question_type="dataset", importance=0.6, searchability=0.9, axis_name="dataset"),
            ResearchQuestionSchema(question="现有方法如何处理冲突记忆？", question_type="method", importance=0.7, searchability=0.8, axis_name="method"),
            ResearchQuestionSchema(question="哪些方法处理false-premise rejection？", question_type="failure", importance=0.5, searchability=0.6, axis_name="failure"),
            ResearchQuestionSchema(question="现有系统在什么条件下会失败？", question_type="problem", importance=0.7, searchability=0.5, axis_name="problem"),
        ],
    ))

    db = MagicMock()
    # Simulate contract exists
    contract = MagicMock()
    contract.id = "contract-id"
    contract.version = 1
    contract.topic = "test topic"
    contract.target_problem = "test"
    contract.target_setting = "test"
    contract.desired_output = "method"
    contract.preferred_directions_json = "[]"
    contract.excluded_directions_json = "[]"
    contract.key_terms_json = "[]"

    db.get.return_value = contract

    # Simulate no existing questions
    mock_query_q = MagicMock()
    mock_filter_q = MagicMock()
    mock_filter_q.all.return_value = []
    mock_query_q.filter.return_value = mock_filter_q

    db.query.return_value = mock_query_q

    with patch("app.agent.steps.decompose_research_space.paper_repo"), \
         patch("app.agent.steps.decompose_research_space.task_repo") as mock_task_repo:
        mock_task_repo.save_state = MagicMock()
        questions = await decompose_research_space(db, state, llm, "test-task")

    assert len(questions) == 5
    assert questions[0].question == "现有方法是否在固定token budget下比较？"

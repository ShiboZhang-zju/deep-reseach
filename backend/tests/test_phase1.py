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

    data = {
        "axes": [
            {"axis_name": "problem", "values": ["oracle generation", "bug detection"]},
            {"axis_name": "method", "values": ["GNN", "Transformer"]},
        ],
        "questions": [
            {
                "question": "现有GNN方法是否在固定token budget下比较？",
                "question_type": "evaluation",
                "importance": 0.8,
                "searchability": 0.7,
                "axis_name": "evaluation",
            },
            {
                "question": "哪些benchmark覆盖了状态变化过程问题？",
                "question_type": "dataset",
                "importance": 0.6,
                "searchability": 0.9,
                "axis_name": "dataset",
            },
        ],
    }
    schema = ResearchDecompositionSchema(**data)
    assert len(schema.axes) == 2
    assert len(schema.questions) == 2
    assert schema.questions[0].question_type == "evaluation"


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
        confidence=0.8,
    ))

    db = MagicMock()
    # Simulate no existing contract
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = None
    mock_query.filter.return_value = mock_filter
    db.query.return_value = mock_query

    with patch("app.agent.steps.build_contract.paper_repo"):
        contract = await build_research_contract(db, state, llm, "test-task")

    assert contract is not None
    assert contract.topic == "graph neural networks for test oracle generation"
    assert state.normalized_topic == "graph neural networks for test oracle generation"
    assert "GNN" in state.keywords


@pytest.mark.asyncio
async def test_build_contract_skips_if_exists():
    """build_research_contract should skip if contract already exists."""
    from app.agent.steps.build_contract import build_research_contract
    from app.agent.state import ResearchState

    state = ResearchState(task_id="test-task", user_input="test")
    llm = AsyncMock()

    existing_contract = MagicMock()
    existing_contract.id = "existing-id"

    db = MagicMock()
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.first.return_value = existing_contract
    mock_query.filter.return_value = mock_filter
    db.query.return_value = mock_query

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

    state = ResearchState(task_id="test-task", user_input="test")
    llm = AsyncMock()
    llm.chat_json = AsyncMock(return_value=ResearchDecompositionSchema(
        axes=[ResearchAxisSchema(axis_name="problem", values=["oracle generation"])],
        questions=[
            ResearchQuestionSchema(
                question="现有方法是否在固定token budget下比较？",
                question_type="evaluation",
                importance=0.8,
                searchability=0.7,
                axis_name="evaluation",
            ),
        ],
    ))

    db = MagicMock()
    # Simulate contract exists
    contract = MagicMock()
    contract.id = "contract-id"
    contract.topic = "test topic"
    contract.target_problem = "test"
    contract.target_setting = "test"
    contract.desired_output = "method"
    contract.preferred_directions_json = "[]"
    contract.excluded_directions_json = "[]"
    contract.key_terms_json = "[]"

    mock_query_contract = MagicMock()
    mock_filter_contract = MagicMock()
    mock_filter_contract.first.return_value = contract
    mock_query_contract.filter.return_value = mock_filter_contract

    # Simulate no existing questions
    mock_query_q = MagicMock()
    mock_count = MagicMock()
    mock_count.count.return_value = 0
    mock_query_q.filter.return_value = mock_count

    db.query.side_effect = [mock_query_contract, mock_query_q, mock_query_q]

    with patch("app.agent.steps.decompose_research_space.paper_repo"):
        questions = await decompose_research_space(db, state, llm, "test-task")

    assert len(questions) == 1
    assert questions[0].question == "现有方法是否在固定token budget下比较？"

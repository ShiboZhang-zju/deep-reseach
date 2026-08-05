"""Tests for the per-task LLM budget enforcement (high-priority #3)."""

import pytest

from app.llm.base import LLMProvider, LLMBudgetExceeded


class _DummyProvider(LLMProvider):
    async def chat(self, messages, temperature=0.7):
        return ""

    async def chat_json(self, messages, schema, temperature=0.3):
        return None


def test_no_budget_by_default():
    p = _DummyProvider()
    for _ in range(100):
        p._track_call()  # unlimited by default -> never raises
    assert p.call_count == 100


def test_call_budget_enforced():
    p = _DummyProvider()
    p.set_budget(max_calls=3)
    p._track_call()
    p._track_call()
    p._track_call()
    with pytest.raises(LLMBudgetExceeded):
        p._track_call()


def test_set_budget_resets_counters():
    p = _DummyProvider()
    p.set_budget(max_calls=5)
    p._track_call()
    p._track_call()
    assert p.call_count == 2
    p.set_budget(max_calls=5)
    assert p.call_count == 0
    assert p.total_tokens_used == 0


def test_token_budget_enforced():
    p = _DummyProvider()
    p.set_budget(max_total_tokens=100)
    # First call fine; simulate usage accumulation over budget.
    p._track_call()
    p.last_usage = {"total_tokens": 150}
    p._record_usage()
    assert p.total_tokens_used == 150
    with pytest.raises(LLMBudgetExceeded):
        p._track_call()

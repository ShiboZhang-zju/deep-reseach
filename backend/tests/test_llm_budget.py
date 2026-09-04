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


def test_budget_isolated_per_task_context():
    """Two tasks sharing the singleton provider must not reset each other's
    spent counters or trip each other's budget (N-way concurrency: task B's
    set_budget used to zero task A's counters, and an exhausted task A used
    to raise LLMBudgetExceeded inside task B)."""
    from app.llm.base import set_task_context

    p = _DummyProvider()

    set_task_context("task-a")
    p.set_budget(max_calls=3)
    p._track_call()
    p._track_call()

    set_task_context("task-b")
    p.set_budget(max_calls=3)
    p._track_call()

    # task-a's counters survive task-b's set_budget: one more call reaches
    # the cap (must not raise), the next one exceeds it.
    set_task_context("task-a")
    p._track_call()
    with pytest.raises(LLMBudgetExceeded):
        p._track_call()

    # task-b is unaffected by task-a exhausting its own budget.
    set_task_context("task-b")
    p._track_call()
    p._track_call()
    with pytest.raises(LLMBudgetExceeded):
        p._track_call()


def test_token_budget_isolated_per_task_context():
    from app.llm.base import set_task_context

    p = _DummyProvider()

    set_task_context("task-a")
    p.set_budget(max_total_tokens=200)
    set_task_context("task-b")
    p.set_budget(max_total_tokens=200)

    p.last_usage = {"total_tokens": 150}
    p._record_usage()  # task-b spends 150

    set_task_context("task-a")
    p.last_usage = {"total_tokens": 150}
    p._record_usage()  # task-a spends its own 150
    assert p.total_tokens_used == 150

    set_task_context("task-b")
    p.last_usage = {"total_tokens": 60}
    p._record_usage()  # task-b accumulates independently: 150 + 60
    assert p.total_tokens_used == 210
    with pytest.raises(LLMBudgetExceeded):
        p._track_call()


@pytest.mark.asyncio
async def test_fallback_wrapper_enforces_token_budget_per_task():
    """Regression (91a1f2e): FallbackLLMProvider._run never called
    _record_usage, so the wrapper's per-task token bucket stayed at zero and
    max_total_tokens silently never fired."""
    from app.llm.base import LLMProvider, set_task_context
    from app.llm.fallback_provider import FallbackLLMProvider

    class _UsageProvider(LLMProvider):
        async def chat(self, messages, temperature=0.7):
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 10,
                               "total_tokens": 80}
            self._track_call()
            self._record_usage()
            return "ok"

        async def chat_json(self, messages, schema, temperature=0.3):
            return None

    wrapper = FallbackLLMProvider([_UsageProvider()])
    set_task_context("task-tok")
    wrapper.set_budget(max_calls=10, max_total_tokens=200)

    await wrapper.chat([], 0.1)
    await wrapper.chat([], 0.1)
    await wrapper.chat([], 0.1)
    assert wrapper._budget_for().total_tokens_used == 240
    with pytest.raises(LLMBudgetExceeded):
        await wrapper.chat([], 0.1)

    # Another task sharing the singleton is unaffected.
    set_task_context("other-task")
    wrapper.set_budget(max_calls=10, max_total_tokens=200)
    await wrapper.chat([], 0.1)


def test_forget_budget_drops_task_bucket():
    from app.llm.base import set_task_context

    p = _DummyProvider()
    set_task_context("task-gone")
    p.set_budget(max_calls=5)
    p._track_call()
    p.forget_budget("task-gone")
    # The bucket is gone: an access recreates a fresh unlimited one.
    assert p._budget_for().max_calls == 0
    assert p._budget_for().call_count == 0

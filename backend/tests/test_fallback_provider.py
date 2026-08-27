"""Circuit-breaker, cooldown and observability tests for LLM failover."""

import asyncio

import pytest

from app.llm.base import LLMBudgetExceeded, LLMProvider
from app.llm.fallback_provider import FallbackLLMProvider, set_observation_context


class _FakeProvider(LLMProvider):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.calls = 0

    async def chat(self, messages, temperature=0.7):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def chat_json(self, messages, schema, temperature=0.3):
        return await self.chat(messages, temperature)


@pytest.mark.asyncio
async def test_transient_failures_open_circuit_and_skip_primary():
    set_observation_context("fallback-test")
    primary = _FakeProvider([
        RuntimeError("LLM call failed: 502 - gateway"),
        RuntimeError("LLM call failed: 502 - gateway"),
    ])
    fallback = _FakeProvider(["fallback-1", "fallback-2", "fallback-3"])
    provider = FallbackLLMProvider(
        [primary, fallback], ["primary", "fallback"],
        failure_threshold=2, cooldown_seconds=30, max_cooldown_seconds=30,
    )

    assert await provider.chat([]) == "fallback-1"
    assert await provider.chat([]) == "fallback-2"
    assert await provider.chat([]) == "fallback-3"
    assert primary.calls == 2
    assert fallback.calls == 3

    metrics = provider.get_observability("fallback-test")
    assert metrics["logical_calls"] == 3
    assert metrics["fallback_calls"] == 3
    assert metrics["providers"]["primary"]["circuit_state"] == "open"
    assert metrics["providers"]["primary"]["circuit_skips"] == 1
    assert metrics["providers"]["primary"]["open_count"] == 1


@pytest.mark.asyncio
async def test_cooldown_allows_one_half_open_probe():
    set_observation_context("cooldown-test")
    primary = _FakeProvider([
        RuntimeError("LLM call failed: 503 - unavailable"),
        "primary-recovered",
    ])
    fallback = _FakeProvider(["fallback-1"])
    provider = FallbackLLMProvider(
        [primary, fallback], ["primary", "fallback"],
        failure_threshold=1, cooldown_seconds=0.01, max_cooldown_seconds=0.01,
    )

    assert await provider.chat([]) == "fallback-1"
    await asyncio.sleep(0.02)
    assert await provider.chat([]) == "primary-recovered"
    assert provider.active is primary
    assert provider.get_observability("cooldown-test")["providers"]["primary"]["circuit_state"] == "closed"


@pytest.mark.asyncio
async def test_connection_refused_is_transient():
    assert FallbackLLMProvider._is_transient(RuntimeError("connection refused"))
    assert FallbackLLMProvider._is_transient(RuntimeError("LLM call failed: 502"))
    assert not FallbackLLMProvider._is_transient(ValueError("invalid JSON"))


@pytest.mark.asyncio
async def test_context_and_budget_errors_are_not_failed_over():
    set_observation_context("deterministic-test")
    primary = _FakeProvider([LLMBudgetExceeded("budget")])
    fallback = _FakeProvider(["must-not-run"])
    provider = FallbackLLMProvider([primary, fallback], ["primary", "fallback"])
    with pytest.raises(LLMBudgetExceeded):
        await provider.chat([])
    assert fallback.calls == 0
    assert provider.get_observability("deterministic-test")["logical_calls"] == 1


@pytest.mark.asyncio
async def test_wrapper_budget_counts_logical_calls_not_retries():
    set_observation_context("budget-test")
    primary = _FakeProvider([RuntimeError("LLM call failed: 502")])
    fallback = _FakeProvider(["ok"])
    provider = FallbackLLMProvider([primary, fallback], ["primary", "fallback"])
    provider.set_budget(max_calls=1)
    assert await provider.chat([]) == "ok"
    with pytest.raises(LLMBudgetExceeded):
        await provider.chat([])
    assert provider.get_observability("budget-test")["logical_calls"] == 1

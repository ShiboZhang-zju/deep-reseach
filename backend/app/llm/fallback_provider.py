"""A provider that fails over to alternates when the primary backend is down."""

from __future__ import annotations

import copy
import logging
import re
import time
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

from app.config import settings
from app.llm.base import (
    LLMBudgetExceeded,
    LLMContextOverflow,
    LLMProvider,
    get_task_context,
    set_task_context,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_TRANSIENT_ERROR_RE = re.compile(
    r"(?:failed|error)[^\d]{0,20}(408|425|429|500|502|503|504)\b",
    re.IGNORECASE,
)


def set_observation_context(task_id: str) -> None:
    """Attribute subsequent calls in this coroutine to one research task."""
    set_task_context(task_id)


class _Circuit:
    def __init__(self) -> None:
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.half_open_probe = False
        self.open_count = 0


class FallbackLLMProvider(LLMProvider):
    """Fail over across providers with per-provider circuit breakers.

    A provider is opened after a configurable number of transient service or
    transport failures. Open providers are skipped until their cooldown expires;
    the first request afterwards is a single half-open probe. Deterministic
    context and budget errors are never retried or counted against a circuit.
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        provider_names: list[str] | None = None,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        max_cooldown_seconds: float | None = None,
    ):
        super().__init__()
        if not providers:
            raise ValueError("FallbackLLMProvider needs at least one provider")
        if provider_names and len(provider_names) != len(providers):
            raise ValueError("provider_names must match providers")
        self.providers = providers
        self.provider_names = provider_names or [
            f"{type(provider).__name__}:{index}"
            for index, provider in enumerate(providers)
        ]
        self._active_index = 0
        self.failure_threshold = max(
            int(failure_threshold if failure_threshold is not None else settings.llm_circuit_failure_threshold), 1)
        self.cooldown_seconds = max(
            float(cooldown_seconds if cooldown_seconds is not None else settings.llm_circuit_cooldown_seconds), 0.0)
        self.max_cooldown_seconds = max(
            float(max_cooldown_seconds if max_cooldown_seconds is not None else settings.llm_circuit_max_cooldown_seconds),
            self.cooldown_seconds,
        )
        self._circuits = [_Circuit() for _ in providers]
        self._stats_by_task: dict[str, dict] = {}

    @property
    def active(self) -> LLMProvider:
        """The provider that last succeeded (or the primary, before any call)."""
        return self.providers[self._active_index]

    def set_budget(self, max_calls: int = 0, max_total_tokens: int = 0) -> None:
        """Budget logical calls at the wrapper, not once per provider retry."""
        super().set_budget(max_calls=max_calls, max_total_tokens=max_total_tokens)
        for provider in self.providers:
            # A failover attempt must not get a second independent task budget.
            provider.set_budget(max_calls=0, max_total_tokens=0)

    def forget_budget(self, context: str | None = None) -> None:
        super().forget_budget(context)
        for provider in self.providers:
            provider.forget_budget(context)

    def _stats(self, task_id: str) -> dict:
        stats = self._stats_by_task.get(task_id)
        if stats is None:
            stats = {
                "task_id": task_id,
                "logical_calls": 0,
                "provider_attempts": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "fallback_calls": 0,
                "circuit_skips": 0,
                "last_call_provider": None,
                "last_call_used_fallback": False,
                "providers": {
                    name: {
                        "attempts": 0,
                        "successes": 0,
                        "failures": 0,
                        "transient_failures": 0,
                        "circuit_skips": 0,
                        "latency_ms_total": 0,
                        "last_latency_ms": None,
                        "last_error": None,
                        "open_count": 0,
                    }
                    for name in self.provider_names
                },
            }
            self._stats_by_task[task_id] = stats
        return stats

    def get_observability(self, task_id: str | None = None) -> dict:
        """Return JSON-safe per-task provider and circuit metrics."""
        key = task_id or get_task_context()
        stats = copy.deepcopy(self._stats(key))
        now = time.monotonic()
        for index, name in enumerate(self.provider_names):
            circuit = self._circuits[index]
            provider_stats = stats["providers"][name]
            provider_stats["circuit_state"] = (
                "open" if circuit.open_until > now else
                "half_open" if circuit.half_open_probe else "closed"
            )
            provider_stats["cooldown_remaining_ms"] = max(
                int((circuit.open_until - now) * 1000), 0
            )
            provider_stats["open_count"] = circuit.open_count
        return stats

    def reset_observability(self, task_id: str | None = None) -> None:
        self._stats_by_task.pop(task_id or get_task_context(), None)

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, (LLMBudgetExceeded, LLMContextOverflow)):
            return False
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
                           ConnectionError, TimeoutError, OSError)):
            return True
        text = str(exc).lower()
        return bool(_TRANSIENT_ERROR_RE.search(text) or any(marker in text for marker in (
            "connection refused", "connection reset", "temporarily unavailable",
            "service unavailable", "bad gateway", "gateway timeout",
        )))

    def _reserve(self, index: int, stats: dict) -> bool:
        circuit = self._circuits[index]
        provider_stats = stats["providers"][self.provider_names[index]]
        now = time.monotonic()
        if circuit.open_until > now:
            circuit_state = "open"
        elif circuit.open_until and circuit.half_open_probe:
            circuit_state = "half_open_busy"
        elif circuit.open_until:
            circuit.half_open_probe = True
            circuit_state = "half_open"
        else:
            circuit_state = "closed"
        if circuit_state in {"open", "half_open_busy"}:
            stats["circuit_skips"] += 1
            provider_stats["circuit_skips"] += 1
            return False
        return True

    def _record_success(self, index: int) -> None:
        circuit = self._circuits[index]
        circuit.consecutive_failures = 0
        circuit.open_until = 0.0
        circuit.half_open_probe = False

    def _record_failure(self, index: int, exc: Exception) -> None:
        if not self._is_transient(exc):
            return
        circuit = self._circuits[index]
        circuit.consecutive_failures += 1
        if circuit.consecutive_failures < self.failure_threshold:
            return
        exponent = max(circuit.consecutive_failures - self.failure_threshold, 0)
        cooldown = min(self.cooldown_seconds * (2 ** exponent), self.max_cooldown_seconds)
        circuit.open_until = time.monotonic() + cooldown
        circuit.half_open_probe = False
        circuit.open_count += 1
        logger.warning(
            "LLM provider %s circuit opened for %.1fs after %d transient failures",
            self.provider_names[index], cooldown, circuit.consecutive_failures,
        )

    async def _run(self, method_name: str, *args, **kwargs):
        self._track_call()
        task_id = get_task_context()
        stats = self._stats(task_id)
        stats["logical_calls"] += 1
        last_error: Exception | None = None
        attempted = 0
        for i, provider in enumerate(self.providers):
            if not self._reserve(i, stats):
                continue
            attempted += 1
            name = self.provider_names[i]
            provider_stats = stats["providers"][name]
            provider_stats["attempts"] += 1
            stats["provider_attempts"] += 1
            started = time.perf_counter()
            try:
                result = await getattr(provider, method_name)(*args, **kwargs)
                elapsed = int((time.perf_counter() - started) * 1000)
                provider_stats["successes"] += 1
                provider_stats["last_latency_ms"] = elapsed
                provider_stats["latency_ms_total"] += elapsed
                self._record_success(i)
                self._active_index = i
                self._sync_state(provider)
                # Attribute the call's token usage to the current task's
                # budget bucket. Regression guard (91a1f2e): _sync_state only
                # mirrors instance fields, so without this the wrapper's
                # per-task total_tokens_used stayed at zero and
                # max_total_tokens never fired.
                self._record_usage()
                stats["successful_calls"] += 1
                stats["fallback_calls"] += int(i > 0)
                stats["last_call_provider"] = name
                stats["last_call_used_fallback"] = i > 0
                return result
            except (LLMBudgetExceeded, LLMContextOverflow):
                raise
            except Exception as exc:
                elapsed = int((time.perf_counter() - started) * 1000)
                provider_stats["failures"] += 1
                provider_stats["last_latency_ms"] = elapsed
                provider_stats["latency_ms_total"] += elapsed
                provider_stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                if self._is_transient(exc):
                    provider_stats["transient_failures"] += 1
                self._record_failure(i, exc)
                last_error = exc
                logger.warning(
                    "LLM provider %s failed (%s); trying next available provider",
                    name, exc,
                )
        stats["failed_calls"] += 1
        if not attempted:
            raise RuntimeError("all LLM providers are circuit-open; retry after cooldown")
        assert last_error is not None
        raise last_error

    def _sync_state(self, provider: LLMProvider) -> None:
        """Mirror usage, while retaining wrapper-level logical counters."""
        self.last_usage = provider.last_usage
        self.total_tokens_used = sum(item.total_tokens_used for item in self.providers)

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return await self._run("chat", messages, temperature)

    async def chat_json(
        self, messages: list[dict], schema: Type[T], temperature: float = 0.3
    ) -> T:
        return await self._run("chat_json", messages, schema, temperature)

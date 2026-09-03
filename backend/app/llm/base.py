"""LLM Provider abstract base class."""

import math
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


_CURRENT_TASK_ID: ContextVar[str] = ContextVar("llm_task_context", default="global")


def set_task_context(task_id: str) -> None:
    """Attribute subsequent calls in this coroutine/thread to one research task.

    The agent runner calls this at every task entry. Budget counters are keyed
    by this context: without per-task keying, concurrent tasks sharing the
    singleton provider would reset each other's spent budget and trip each
    other's LLMBudgetExceeded.
    """
    _CURRENT_TASK_ID.set(task_id or "global")


def get_task_context() -> str:
    """The task id calls are currently attributed to (default: "global")."""
    return _CURRENT_TASK_ID.get()


class LLMBudgetExceeded(RuntimeError):
    """Raised when an LLM call would exceed the per-task call/token budget.

    The agent runner catches this and degrades gracefully to whatever output
    already exists (e.g. emits a landscape brief) rather than hard-failing or
    running unbounded cost.
    """


class LLMContextOverflow(RuntimeError):
    """The prompt alone does not fit the model's context window.

    Distinct from a transport error: the backend rejects such a request with a
    plain HTTP 400, which used to surface as an opaque "LLM call failed: 400"
    and abort the whole task. Callers that build prompts from an unbounded
    collection (evidence units, papers) need to tell this apart so they can
    shrink the input instead of retrying it verbatim.
    """


def estimate_tokens(text: str) -> int:
    """Rough, deliberately conservative token estimate for mixed CJK/Latin text.

    Used only to keep a request inside the model's context window before it is
    sent, so over-estimating is safe and under-estimating is not: CJK
    characters are counted as one token each (they usually are), everything
    else at ~3.2 characters per token, which is below the ~4 that Latin text
    typically achieves.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef")
    return cjk + math.ceil((len(text) - cjk) / 3.2)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate the prompt size of a chat message list (with per-message overhead)."""
    return sum(estimate_tokens(str(message.get("content") or "")) + 4
               for message in messages)


# How fast the measured-vs-estimated ratio is allowed to fall back towards 1.
# Keeping the worst recent observation is deliberate: the cost of forgetting it
# is a rejected request, the cost of remembering it is a slightly stricter guard.
_TOKEN_RATIO_DECAY = 0.9
_MAX_TOKEN_RATIO = 4.0


class _TaskBudget:
    """Per-task budget counters, keyed by the current task context."""

    __slots__ = ("max_calls", "max_total_tokens", "call_count", "total_tokens_used")

    def __init__(self, max_calls: int = 0, max_total_tokens: int = 0):
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.call_count = 0
        self.total_tokens_used = 0


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Subclasses should update self.last_usage after each API call
    so callers can record token usage in agent traces, and call
    self._track_call() at the start of each chat/chat_json.
    """

    def __init__(self):
        self.last_usage: dict | None = None  # {"prompt_tokens": N, "completion_tokens": M, "total_tokens": K}
        # Per-task budgets, keyed by the task context (see set_task_context).
        # The instance fields below mirror the budget of the most recent call
        # context, for introspection and for legacy context-less callers.
        self._task_budgets: dict[str, _TaskBudget] = {}
        self.max_calls: int = 0
        self.max_total_tokens: int = 0
        self.call_count: int = 0
        self.total_tokens_used: int = 0
        # Correction factor between measured prompt tokens and the
        # character-based estimate. No single divisor fits both shapes of prompt
        # this pipeline sends: a prompt dense with UUIDs tokenizes at roughly
        # 1.8 characters per token, while prose reaches about 4, so a fixed
        # divisor either under-protects the first or needlessly rejects the
        # second. Measured usage from the backend calibrates it instead.
        self.token_estimate_ratio: float = 1.0

    def calibrate_token_estimate(self, estimated: int, measured: int) -> None:
        """Update the estimate correction factor from one measured call."""
        if estimated <= 0 or measured <= 0:
            return
        observed = measured / estimated
        self.token_estimate_ratio = min(
            max(observed, self.token_estimate_ratio * _TOKEN_RATIO_DECAY, 1.0),
            _MAX_TOKEN_RATIO,
        )

    def set_budget(self, max_calls: int = 0, max_total_tokens: int = 0) -> None:
        """Configure and reset the per-task budget.

        Counters are stored per task context, so concurrent tasks on the
        shared singleton provider each get an independent budget instead of
        overwriting one another (a per-task budget on shared instance fields
        is a no-op at N-way concurrency, and an exhausted task trips
        LLMBudgetExceeded in unrelated tasks).
        """
        self._task_budgets[get_task_context()] = _TaskBudget(
            max_calls=max_calls, max_total_tokens=max_total_tokens)
        # Mirror for introspection and legacy context-less reads.
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.call_count = 0
        self.total_tokens_used = 0

    def _budget_for(self) -> _TaskBudget:
        """The budget counters for the current task context."""
        budget = self._task_budgets.get(get_task_context())
        if budget is None:
            # No set_budget in this context (e.g. direct provider use in eval
            # scripts): unlimited, matching the legacy zero-field default.
            budget = _TaskBudget()
            self._task_budgets[get_task_context()] = budget
        return budget

    def _track_call(self) -> None:
        """Increment the call counter and enforce the budget. Call at chat entry."""
        budget = self._budget_for()
        budget.call_count += 1
        self.call_count = budget.call_count
        if budget.max_calls and budget.call_count > budget.max_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exceeded: {budget.call_count} > {budget.max_calls}"
            )
        if budget.max_total_tokens and budget.total_tokens_used > budget.max_total_tokens:
            raise LLMBudgetExceeded(
                f"LLM token budget exceeded: {budget.total_tokens_used} > {budget.max_total_tokens}"
            )

    def _record_usage(self) -> None:
        """Accumulate token usage from last_usage. Call after each API call."""
        if self.last_usage:
            used = int(self.last_usage.get("total_tokens", 0) or 0)
            budget = self._budget_for()
            budget.total_tokens_used += used
            self.total_tokens_used = budget.total_tokens_used

    @abstractmethod
    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Plain-text chat completion."""
        ...

    @abstractmethod
    async def chat_json(
        self,
        messages: list[dict],
        schema: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """Structured JSON chat completion validated against a Pydantic schema."""
        ...

    def get_last_usage(self) -> dict | None:
        """Return token usage from the most recent API call."""
        return self.last_usage

"""LLM Provider abstract base class."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMBudgetExceeded(RuntimeError):
    """Raised when an LLM call would exceed the per-task call/token budget.

    The agent runner catches this and degrades gracefully to whatever output
    already exists (e.g. emits a landscape brief) rather than hard-failing or
    running unbounded cost.
    """


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Subclasses should update self.last_usage after each API call
    so callers can record token usage in agent traces, and call
    self._track_call() at the start of each chat/chat_json.
    """

    def __init__(self):
        self.last_usage: dict | None = None  # {"prompt_tokens": N, "completion_tokens": M, "total_tokens": K}
        # Per-task budget (0 = unlimited). Set by the runner before a task.
        self.max_calls: int = 0
        self.max_total_tokens: int = 0
        self.call_count: int = 0
        self.total_tokens_used: int = 0

    def set_budget(self, max_calls: int = 0, max_total_tokens: int = 0) -> None:
        """Configure and reset the per-task budget."""
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.call_count = 0
        self.total_tokens_used = 0

    def _track_call(self) -> None:
        """Increment the call counter and enforce the budget. Call at chat entry."""
        self.call_count += 1
        if self.max_calls and self.call_count > self.max_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exceeded: {self.call_count} > {self.max_calls}"
            )
        if self.max_total_tokens and self.total_tokens_used > self.max_total_tokens:
            raise LLMBudgetExceeded(
                f"LLM token budget exceeded: {self.total_tokens_used} > {self.max_total_tokens}"
            )

    def _record_usage(self) -> None:
        """Accumulate token usage from last_usage. Call after each API call."""
        if self.last_usage:
            self.total_tokens_used += int(self.last_usage.get("total_tokens", 0) or 0)

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

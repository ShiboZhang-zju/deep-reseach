"""LLM Provider abstract base class."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Subclasses should update self.last_usage after each API call
    so callers can record token usage in agent traces.
    """

    def __init__(self):
        self.last_usage: dict | None = None  # {"prompt_tokens": N, "completion_tokens": M, "total_tokens": K}

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

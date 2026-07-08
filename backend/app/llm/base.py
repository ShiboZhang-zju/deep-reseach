"""LLM Provider abstract base class."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

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

"""OpenAI direct provider (fallback)."""

import json
import logging

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Direct OpenAI API provider."""

    def __init__(self):
        super().__init__()
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = await self._post(payload)
        return resp["choices"][0]["message"]["content"]

    async def chat_json(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        temperature: float = 0.3,
    ) -> BaseModel:
        schema_json = schema.model_json_schema()
        system_msg = {
            "role": "system",
            "content": (
                "You must respond with valid JSON matching this schema. "
                f"Do not include markdown code fences.\n\n{json.dumps(schema_json, ensure_ascii=False)}"
            ),
        }
        payload = {
            "model": self.model,
            "messages": [system_msg] + messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        resp = await self._post(payload)
        content = resp["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("OpenAI JSON parse error: %s", e)
            raise

    async def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI call failed: {resp.status_code} - {resp.text[:200]}")
            data = resp.json()
            usage = data.get("usage")
            if usage:
                self.last_usage = usage
            return data

"""Venus LLM Proxy provider (OpenAI-compatible API)."""

import json
import logging

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class VenusProvider(LLMProvider):
    """Calls the Venus LLM Proxy which is OpenAI-API compatible."""

    def __init__(self):
        super().__init__()
        self.base_url = settings.venus_llm_proxy_url
        token = settings.env_venus_openapi_secret_id
        self.token = f"{token}@4083" if token else ""
        self.model = settings.venus_llm_model

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
            logger.error("LLM JSON parse error: %s\nContent: %s", e, content[:500])
            # Retry once with explicit error feedback
            retry_messages = [system_msg] + messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Your previous response was invalid: {e}. Please respond again with valid JSON."},
            ]
            payload2 = {
                "model": self.model,
                "messages": retry_messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            resp2 = await self._post(payload2)
            content2 = resp2["choices"][0]["message"]["content"]
            data2 = json.loads(content2)
            return schema.model_validate(data2)

    async def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                logger.error("Venus LLM error %d: %s", resp.status_code, resp.text[:500])
                raise RuntimeError(f"LLM call failed: {resp.status_code} - {resp.text[:200]}")
            data = resp.json()
            # Record token usage for cost tracking
            usage = data.get("usage")
            if usage:
                self.last_usage = usage
            return data

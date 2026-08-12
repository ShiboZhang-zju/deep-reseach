"""Venus LLM Proxy provider (OpenAI-compatible API)."""

import json
import logging
import re

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.base import (
    LLMContextOverflow,
    LLMProvider,
    estimate_messages_tokens,
)

logger = logging.getLogger(__name__)

# How much of an unparseable response is echoed back when asking for a retry.
# The full response used to be appended verbatim, which pushed the retry over
# the context limit precisely when the first attempt had been truncated by it.
_RETRY_EXCERPT_CHARS = 400

# The backend rejects a request when input + max_tokens exceeds the window, so
# the output allowance has to be derived from an *upper bound* on the prompt
# rather than from the estimate itself. Observed: a 513-token under-estimate of
# a 5005-token prompt produced "passed 5005 input tokens and requested 35956
# output tokens" — one token over the 40960 window — and failed the phase.
_ESTIMATE_UPPER_FACTOR = 1.5
_OUTPUT_ALLOWANCE_MARGIN = 2048

# Kept between the admitted prompt size and the reserved output, so that the
# reserved floor still fits when the estimate was slightly optimistic.
_GUARD_SAFETY_MARGIN = 512

# The backend reports the measured prompt size when it rejects an oversize
# request; that number is the ground truth our estimate should be calibrated
# against.
_PASSED_TOKENS_RE = re.compile(r"passed (\d+) input tokens")


class VenusProvider(LLMProvider):
    """Calls the Venus LLM Proxy which is OpenAI-API compatible."""

    def __init__(self):
        super().__init__()
        self.base_url = settings.venus_llm_proxy_url
        token = settings.env_venus_openapi_secret_id
        self.token = f"{token}@4083" if token else ""
        self.model = settings.venus_llm_model
        # Extra sampling / template params forwarded to OpenAI-compatible
        # backends that accept them (e.g. local Qwen: top_k, repetition_penalty,
        # chat_template_kwargs.enable_thinking). Configurable so switching model
        # backends does not require code changes.
        self.extra_body = settings.llm_extra_body or {}
        self.context_tokens = settings.llm_context_tokens
        self.max_output_tokens = settings.llm_max_output_tokens

    @property
    def input_token_budget(self) -> int:
        """Prompt tokens that still leave room for the reserved output budget."""
        if not self.context_tokens:
            return 0
        return max(self.context_tokens - self.max_output_tokens - _GUARD_SAFETY_MARGIN, 0)

    def _with_extra(self, payload: dict) -> dict:
        """Merge configured extra_body params into a request payload."""
        if self.max_output_tokens:
            payload.setdefault("max_tokens", self._output_allowance(payload.get("messages") or []))
        if self.extra_body:
            for key, value in self.extra_body.items():
                payload.setdefault(key, value)
        return payload

    def _output_allowance(self, messages: list[dict]) -> int:
        """How many output tokens to request for this specific prompt.

        `llm_max_output_tokens` is what admitting a prompt reserved, i.e. a
        floor, not a ceiling: pinning every request to it would truncate the long
        outputs this pipeline needs (a research report runs well past 4k tokens).
        A short prompt leaves most of the window free, so ask for what is left —
        but compute that from an upper bound on the prompt, because the backend
        rejects `input + max_tokens > context` outright and the prompt estimate
        is only approximate. When the prompt is large the floor applies, and
        admission has already guaranteed it fits.
        """
        if not self.context_tokens:
            return self.max_output_tokens
        estimated = estimate_messages_tokens(messages) * self.token_estimate_ratio
        upper_bound = int(estimated * _ESTIMATE_UPPER_FACTOR) + _OUTPUT_ALLOWANCE_MARGIN
        return max(self.max_output_tokens, self.context_tokens - upper_bound)

    def _guard_input_size(self, messages: list[dict], purpose: str) -> None:
        """Reject a prompt that cannot fit the context window before sending it.

        The backend answers such a request with an opaque HTTP 400, so callers
        could not tell "the prompt is too large" from "the service is broken"
        and a whole task was failed instead of the prompt being shrunk. The
        character estimate is scaled by the correction factor measured from
        previous calls, so this neither under-protects ID-dense prompts nor
        rejects prose that would have fit.
        """
        budget = self.input_token_budget
        if not budget:
            return
        estimated = int(estimate_messages_tokens(messages) * self.token_estimate_ratio)
        if estimated > budget:
            raise LLMContextOverflow(
                f"{purpose}: prompt is ~{estimated} tokens but only {budget} are "
                f"available (context {self.context_tokens} minus {self.max_output_tokens} "
                "reserved for the response)"
            )

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        self._guard_input_size(messages, "chat")
        payload = self._with_extra({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        })
        resp = await self._post(payload)
        return self._content_of(resp)

    @staticmethod
    def _content_of(resp: dict) -> str:
        """Extract the message text, tolerating a null content field.

        This backend returns `content: null` when the answer went into the
        `reasoning` channel or when generation stopped at the token limit before
        any content was emitted. Passing None on to json.loads raised a TypeError
        that no caller was catching, turning an empty answer into a crash.
        """
        choice = (resp.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if not content:
            logger.warning("LLM returned no content (finish_reason=%s); treating as empty",
                           choice.get("finish_reason"))
        return content or ""

    @staticmethod
    def _inject_schema_instruction(messages: list[dict], schema_json: dict) -> list[dict]:
        """Return messages with the JSON-schema instruction merged into the
        leading system message.

        Some OpenAI-compatible backends (e.g. local Qwen) require that a system
        message, if present, appears exactly once at the very beginning. So we
        merge the schema directive into an existing leading system message
        instead of prepending a second one.
        """
        directive = (
            "You must respond with valid JSON matching this schema. "
            f"Do not include markdown code fences.\n\n{json.dumps(schema_json, ensure_ascii=False)}"
        )
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0]['content']}\n\n{directive}"
            return msgs
        return [{"role": "system", "content": directive}] + msgs

    def _build_retry_messages(self, prepared: list[dict], content: str,
                              error: Exception) -> list[dict]:
        """Build a correction turn that is never larger than the first attempt.

        A response can be invalid *because* it was cut off when the prompt left
        too little room for it. Echoing that response back grew the prompt
        further, so the retry was rejected outright (observed: a 400 on 40961
        input tokens against a 40960 window, which failed the whole task).
        Only a short excerpt is echoed, and it is dropped entirely when even
        that would not fit.
        """
        excerpt = (content or "")[:_RETRY_EXCERPT_CHARS]
        instruction = (
            f"Your previous response was not valid JSON ({error}). "
            "Respond again with a single valid JSON object and nothing else. "
            "Keep every string short so the response is complete and not cut off."
        )
        if excerpt:
            with_excerpt = prepared + [{
                "role": "user",
                "content": f"{instruction}\n\nIt started with:\n{excerpt}",
            }]
            budget = self.input_token_budget
            estimated = estimate_messages_tokens(with_excerpt) * self.token_estimate_ratio
            if not budget or estimated <= budget:
                return with_excerpt
        return prepared + [{"role": "user", "content": instruction}]

    async def chat_json(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        temperature: float = 0.3,
    ) -> BaseModel:
        schema_json = schema.model_json_schema()
        prepared = self._inject_schema_instruction(messages, schema_json)
        self._guard_input_size(prepared, f"chat_json({schema.__name__})")
        payload = self._with_extra({
            "model": self.model,
            "messages": prepared,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        })
        resp = await self._post(payload)
        content = self._content_of(resp)

        try:
            data = json.loads(content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("LLM JSON parse error: %s\nContent: %s", e, content[:500])
            # Retry once with explicit error feedback. Keep the system message
            # at the beginning; append the correction as a user turn.
            retry_messages = self._build_retry_messages(prepared, content, e)
            # The correction turn itself costs tokens, so a prompt that only just
            # fit can no longer be repaired in place. Say so instead of sending a
            # request the backend will reject: the caller has to shrink its input.
            self._guard_input_size(retry_messages, f"chat_json({schema.__name__}) repair retry")
            payload2 = self._with_extra({
                "model": self.model,
                "messages": retry_messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            })
            resp2 = await self._post(payload2)
            data2 = json.loads(self._content_of(resp2))
            return schema.model_validate(data2)

    async def _post(self, payload: dict, _output_floor_retry: bool = False) -> dict:
        # Budget enforcement (raises LLMBudgetExceeded if over budget).
        self._track_call()
        headers = {"Content-Type": "application/json"}
        # Only send Authorization when a token is configured. Local/OpenAI-
        # compatible backends without auth reject/ignore a bogus bearer.
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                body = resp.text or ""
                # Log enough of the body to be diagnosable: this backend wraps
                # the real reason inside a `forward bad request` envelope, and a
                # 500-char cut left only `{"error":{"message":` visible, which
                # made an oversize prompt indistinguishable from a broken
                # service in the logs.
                logger.error("Venus LLM error %d: %s", resp.status_code, body[:1500])
                if resp.status_code == 400 and (
                    "context length" in body or "input_tokens" in body
                ):
                    # The rejection carries the measured prompt size. Feeding it
                    # back is the only ground truth available for the estimate,
                    # and it makes the guard stricter for the next call of the
                    # same shape instead of failing the same way again.
                    measured = _PASSED_TOKENS_RE.search(body)
                    if measured:
                        self.calibrate_token_estimate(
                            estimate_messages_tokens(payload.get("messages") or []),
                            int(measured.group(1)))
                    # "input + max_tokens > context" is not the same failure as
                    # "the prompt does not fit". If we asked for more output than
                    # the window can spare, shrink the request to the reserved
                    # floor — which admission already guaranteed fits — instead
                    # of failing a phase over an output allowance we chose.
                    if not _output_floor_retry and (
                            payload.get("max_tokens") or 0) > self.max_output_tokens:
                        logger.warning("Retrying with the reserved output floor (%d tokens)",
                                       self.max_output_tokens)
                        retry_payload = dict(payload)
                        retry_payload["max_tokens"] = self.max_output_tokens
                        return await self._post(retry_payload, _output_floor_retry=True)
                    raise LLMContextOverflow(
                        f"backend rejected the prompt as too long: {body[:400]}")
                raise RuntimeError(f"LLM call failed: {resp.status_code} - {body[:300]}")
            data = resp.json()
            # Record token usage for cost tracking
            usage = data.get("usage")
            if usage:
                self.last_usage = usage
                self._record_usage()
                # Teach the size guard what this backend's tokenizer actually
                # charges for the prompt we just sent.
                self.calibrate_token_estimate(
                    estimate_messages_tokens(payload.get("messages") or []),
                    int(usage.get("prompt_tokens") or 0),
                )
            return data

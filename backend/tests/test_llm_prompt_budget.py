"""Prompt-size budgeting in the LLM provider.

A real run failed with `LLM call failed: 400 - ... You passed 40961 input tokens
... context length is only 40960`. The prompt filled the window, so the response
was truncated mid-string, and the JSON-repair retry then appended that truncated
response verbatim — making the retry larger than the attempt that had already run
out of room.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_estimate_tokens_counts_cjk_per_character():
    from app.llm.base import estimate_tokens

    assert estimate_tokens("错误累积") == 4
    # Latin text is cheaper per character but never free.
    assert 0 < estimate_tokens("error accumulation") < len("error accumulation")


def test_guard_rejects_a_prompt_that_cannot_fit_the_context():
    from app.llm.base import LLMContextOverflow
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 1000
    provider.max_output_tokens = 200
    assert provider.input_token_budget == 800

    with pytest.raises(LLMContextOverflow):
        provider._guard_input_size([{"role": "user", "content": "证" * 900}], "test")

    provider._guard_input_size([{"role": "user", "content": "证" * 100}], "test")


def test_repair_retry_never_grows_past_the_first_attempt():
    from app.llm.base import estimate_messages_tokens
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 1000
    provider.max_output_tokens = 200
    prepared = [{"role": "system", "content": "系统提示" * 20},
                {"role": "user", "content": "证据" * 100}]
    truncated = "{\n  \"gaps\": [" + "证据描述" * 400

    retry = provider._build_retry_messages(prepared, truncated, ValueError("unterminated string"))

    # The whole truncated response used to be echoed back, which pushed the
    # retry past the window that had just truncated it.
    assert estimate_messages_tokens(retry) < estimate_messages_tokens(
        prepared + [{"role": "assistant", "content": truncated}])
    assert estimate_messages_tokens(retry) <= provider.input_token_budget
    assert "valid JSON" in retry[-1]["content"]


def test_repair_retry_keeps_a_short_excerpt_when_there_is_room():
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 40960
    provider.max_output_tokens = 4096
    prepared = [{"role": "system", "content": "schema"}, {"role": "user", "content": "prompt"}]

    retry = provider._build_retry_messages(prepared, '{"gaps": [{"gap_type"', ValueError("boom"))

    assert "It started with:" in retry[-1]["content"]
    assert '{"gaps"' in retry[-1]["content"]


@pytest.mark.asyncio
async def test_repair_retry_that_cannot_fit_is_reported_not_sent():
    from pydantic import BaseModel

    from app.llm.base import LLMContextOverflow
    from app.llm.venus_provider import VenusProvider

    class Answer(BaseModel):
        value: str = ""

    calls = []

    provider = VenusProvider()
    provider.context_tokens = 1000
    provider.max_output_tokens = 200

    async def fake_post(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "{\"value\": "}}]}

    provider._post = fake_post
    # A prompt that fits, but leaves no room for a correction turn on top of it.
    with pytest.raises(LLMContextOverflow):
        await provider.chat_json([{"role": "user", "content": "证" * 700}], Answer)

    assert len(calls) == 1


def test_requests_reserve_an_output_budget():
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.max_output_tokens = 4096
    payload = provider._with_extra({"model": provider.model, "messages": []})
    # Without this the backend derives the output budget from what is left of the
    # context window and answers "requested 0 output tokens".
    assert payload["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_oversize_rejection_from_backend_is_classified():
    import httpx

    from app.llm.base import LLMContextOverflow
    from app.llm.venus_provider import VenusProvider

    class FakeResponse:
        status_code = 400
        text = ('{"error":{"message":"You passed 40961 input tokens ... the model\'s '
                'context length is only 40960 tokens","code":400}}')

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    original = httpx.AsyncClient
    httpx.AsyncClient = FakeClient
    try:
        provider = VenusProvider()
        with pytest.raises(LLMContextOverflow):
            await provider._post({"model": provider.model, "messages": []})
    finally:
        httpx.AsyncClient = original


def test_measured_usage_calibrates_the_size_guard():
    from app.llm.base import LLMContextOverflow
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 40960
    provider.max_output_tokens = 4096
    prompt = [{"role": "user", "content": "x" * 100_000}]

    # A character estimate alone says this fits.
    provider._guard_input_size(prompt, "test")

    # The real mining prompt measured ~1.5x its estimate (UUID-dense text
    # tokenizes far worse than prose), which is the gap that let a doomed
    # request through.
    provider.calibrate_token_estimate(31_250, 47_000)
    assert provider.token_estimate_ratio == pytest.approx(1.504, abs=0.01)
    with pytest.raises(LLMContextOverflow):
        provider._guard_input_size(prompt, "test")


def test_calibration_decays_but_never_below_one():
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.calibrate_token_estimate(1000, 1800)
    assert provider.token_estimate_ratio == pytest.approx(1.8)

    # A friendlier prompt relaxes the guard gradually instead of forgetting the
    # worst case immediately.
    provider.calibrate_token_estimate(1000, 900)
    assert provider.token_estimate_ratio == pytest.approx(1.62)
    for _ in range(50):
        provider.calibrate_token_estimate(1000, 900)
    assert provider.token_estimate_ratio == 1.0

    # Nonsense measurements are ignored.
    provider.calibrate_token_estimate(0, 5000)
    assert provider.token_estimate_ratio == 1.0

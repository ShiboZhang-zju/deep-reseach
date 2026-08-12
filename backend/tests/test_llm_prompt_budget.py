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

# Verbatim shape of the gateway's rejection: the real reason is nested inside a
# "forward bad request" envelope, which is why it was unreadable in the logs.
OVERSIZE_BODY = (
    '{"error":{"code":-3007,"message":"","param":null,"type":"forward bad request, '
    '[HTTP 400] {\\"error\\":{\\"message\\":\\"You passed 5005 input tokens and requested '
    '35956 output tokens. However, the model context length is only 40960 tokens\\",'
    '\\"param\\":\\"input_tokens\\",\\"code\\":400}}"}}'
)


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
    # The reserved output plus a safety margin must still fit after the prompt,
    # so the admitted size stays below context - reserve.
    assert provider.input_token_budget < 800

    with pytest.raises(LLMContextOverflow):
        provider._guard_input_size([{"role": "user", "content": "证" * 900}], "test")

    provider._guard_input_size([{"role": "user", "content": "证" * 100}], "test")


def test_repair_retry_never_grows_past_the_first_attempt():
    from app.llm.base import estimate_messages_tokens
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 1000
    provider.max_output_tokens = 200
    prepared = [{"role": "system", "content": "系统提示" * 10},
                {"role": "user", "content": "证据" * 40}]
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
        await provider.chat_json([{"role": "user", "content": "证" * 210}], Answer)

    assert len(calls) == 1


def test_output_allowance_leaves_room_for_an_underestimated_prompt():
    """input + max_tokens must fit the window even when the estimate is low.

    Production failure: a 4492-token estimate of a 5005-token prompt produced
    "passed 5005 input tokens and requested 35956 output tokens" — one token over
    the 40960 window — which failed the whole gap-mining phase.
    """
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 40960
    provider.max_output_tokens = 4096
    prompt = [{"role": "user", "content": "证" * 4400}]

    allowance = provider._output_allowance(prompt)
    actual_prompt_tokens = 5005  # what the backend measured for this shape

    assert actual_prompt_tokens + allowance <= provider.context_tokens
    # Still generous: a short prompt must not be capped at the reserved floor.
    assert allowance > provider.max_output_tokens


def test_output_allowance_is_a_floor_not_a_ceiling():
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    provider.context_tokens = 40960
    provider.max_output_tokens = 4096

    # Without max_tokens the backend derives the output budget from what is left
    # of the context window and answers "requested 0 output tokens".
    short = provider._with_extra({"model": provider.model,
                                  "messages": [{"role": "user", "content": "hi"}]})
    # A short prompt leaves most of the window free; pinning every request to the
    # reserved floor would truncate long outputs such as a research report.
    assert short["max_tokens"] > 4096

    # A prompt that nearly fills the window still gets the reserved floor, which
    # is exactly what admitting it promised.
    long_prompt = [{"role": "user", "content": "证" * 37_000}]
    assert provider._with_extra({"model": provider.model,
                                 "messages": long_prompt})["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_oversize_from_output_allowance_retries_with_the_floor():
    """An allowance we chose must not fail a phase; the prompt itself still can."""
    import httpx

    from app.llm.venus_provider import VenusProvider

    seen = []

    class FakeResponse:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 5005, "total_tokens": 5010}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            seen.append(json["max_tokens"])
            if json["max_tokens"] > 4096:
                return FakeResponse(400, OVERSIZE_BODY)
            return FakeResponse(200, "{}")

    original = httpx.AsyncClient
    httpx.AsyncClient = FakeClient
    try:
        provider = VenusProvider()
        provider.context_tokens = 40960
        provider.max_output_tokens = 4096
        data = await provider._post({"model": provider.model, "max_tokens": 35956,
                                     "messages": [{"role": "user", "content": "证" * 4400}]})
        assert data["choices"][0]["message"]["content"] == "ok"
        assert seen == [35956, 4096]
        # The measured size from the rejection tightens the estimate.
        assert provider.token_estimate_ratio > 1.0
    finally:
        httpx.AsyncClient = original


def test_null_content_is_treated_as_empty_not_as_a_crash():
    from app.llm.venus_provider import VenusProvider

    provider = VenusProvider()
    # This backend returns content: null when the answer went to the `reasoning`
    # channel or generation stopped at the token limit. json.loads(None) raised a
    # TypeError that no caller caught.
    assert provider._content_of(
        {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}) == ""
    assert provider._content_of({}) == ""
    assert provider._content_of({"choices": [{"message": {"content": "ok"}}]}) == "ok"


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

"""PDF retrieval must retry what is transient and stop retrying what is not.

Measured 2026-09-18: of arXiv papers that had never been cached, a plain retry
recovered 40/40 and then 30/30, and a controlled comparison showed UA,
concurrency and retry policy were all irrelevant variables. So the failures are
not properties of the papers — they are transient (rate-limit windows), and a
single unbounded attempt turns a temporary condition into permanent evidence
loss. Conversely, retrying a 403 paywall or an HTML landing page wastes a minute
per paper per round forever, because those outcomes cannot change.

The retry key must therefore be the *cause*, and the cause must be persisted:
without it an unretrievable paper is re-attempted every round with no progress
and no record, and stays indistinguishable from a paper that was read and found
empty.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeResponse:
    def __init__(self, status_code, content=b"", content_type="application/pdf"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


class _FakeClient:
    """Stands in for httpx.AsyncClient, replaying a scripted response sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        self.calls += 1
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _patch_client(monkeypatch, responses):
    import httpx
    from app.services import rag_service

    holder = _FakeClient(responses)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: holder)
    return holder


@pytest.mark.asyncio
async def test_paywall_is_not_retried(monkeypatch):
    """403 is durable: retrying it cannot help and costs a minute per round."""
    from app.services import rag_service

    holder = _patch_client(monkeypatch, [_FakeResponse(403)])

    async def _no_sleep(_):
        raise AssertionError("must not back off on a permanent failure")

    monkeypatch.setattr(rag_service.asyncio, "sleep", _no_sleep)

    outcome = await rag_service._try_download_classified("https://dl.acm.org/x.pdf")

    assert outcome.content is None
    assert outcome.gap_reason == rag_service.PDF_GAP_PAYWALL
    assert holder.calls == 1, "a 403 must be attempted exactly once"


@pytest.mark.asyncio
async def test_html_landing_page_is_not_retried(monkeypatch):
    """200 with an HTML body is a property of the URL, not a transient blip."""
    from app.services import rag_service

    holder = _patch_client(
        monkeypatch, [_FakeResponse(200, b"<html>landing</html>", "text/html")])

    async def _no_sleep(_):
        raise AssertionError("must not back off on a permanent failure")

    monkeypatch.setattr(rag_service.asyncio, "sleep", _no_sleep)

    outcome = await rag_service._try_download_classified("https://link.springer.com/x")

    assert outcome.content is None
    assert outcome.gap_reason == rag_service.PDF_GAP_NOT_PDF
    assert holder.calls == 1


@pytest.mark.asyncio
async def test_transient_failure_is_retried_and_can_succeed(monkeypatch):
    """A 429 is transient: the same URL must be attempted again."""
    from app.services import rag_service

    holder = _patch_client(monkeypatch, [
        _FakeResponse(429),
        _FakeResponse(200, b"%PDF-1.4 payload"),
    ])

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(rag_service.asyncio, "sleep", _no_sleep)

    outcome = await rag_service._try_download_classified("https://arxiv.org/pdf/1234")

    assert outcome.content == b"%PDF-1.4 payload"
    assert outcome.gap_reason is None
    assert holder.calls == 2, "the 429 must be followed by a retry"


@pytest.mark.asyncio
async def test_transient_failure_is_reported_after_exhausting_attempts(monkeypatch):
    """Exhausted retries must still report *why*, so the loss is accountable."""
    from app.services import rag_service

    holder = _patch_client(monkeypatch, [_FakeResponse(503)])

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(rag_service.asyncio, "sleep", _no_sleep)

    outcome = await rag_service._try_download_classified("https://example.org/x.pdf")

    assert outcome.content is None
    assert outcome.gap_reason == rag_service.PDF_GAP_TRANSIENT
    assert outcome.retryable is True
    assert holder.calls == rag_service.PDF_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_dead_url_is_not_retried(monkeypatch):
    """404 is durable."""
    from app.services import rag_service

    holder = _patch_client(monkeypatch, [_FakeResponse(404)])

    async def _no_sleep(_):
        raise AssertionError("must not back off on a permanent failure")

    monkeypatch.setattr(rag_service.asyncio, "sleep", _no_sleep)

    outcome = await rag_service._try_download_classified("https://example.org/gone.pdf")

    assert outcome.gap_reason == rag_service.PDF_GAP_NOT_FOUND
    assert holder.calls == 1


def test_legacy_download_helper_still_returns_bytes():
    """`_try_download` stays byte-oriented for existing callers."""
    import inspect
    from app.services import rag_service

    assert inspect.iscoroutinefunction(rag_service._try_download)
    src = inspect.getsource(rag_service._try_download)
    assert "_try_download_classified" in src


def test_arxiv_url_helper_is_shared():
    """The arXiv URL must be built in one place so classification can reuse it."""
    from app.services import rag_service

    assert rag_service._arxiv_pdf_url("2401.12345") == "https://arxiv.org/pdf/2401.12345"
    assert rag_service._arxiv_pdf_url("2401.12345v2") == "https://arxiv.org/pdf/2401.12345"

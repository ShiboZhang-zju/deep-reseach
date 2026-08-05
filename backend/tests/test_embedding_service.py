"""Tests for O5a embedding service (cosine + backend selection, no network)."""

import pytest

from app.services import embedding_service as es
from app.config import settings


def test_cosine_similarity_basic():
    assert es.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert es.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert es.cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)


def test_cosine_similarity_degenerate():
    assert es.cosine_similarity([], [1, 2]) == 0.0
    assert es.cosine_similarity([0, 0], [1, 2]) == 0.0
    assert es.cosine_similarity([1, 2, 3], [1, 2]) == 0.0  # dim mismatch


def test_embed_texts_empty_returns_empty():
    assert es.embed_texts([]) == []


def test_embedding_enabled_api_backend(monkeypatch):
    monkeypatch.setattr(settings, "embedding_backend", "api")
    assert es.embedding_enabled() is True
    monkeypatch.undo()


def test_embed_texts_api_batches(monkeypatch):
    """embed_texts (api) should call the endpoint and preserve input order."""
    monkeypatch.setattr(settings, "embedding_backend", "api")
    monkeypatch.setattr(settings, "embedding_batch_size", 2)

    calls = {"n": 0}

    class FakeResp:
        status_code = 200
        def json(self):
            # Return embeddings tagged by index so we can assert ordering
            return {"data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k):
            calls["n"] += 1
            return FakeResp()

    monkeypatch.setattr(es.httpx, "Client", FakeClient)
    out = es.embed_texts(["a", "b"])
    # Sorted by index -> [1.0], [2.0]
    assert out == [[1.0], [2.0]]
    assert calls["n"] == 1
    monkeypatch.undo()


def test_embed_texts_api_raises_on_error(monkeypatch):
    monkeypatch.setattr(settings, "embedding_backend", "api")

    class FakeResp:
        status_code = 500
        text = "boom"

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k): return FakeResp()

    monkeypatch.setattr(es.httpx, "Client", FakeClient)
    with pytest.raises(RuntimeError):
        es.embed_texts(["x"])
    monkeypatch.undo()

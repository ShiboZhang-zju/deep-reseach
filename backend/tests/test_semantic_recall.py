"""Semantic recall: the filter must be honest about what it removed.

Regression target (2026-11-02): the pilot adjudication failed because keyword
search returned pools where ZERO papers were on-topic (fbp-089: 12 statistical-
mediation / medical-GNN papers for an MoE-routing claim), and the reviewer had
no way to tell "no prior art" from "retrieval missed". The fix filters by
embedding similarity, so the tests that matter are about DISCLOSURE: a silent
filter would recreate the same bug in a new form.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _vec(first: float, dim: int = 4) -> list[float]:
    """A unit-ish vector whose cosine with _vec(1.0) equals `first` (approx)."""
    import math
    rest = math.sqrt(max(1.0 - first * first, 0.0)) / math.sqrt(dim - 1)
    return [first] + [rest] * (dim - 1)


class _FakeEmbedding:
    """Maps text -> a preset vector via a lookup table."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            key = (t or "").strip()[:60]
            out.append(self.table.get(key, [0.0, 1.0, 0.0, 0.0]))
        return out

    def cosine_similarity(self, a, b):
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class _ExplodingEmbedding:
    def embed_texts(self, texts):
        raise RuntimeError("embedding API down")

    def cosine_similarity(self, a, b):  # pragma: no cover
        raise AssertionError("should not be reached")


@pytest.fixture
def fake_emb(monkeypatch):
    def _install(table):
        from app.services import semantic_recall as mod
        fake = _FakeEmbedding(table)
        monkeypatch.setattr(mod, "embedding_service", fake)
        return fake
    return _install


def test_filter_keeps_relevant_and_reports_the_dropped_count(fake_emb):
    """The kept/filtered counts are the contract, not a nicety.

    A consumer that only sees `kept` cannot distinguish "nothing relevant
    exists" from "the filter removed everything" — the exact ambiguity that
    made pass 1 unusable.
    """
    from app.services.semantic_recall import filter_by_similarity

    fake_emb({
        "claim": _vec(1.0),
        "on topic": _vec(0.90),
        "also on topic": _vec(0.60),
        "borderline": _vec(0.42),
        "off topic": _vec(0.30),
        "way off": _vec(0.05),
    })
    items = [{"text": t} for t in
             ("on topic", "also on topic", "borderline", "off topic", "way off")]

    kept, diag = filter_by_similarity("claim", items, threshold=0.5)

    titles = [k["text"] for k in kept]
    assert titles == ["on topic", "also on topic"]  # sorted desc, filtered
    assert diag["raw_count"] == 5
    assert diag["kept_count"] == 2
    assert diag["filtered_count"] == 3
    assert diag["semantic_filtered"] is True
    assert diag["threshold"] == 0.5
    # The best rejected score tells a reader whether the cut was clean
    # (far below threshold) or merely trimmed a continuous tail.
    assert diag["best_rejected_similarity"] == pytest.approx(0.42, abs=0.01)


def test_threshold_zero_disables_filtering_but_keeps_rerank(fake_emb):
    from app.services.semantic_recall import filter_by_similarity

    fake_emb({"claim": _vec(1.0), "a": _vec(0.2), "b": _vec(0.8)})
    items = [{"text": "a"}, {"text": "b"}]

    kept, diag = filter_by_similarity("claim", items, threshold=0.0)

    assert len(kept) == 2
    assert diag["filtered_count"] == 0
    # Still reranked by relevance.
    assert [k["text"] for k in kept] == ["b", "a"]


def test_empty_claim_does_not_silently_filter_everything(fake_emb):
    """An empty claim must NOT look like a successful semantic pass."""
    from app.services.semantic_recall import filter_by_similarity

    fake_emb({})
    items = [{"text": "a"}, {"text": "b"}]
    kept, diag = filter_by_similarity("   ", items, threshold=0.9)

    assert len(kept) == 2, "must keep all rather than filter on a blank claim"
    assert diag["semantic_filtered"] is False
    assert diag["degraded"] == "empty_claim"


def test_embedding_failure_degrades_to_unfiltered_and_says_so(monkeypatch):
    """Infrastructure failure must be disclosed, never passed off as success."""
    from app.services import semantic_recall as mod

    monkeypatch.setattr(mod, "embedding_service", _ExplodingEmbedding())
    items = [{"text": "a"}, {"text": "b"}]
    kept, diag = mod.filter_by_similarity("claim", items, threshold=0.9)

    assert len(kept) == 2, "an outage is not evidence that everything is off-topic"
    assert diag["semantic_filtered"] is False
    assert "embedding API down" in diag["degraded"]


@pytest.mark.asyncio
async def test_semantic_recall_returns_diagnostics_and_respects_limit(
        fake_emb, monkeypatch):
    from app.services import semantic_recall as mod

    fake_emb({
        "claim": _vec(1.0),
        "p1": _vec(0.9), "p2": _vec(0.8), "p3": _vec(0.7),
        "noise": _vec(0.1), "junk": _vec(0.05),
    })

    class _P:
        def __init__(self, title):
            self.title = title
            self.abstract = ""
            self.year = 2026
            self.venue = "X"

    class _FakeService:
        async def search_multiple_queries(self, queries, limit=15,
                                          allowed_sources=None):
            return [_P(t) for t in ("p1", "p2", "p3", "noise", "junk")]

    import app.services.search_service as ss
    monkeypatch.setattr(ss, "SearchService", _FakeService)

    kept, diag = await mod.semantic_recall("claim", ["q"], limit=2, threshold=0.5)

    assert len(kept) == 2, "limit applies after filtering"
    assert [k["title"] for k in kept] == ["p1", "p2"]
    assert diag["raw_count"] == 5
    assert diag["kept_count"] == 3
    assert diag["filtered_count"] == 2
    assert diag["returned_count"] == 2
    assert diag["queries"] == ["q"]
    assert all(k["similarity"] >= 0.5 for k in kept)


@pytest.mark.asyncio
async def test_semantic_recall_without_queries_is_a_noop():
    from app.services.semantic_recall import semantic_recall

    kept, diag = await semantic_recall("claim", [], limit=5)
    assert kept == []
    assert diag["raw_count"] == 0


@pytest.mark.asyncio
async def test_search_failure_is_reported_not_raised(monkeypatch, fake_emb):
    from app.services import semantic_recall as mod

    fake_emb({})

    class _BoomService:
        async def search_multiple_queries(self, **kwargs):
            raise RuntimeError("S2 down")

    import app.services.search_service as ss
    monkeypatch.setattr(ss, "SearchService", _BoomService)

    kept, diag = await mod.semantic_recall("claim", ["q"], limit=5)
    assert kept == []
    assert "search_failed" in diag["degraded"]

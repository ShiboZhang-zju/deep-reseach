"""Prewarming must make the corpus a settled input before extraction spends LLM budget.

The distinction matters because the two stages fail differently. Downloading is
network-bound and *recoverable* — measured 2026-09-18, of arXiv papers that had
never been cached a plain retry recovered 40/40 and then 30/30, and a controlled
comparison showed UA/concurrency/retry policy were all irrelevant, i.e. failures
track rate-limit windows rather than papers. Extraction is LLM-bound and
expensive. Interleaving them turns a transient network condition into permanent
evidence loss and makes the round's duration unpredictable.

So prewarm has to do two things: fetch first, and *report what it could not get*
so a truncated corpus is visible instead of silently reducing evidence yield.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _Paper:
    def __init__(self, pid):
        self.id = pid


def _patch_downloads(monkeypatch, available: set, reasons: dict | None = None):
    """Make `_download_pdfs` behave as if only `available` ids are retrievable."""
    from app.agent.steps import extract_evidence

    async def fake_download(papers, task_id, concurrency=3):
        paths = {}
        diagnoses = {}
        for paper, _tp in papers:
            if paper.id in available:
                paths[paper.id] = os.path.join("pdf_cache", paper.id + ".pdf")
                diagnoses[paper.id] = None
            else:
                paths[paper.id] = None
                diagnoses[paper.id] = (reasons or {}).get(paper.id, "transient")
        extract_evidence._download_diagnoses[task_id] = diagnoses
        return paths

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fake_download)
    return fake_download


@pytest.mark.asyncio
async def test_prewarm_reports_retrieved_and_unavailable_counts(monkeypatch):
    from app.agent.steps import extract_evidence

    papers = [(_Paper("p1"), None), (_Paper("p2"), None), (_Paper("p3"), None)]
    _patch_downloads(monkeypatch, available={"p1", "p2"})

    summary = await extract_evidence.prewarm_pdfs(papers, "task-1")

    assert summary["total"] == 3
    assert summary["retrieved"] == 2
    assert summary["unavailable"] == 1


@pytest.mark.asyncio
async def test_prewarm_breaks_down_failures_by_reason(monkeypatch):
    """The reason distribution is what makes a truncated corpus actionable.

    "paywall" says stop retrying and disclose it; "transient" says try again next
    round. Collapsing both into one counter loses that distinction.
    """
    from app.agent.steps import extract_evidence

    papers = [(_Paper("p1"), None), (_Paper("p2"), None), (_Paper("p3"), None)]
    _patch_downloads(
        monkeypatch, available={"p1"},
        reasons={"p2": "paywall", "p3": "transient"},
    )

    summary = await extract_evidence.prewarm_pdfs(papers, "task-2")

    assert summary["reasons"] == {"paywall": 1, "transient": 1}


@pytest.mark.asyncio
async def test_prewarm_labels_missing_diagnosis_as_unknown(monkeypatch):
    """A paper with no recorded verdict must not be silently dropped from the report."""
    from app.agent.steps import extract_evidence

    async def fake_download(papers, task_id, concurrency=3):
        # Records no diagnosis at all — the report must still account for it.
        return {paper.id: None for paper, _tp in papers}

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fake_download)

    papers = [(_Paper("p1"), None)]
    summary = await extract_evidence.prewarm_pdfs(papers, "task-3")

    assert summary["unavailable"] == 1
    assert summary["reasons"] == {"unknown": 1}


@pytest.mark.asyncio
async def test_prewarm_handles_empty_input(monkeypatch):
    from app.agent.steps import extract_evidence

    summary = await extract_evidence.prewarm_pdfs([], "task-4")

    assert summary == {"total": 0, "retrieved": 0, "unavailable": 0, "reasons": {}}


@pytest.mark.asyncio
async def test_prewarm_skips_none_papers(monkeypatch):
    """A TaskPaper whose Paper is missing must not crash the prewarm."""
    from app.agent.steps import extract_evidence

    papers = [(_Paper("p1"), None), (None, None)]
    _patch_downloads(monkeypatch, available={"p1"})

    summary = await extract_evidence.prewarm_pdfs(papers, "task-5")

    assert summary["total"] == 1
    assert summary["retrieved"] == 1


@pytest.mark.asyncio
async def test_prewarm_uses_the_configured_concurrency(monkeypatch):
    """Concurrency must be honoured: raising it triggers the 429s it cannot recover."""
    from app.agent.steps import extract_evidence
    from app.config import settings

    seen = {}

    async def fake_download(papers, task_id, concurrency=3):
        seen["concurrency"] = concurrency
        return {paper.id: None for paper, _tp in papers}

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fake_download)
    monkeypatch.setattr(settings, "pdf_prewarm_concurrency", 2, raising=False)

    await extract_evidence.prewarm_pdfs([(_Paper("p1"), None)], "task-6")

    assert seen["concurrency"] == 2


def test_prewarm_is_wired_into_the_extraction_entry_point():
    """Guard: prewarm must actually run before extraction spends anything."""
    import inspect
    from app.agent.steps import extract_evidence

    src = inspect.getsource(extract_evidence.extract_evidence_units)
    assert "prewarm_pdfs" in src
    assert "enable_pdf_prewarm" in src
    # The prewarm call must precede the extraction batch loop.
    assert src.index("prewarm_pdfs") < src.index("_extract_from_paper_safe(")


def test_prewarm_can_be_disabled():
    """The switch must exist so an operator can trade the delay away."""
    from app.config import settings

    assert hasattr(settings, "enable_pdf_prewarm")
    assert hasattr(settings, "pdf_prewarm_concurrency")

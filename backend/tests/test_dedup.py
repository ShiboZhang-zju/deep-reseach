"""Tests for paper deduplication logic (scoring_service.py).

Covers: DOI dedup, arXiv ID dedup, title hash dedup, source merging, field filling.
"""

import pytest

from app.paper_sources.base import RawPaper
from app.services.scoring_service import (
    normalize_title,
    title_hash,
    normalize_paper,
    deduplicate_papers,
)


class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("Attention Is All You Need") == "attention is all you need"

    def test_strip_punctuation(self):
        assert normalize_title("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_title("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalize_title("") == ""


class TestTitleHash:
    def test_consistent_hash(self):
        h1 = title_hash("Attention Is All You Need")
        h2 = title_hash("attention is all you need")
        assert h1 == h2

    def test_different_titles_different_hash(self):
        assert title_hash("Paper A") != title_hash("Paper B")

    def test_returns_hex_string(self):
        h = title_hash("test")
        assert len(h) == 64  # SHA-256 hex
        int(h, 16)  # valid hex


class TestNormalizePaper:
    def test_basic_conversion(self):
        raw = RawPaper(
            title="Test Paper",
            abstract="An abstract",
            authors=["Author A"],
            year=2024,
            doi="10.1234/test",
            source="semantic_scholar",
        )
        result = normalize_paper(raw)
        assert result["title"] == "Test Paper"
        assert result["doi"] == "10.1234/test"
        assert result["year"] == 2024
        assert "semantic_scholar" in result["sources_json"]

    def test_none_ids_become_none(self):
        raw = RawPaper(title="Test", source="arxiv")
        result = normalize_paper(raw)
        assert result["doi"] is None
        assert result["arxiv_id"] is None
        assert result["semantic_scholar_id"] is None

    def test_title_hash_computed(self):
        raw = RawPaper(title="Test Paper", source="arxiv")
        result = normalize_paper(raw)
        assert result["title_hash"] == title_hash("Test Paper")
        assert result["normalized_title"] == "test paper"


class TestDeduplicatePapers:
    def test_dedup_by_doi(self):
        p1 = RawPaper(title="Paper One", doi="10.1234/abc", source="crossref", citation_count=10)
        p2 = RawPaper(title="Paper One (Duplicate)", doi="10.1234/abc", source="openalex", citation_count=20)
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        # Should keep the higher citation count
        assert result[0].citation_count == 20

    def test_dedup_by_arxiv_id(self):
        p1 = RawPaper(title="arXiv Paper", arxiv_id="2401.12345", source="arxiv")
        p2 = RawPaper(title="Same arXiv Paper", arxiv_id="2401.12345", source="semantic_scholar")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_dedup_by_title_hash(self):
        p1 = RawPaper(title="Attention Is All You Need", source="semantic_scholar")
        p2 = RawPaper(title="attention is all you need", source="openalex")
        # No DOI, arxiv_id, s2_id, openalex_id → falls back to title hash
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_no_dedup_different_papers(self):
        p1 = RawPaper(title="Paper A", doi="10.1234/a", source="crossref")
        p2 = RawPaper(title="Paper B", doi="10.1234/b", source="crossref")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2

    def test_source_merging(self):
        p1 = RawPaper(title="Test", doi="10.1234/x", source="crossref", abstract="abs1")
        p2 = RawPaper(title="Test", doi="10.1234/x", source="openalex", abstract="")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        # Should fill in missing fields
        assert result[0].abstract == "abs1"

    def test_empty_list(self):
        assert deduplicate_papers([]) == []

    def test_merges_fields_from_duplicate(self):
        p1 = RawPaper(title="Paper", source="arxiv", arxiv_id="2401.1", abstract="")
        p2 = RawPaper(title="Paper", source="semantic_scholar", arxiv_id="2401.1",
                      abstract="has abstract", semantic_scholar_id="abc123")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        assert result[0].abstract == "has abstract"
        assert result[0].semantic_scholar_id == "abc123"

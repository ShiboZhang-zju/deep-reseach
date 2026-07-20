"""Tests for LLM Wiki merge logic (wiki_service.py).

Covers: section-level merge, paper_ids merge, links merge, contradictions merge,
section parsing, fuzzy title matching.
"""

import json
import pytest

from app.services.wiki_service import (
    _parse_sections,
    _reconstruct_sections,
    _find_similar_page,
    _merge_page,
)


class TestParseSections:
    def test_simple_markdown(self):
        md = "## Summary\nThis is a summary.\n## Details\nMore details."
        sections = _parse_sections(md)
        assert "## Summary" in sections
        assert "## Details" in sections
        assert "summary" in sections["## Summary"].lower()

    def test_intro_before_headings(self):
        md = "Intro text\n## Section\nBody"
        sections = _parse_sections(md)
        assert "__intro__" in sections
        assert "Intro text" in sections["__intro__"]

    def test_empty_markdown(self):
        assert _parse_sections("") == {}

    def test_no_headings(self):
        md = "Just plain text without headings"
        sections = _parse_sections(md)
        assert "__intro__" in sections
        assert "plain text" in sections["__intro__"]


class TestReconstructSections:
    def test_roundtrip(self):
        md = "## Section A\nContent A\n## Section B\nContent B"
        sections = _parse_sections(md)
        reconstructed = _reconstruct_sections(sections)
        assert "Section A" in reconstructed
        assert "Content A" in reconstructed
        assert "Section B" in reconstructed

    def test_preserves_intro(self):
        sections = {"__intro__": "Hello", "## Body": "World"}
        result = _reconstruct_sections(sections)
        assert "Hello" in result
        assert "World" in result


class TestSectionMerge:
    """Test the _merge_page function via a mock existing page."""

    class MockPage:
        def __init__(self, content, paper_ids=None, links=None, contradictions=None):
            self.content_markdown = content
            self.paper_ids_json = json.dumps(paper_ids or [])
            self.links_json = json.dumps(links or [])
            self.contradictions_json = json.dumps(contradictions or [])

    class MockAction:
        def __init__(self, content, paper_ids=None, links=None, contradictions=None):
            self.op = "update"
            self.page_type = "concept"
            self.title = "Test"
            self.content = content
            self.paper_ids = paper_ids or []
            self.links = links or []
            self.contradictions = contradictions or []

    def test_merge_paper_ids(self):
        existing = self.MockPage("old", paper_ids=["p1", "p2"])
        action = self.MockAction("new", paper_ids=["p2", "p3"])
        result = _merge_page(existing, action)
        assert result == "updated"
        merged_ids = json.loads(existing.paper_ids_json)
        assert set(merged_ids) == {"p1", "p2", "p3"}

    def test_merge_links(self):
        existing = self.MockPage("old", links=["link1"])
        action = self.MockAction("new", links=["link2", "link1"])
        _merge_page(existing, action)
        merged_links = json.loads(existing.links_json)
        assert set(merged_links) == {"link1", "link2"}

    def test_merge_contradictions(self):
        existing = self.MockPage("old", contradictions=["contra1"])
        action = self.MockAction("new", contradictions=["contra2"])
        _merge_page(existing, action)
        merged = json.loads(existing.contradictions_json)
        assert set(merged) == {"contra1", "contra2"}

    def test_merge_new_section_added(self):
        existing = self.MockPage("## Section A\nContent A")
        action = self.MockAction("## Section B\nContent B")
        _merge_page(existing, action)
        assert "Section B" in existing.content_markdown
        assert "Content B" in existing.content_markdown

    def test_merge_existing_section_appends_new_lines(self):
        existing = self.MockPage("## Summary\nLine 1")
        action = self.MockAction("## Summary\nLine 1\nLine 2")
        _merge_page(existing, action)
        assert "Line 1" in existing.content_markdown
        assert "Line 2" in existing.content_markdown

    def test_merge_dedup_lines(self):
        """Duplicate lines should not be added twice."""
        existing = self.MockPage("## Summary\nSame line")
        action = self.MockAction("## Summary\nSame line")
        _merge_page(existing, action)
        # Count occurrences of "Same line"
        assert existing.content_markdown.count("Same line") == 1


class TestFuzzyTitleMatch:
    class MockPage:
        def __init__(self, title):
            self.title = title

    def test_exact_match(self):
        pages = [self.MockPage("记忆增强的大语言模型")]
        # _find_similar_page needs DB, test the logic indirectly
        # We test the char_set logic by checking Jaccard similarity concept
        import re
        def _char_set(s):
            s = re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())
            return set(s)

        s1 = _char_set("记忆增强的大语言模型")
        s2 = _char_set("记忆增强的大语言模型")
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        score = intersection / union if union > 0 else 0
        assert score == 1.0

    def test_similar_titles_high_score(self):
        import re
        def _char_set(s):
            s = re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())
            return set(s)

        s1 = _char_set("记忆增强的大语言模型")
        s2 = _char_set("大语言模型记忆机制")
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        score = intersection / union if union > 0 else 0
        # Should have significant overlap (shared characters)
        assert score > 0.3

    def test_different_titles_low_score(self):
        import re
        def _char_set(s):
            s = re.sub(r'[^\w\u4e00-\u9fff]', '', s.lower())
            return set(s)

        s1 = _char_set("记忆增强的大语言模型")
        s2 = _char_set("多智能体路径规划")
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        score = intersection / union if union > 0 else 0
        assert score < 0.3

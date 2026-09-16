"""PDF extraction quality tests.

The first version of the backfill tool fed PyMuPDF's raw visual lines straight
into `_split_into_chunks`, which assumes each line is a semantic unit. Measured
consequences on real papers:

* 69% of chunks collapsed into the default `method` section (a heading glued to
  body text never satisfies the whole-line header rule);
* 53% of chunks were under 50 words and 13k under 15 words (`4.4.1 XNER` and
  wrapped fragments became standalone chunks);
* hyphenated words stayed split ("formances are far below"), corrupting the text
  a reviewer must read;
* 18 of the sampled chunks were bibliography entries.

These tests pin the extraction-side fixes. They run against synthetic layout
strings rather than real PDFs so they are deterministic, plus one optional
live check that is skipped when the corpus is absent.
"""

import pytest

from eval.production_e2e.backfill_pdf_chunks import (
    _drop_reference_chunks,
    _heal_hyphenation,
    _is_boilerplate,
    _split_leading_heading,
)


class TestHyphenationHealing:
    def test_joins_a_split_word(self):
        healed = _heal_hyphenation("world knowl-", "edge inside it")
        assert healed is not None
        assert healed[0] + healed[1] == "world knowledge inside it"

    def test_does_not_join_an_em_dash(self):
        # An em dash at a line end is punctuation, not a split word.
        assert _heal_hyphenation("results —", "and more") is None

    def test_does_not_join_when_next_line_starts_with_punctuation(self):
        assert _heal_hyphenation("see Fig-", ", page 3") is None

    def test_does_not_join_without_a_trailing_hyphen(self):
        assert _heal_hyphenation("a complete sentence.", "Next one") is None

    def test_ignores_a_single_letter_fragment(self):
        # "-a" style artifacts are not real hyphenation.
        assert _heal_hyphenation("x-", "continuation") is None


class TestBoilerplate:
    @pytest.mark.parametrize("line", [
        "arXiv:2401.12345v1 [cs.CL] 12 Jan 2024",
        "42",
        "Downloaded from https://example.org/article",
        "All rights reserved.",
        "This article is protected by copyright.",
    ])
    def test_flags_boilerplate(self, line):
        assert _is_boilerplate(line)

    @pytest.mark.parametrize("line", [
        "3.1 Concept Memory",
        "We introduce a concept memory module.",
        "The model achieves 91.2% accuracy on the benchmark.",
    ])
    def test_keeps_real_content(self, line):
        assert not _is_boilerplate(line)


class TestLeadingHeadingSplit:
    def test_splits_numbered_heading_glued_to_body(self):
        """The specific failure that collapsed sections to `unknown`."""
        parts = _split_leading_heading(
            "2.1 Concept Memory We introduce a concept memory module.")
        assert parts == ["2.1 Concept Memory", "We introduce a concept memory module."]

    def test_keeps_a_bare_heading_intact(self):
        assert _split_leading_heading("3.1 Setup") == ["3.1 Setup"]

    def test_does_not_split_a_plain_sentence(self):
        text = "We evaluate on three benchmarks and report accuracy."
        assert _split_leading_heading(text) == [text]

    def test_does_not_split_when_title_is_too_long(self):
        # No numbered heading present, so nothing to split.
        text = ("1 this is a very long run of lowercase words that should not be "
                "treated as a heading at all because it is prose")
        assert _split_leading_heading(text) == [text]


class TestReferenceChunkDropping:
    class _C:
        def __init__(self, text, section="unknown"):
            self.text = text
            self.section = section

    def test_drops_a_bibliography_chunk(self):
        bib = ("[12] Zhihong Xu, et al. 2019. The effectiveness of intelligent "
               "tutoring systems. arXiv preprint arXiv:2308.08155. pp. 1-14. "
               "doi: 10.1145/1234567.1234568. vol. 3, no. 2.")
        assert _drop_reference_chunks([self._C(bib)]) == []

    def test_drops_chunk_announcing_references(self):
        text = ("References\n" + "A authors and B writers. Some title. 2021. " * 4)
        assert _drop_reference_chunks([self._C(text)]) == []

    def test_drops_a_tiny_fragment(self):
        """`4.4.1 XNER` and wrapped fragments used to become standalone chunks."""
        assert _drop_reference_chunks([self._C("4.4.1 XNER")]) == []

    def test_keeps_real_prose(self):
        prose = (
            "We propose a method that clips activation outliers per layer, "
            "measuring sensitivity with a calibration set. Our experiments on "
            "four benchmarks show consistent gains over uniform clipping, and "
            "ablation confirms the gain is not from extra compute.")
        kept = _drop_reference_chunks([self._C(prose, "method")])
        assert len(kept) == 1

    def test_keeps_prose_with_a_single_citation(self):
        # A methods paragraph may legitimately cite one paper.
        prose = (
            "Following Smith et al., 2020 we adopt the standard protocol and "
            "report the mean over three seeds for every configuration tested, "
            "using the same evaluation harness throughout the study.")
        kept = _drop_reference_chunks([self._C(prose, "experiment")])
        assert len(kept) == 1


class TestSnippetBodyFilter:
    """The snippet picker must not hand a reviewer a bibliography entry.

    Before this filter 11% of snippets were bibliography or table content.
    """

    def test_rejects_citation_dense_text(self):
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        bib = ("arXiv preprint arXiv:2308.08155. 12 Zhihong Xu, Kausalai "
               "Wijekumar. 2019. The effectiveness of intelligent tutoring "
               "systems. doi: 10.1145/1234.5678, pp. 1-14.")
        assert not _looks_like_body(bib)

    def test_rejects_table_rows(self):
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        assert not _looks_like_body("2880 36:16 100% 1512.1 86.5 4096 2:04 99%")

    def test_rejects_short_fragments(self):
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        assert not _looks_like_body("4.4.1 XNER")

    def test_accepts_prose(self):
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        prose = ("We show that per-layer clipping thresholds derived from an "
                 "activation sensitivity measure improve 4-bit quantization "
                 "accuracy across four benchmarks without added inference cost.")
        assert _looks_like_body(prose)

    def test_rejects_proceedings_frontmatter(self):
        """Real leftover: a proceedings header matched a RAG claim by title words.

        It repeated the paper's own title ("Retrieval Augmented Generation or
        Long-Context LLMs?") so it scored highly on claim terms while containing
        no evidence about the claim at all.
        """
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        front = ("Proceedings of the 2024 Conference on Empirical Methods in "
                 "Natural Language Processing: Industry Track, pages 881-893 "
                 "November 12-16, 2024 (c)2024 Association for Computational "
                 "Linguistics Retrieval Augmented Generation or Long-Context LLMs?")
        assert not _looks_like_body(front)

    def test_rejects_copyright_and_page_range_lines(self):
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        assert not _looks_like_body(
            "Published as a conference paper at ICLR 2024. Volume 12, pages "
            "4401-4417. ISBN 978-1-4503-1234-5. All content in this area was "
            "uploaded by the contributing author.")

    def test_rejects_text_without_sentence_punctuation(self):
        """Headers and front-matter rarely contain a real sentence."""
        from eval.production_e2e.enrich_audit_snippets import _looks_like_body
        assert not _looks_like_body(
            "Retrieval Augmented Generation Long Context Language Models "
            "Efficient Inference Attention Dilution Benchmark Evaluation "
            "Empirical Study Results Analysis Discussion Conclusion")

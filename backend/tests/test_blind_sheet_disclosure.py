"""Blind-sheet disclosure tests.

The sheet is the ONLY channel through which a human reviewer learns how
trustworthy the candidate list is. Two defects are guarded here:

1. The header existed as two separate copies (the generator's and the
   `--rebuild-sheet` path's). The disclosure was added to one, so rebuilding the
   sheet silently produced one WITHOUT the limitation notice — the reviewer would
   then import 39%-irrelevant candidate lists into their verdicts with no warning.
2. A submission whose candidates are all unfiltered keyword matches must say so,
   otherwise "false_open: no" is indistinguishable from "we searched badly".
"""

from eval.production_e2e.super_audit import blind_review_section, blind_sheet_header


class TestSheetHeader:
    def test_states_the_candidate_noise_rate(self):
        text = "\n".join(blind_sheet_header())
        assert "39%" in text, "the measured noise rate must be stated"

    def test_warns_against_reading_noise_as_novelty(self):
        text = "\n".join(blind_sheet_header())
        assert "IN FAVOUR of" in text, \
            "must explicitly forbid treating noise as novelty evidence"

    def test_distinguishes_search_failure_from_absence_of_prior_art(self):
        text = "\n".join(blind_sheet_header())
        assert "FAILED TO FIND" in text

    def test_records_that_automatic_filtering_was_rejected(self):
        """A reviewer who wonders why there is no filter deserves the answer."""
        text = "\n".join(blind_sheet_header())
        assert "attempted and rejected" in text
        assert "41-70%" in text

    def test_header_is_stable_across_calls(self):
        assert blind_sheet_header() == blind_sheet_header()


class TestPerSubmissionQualitySignal:
    def test_reports_gated_and_quoted_counts(self):
        cands = [
            {"title": "A", "source": "local_corpus", "snippet": "some prose"},
            {"title": "B", "source": "local_corpus"},
            {"title": "C", "source": "openalex"},
        ]
        text = blind_review_section("sid", "topic", "claim text", cands)
        assert "3 candidates" in text
        assert "2 relevance-gated" in text
        assert "1 with a quoted passage" in text

    def test_flags_a_wholly_unfiltered_list(self):
        """The case where a `no prior art` verdict deserves the least trust."""
        cands = [{"title": "x", "source": "openalex"},
                 {"title": "y", "source": "arxiv"}]
        text = blind_review_section("sid", "topic", "claim", cands)
        assert "0 relevance-gated" in text
        assert "unfiltered keyword matches" in text

    def test_leaks_no_internal_scores_or_identity(self):
        """Blind protocol: no system name, no internal score may appear."""
        cands = [{"title": "T", "source": "openalex",
                  "novelty": 0.91, "final_score": 4.2}]
        text = blind_review_section("sid", "topic", "claim", cands)
        for leak in ("openalex", "local_corpus", "novelty", "final_score",
                     "direct_llm", "retrieval_llm", "full_v2"):
            assert leak not in text, f"sheet leaked {leak!r}"

    def test_retrieval_failure_is_marked(self):
        text = blind_review_section("sid", "topic", "claim", [],
                                    query_failure="502 from gateway")
        assert "RETRIEVAL FAILED" in text
        assert "absence of prior art" in text

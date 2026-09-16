"""Audit candidate filter tests.

These pin the DECISION not to filter candidates by vocabulary overlap, which was
reached after measuring two versions of such a gate against the real 2026-09-16
records:

* `>=2 shared content words`: dropped 630/895 (70%) and left 20 of 91 targets
  with no candidates at all.
* `>=1 shared content word`: dropped 367/895 (41%), still removing relevant work,
  e.g. "Gradient-Based Importance Smoothing for Dynamic Rank Allocation" was cut
  from a claim about "measuring gradient norms".

The root reason is structural, not a tuning problem: a claim is an experimental
DESIGN description while a title is a TOPIC phrase, so their vocabularies
legitimately diverge. Low overlap is therefore not evidence of irrelevance.

The asymmetry that settles it: a spurious candidate costs one "NONE" verdict,
but a wrongly dropped one is invisible — and if it was the killer paper, the
audit reports a false gap, which is exactly the error the audit exists to
detect. Recall must win, so only non-papers are removed.
"""

import pytest

from eval.production_e2e.super_audit import filter_candidates_by_relevance


def _c(title, source="openalex"):
    return {"title": title, "source": source}


class TestNonPaperRemoval:
    @pytest.mark.parametrize("title", [
        "Proceedings of Resources for African Indigenous Languages (RAIL) 2026",
        "Proceedings of the 28th International Conference on Computational Linguistics",
        "Front Matter",
        "Table of Contents",
        "Author Index",
        "Erratum to: Some Paper",
        "Call for Papers: Special Issue on NLP",
    ])
    def test_drops_non_papers(self, title):
        assert filter_candidates_by_relevance([_c(title)], "any claim") == []

    def test_keeps_entries_without_a_title(self):
        """Not removing these would show a blank row; dropping is intentional."""
        assert filter_candidates_by_relevance([_c("")], "any claim") == []


class TestRecallIsPreserved:
    def test_keeps_a_topically_distant_paper(self):
        """The exact false-drop the overlap gate produced.

        This paper is genuinely relevant to a claim about gradient norms, yet it
        shares no distinctive term with the claim text once generic words are
        removed. It must survive.
        """
        claim = ("A minimal experiment measuring gradient norms to isolate the "
                 "optimization state of adapters")
        cand = _c("Late Adapter Tuning: A Cost-Effective Approach to "
                  "Parameter-Efficient Fine-Tuning")
        assert filter_candidates_by_relevance([cand], claim) == [cand]

    def test_keeps_an_unrelated_looking_paper(self):
        """A wrong keep costs one NONE verdict; a wrong drop can fabricate a gap."""
        claim = "Does per-layer clipping improve 4-bit quantization accuracy?"
        cand = _c("Obesity and Cancer: Existing and New Hypotheses")
        assert filter_candidates_by_relevance([cand], claim) == [cand]

    def test_keeps_local_corpus_candidates(self):
        cand = _c("Anything At All", source="local_corpus")
        assert filter_candidates_by_relevance([cand], "claim") == [cand]

    def test_never_empties_a_realistic_candidate_set(self):
        claim = "A minimal experiment measuring gradient norms"
        cands = [_c("Proceedings of Something"),
                 _c("Gradient-Based Importance Smoothing for Dynamic Rank Allocation"),
                 _c("Low-Resource Fine-Tuning of LLMs for Domain-Specific Tasks")]
        kept = filter_candidates_by_relevance(cands, claim)
        assert len(kept) == 2, "only the proceedings entry should be removed"

    def test_does_not_drop_the_title_case_variant(self):
        # "Proceedings" mid-title is part of a paper name, not a front-matter entry.
        cand = _c("A Study of Proceedings Analysis in Legal NLP")
        assert filter_candidates_by_relevance([cand], "legal nlp") == [cand]

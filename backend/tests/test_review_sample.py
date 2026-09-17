"""Blind-review sample builder tests.

The sample is what a human actually reads to decide FULL/PARTIAL/NONE, so two
properties matter more than formatting:

1. It must not leak system identity. `source` leaked once in the full sheet
   (printed as a venue fallback) and again here; because local-corpus hits carry
   quoted passages, printing "local_corpus" both reveals the retrieval path and
   lets the reviewer infer which arm they are reading.
2. The sample must include the HARD cases — title-only candidate lists — not just
   the convenient ones with quotes. A calibration that only sees easy evidence
   cannot tell you whether the procedure works on the rest.
"""

import json
from pathlib import Path

from eval.production_e2e.make_review_sample import build_sample, render


def _rec(sid, system="full_v2", target_type="idea", n_snip=0, n_cand=3,
         claim="a claim"):
    cands = []
    for i in range(n_cand):
        c = {"title": f"Paper {i}", "source": "local_corpus" if i < 1 else "openalex"}
        if i < n_snip:
            c["snippet"] = f"passage {i}"
        cands.append(c)
    return {"submission_id": sid, "system": system, "topic_id": "t1",
            "target_type": target_type, "claim": claim, "_topic": "topic",
            "candidate_papers": cands}


class TestSampleSelection:
    def test_covers_every_system(self):
        recs = ([_rec(f"d{i}", "direct_llm") for i in range(5)]
                + [_rec(f"r{i}", "retrieval_llm") for i in range(5)]
                + [_rec(f"f{i}", "full_v2") for i in range(5)])
        sample = build_sample(recs, per_system=2, seed=1)
        systems = {r["system"] for r in sample}
        assert systems == {"direct_llm", "retrieval_llm", "full_v2"}

    def test_includes_a_title_only_hard_case(self):
        """Hard cases are the point of calibrating first."""
        recs = ([_rec(f"e{i}", n_snip=3) for i in range(3)]
                + [_rec(f"h{i}", n_snip=0) for i in range(3)])
        sample = build_sample(recs, per_system=3, seed=2)
        assert any(not any(c.get("snippet") for c in r["candidate_papers"])
                   for r in sample), "sample must contain a title-only submission"

    def test_excludes_gap_targets(self):
        """Gaps are not what the idea-level headline scores."""
        recs = [_rec("g1", target_type="gap")] + [_rec("i1")]
        sample = build_sample(recs, per_system=3, seed=3)
        assert all(r["target_type"] == "idea" for r in sample)

    def test_is_deterministic_for_a_seed(self):
        recs = [_rec(f"x{i}") for i in range(9)]
        assert ([r["submission_id"] for r in build_sample(recs, 3, 7)]
                == [r["submission_id"] for r in build_sample(recs, 3, 7)])

    def test_handles_fewer_records_than_requested(self):
        recs = [_rec("only", "full_v2")]
        sample = build_sample(recs, per_system=5, seed=1)
        assert [r["submission_id"] for r in sample] == ["only"]


class TestRenderedOutput:
    def _render(self, tmp_path, recs):
        mapping = {"submissions": [
            {"submission_id": r["submission_id"], "system": r["system"],
             "topic_id": r["topic_id"], "target_type": r["target_type"]}
            for r in recs]}
        out = tmp_path / "sample.md"
        render(recs, mapping, out)
        return out.read_text(encoding="utf-8")

    def test_does_not_leak_system_identity_or_source(self, tmp_path):
        recs = [_rec("s1", "full_v2", n_snip=2)]
        text = self._render(tmp_path, recs)
        for leak in ("full_v2", "direct_llm", "retrieval_llm",
                     "local_corpus", "openalex", "arxiv"):
            assert leak not in text, f"sample leaked {leak!r}"

    def test_scrubs_a_source_name_stored_in_the_venue_field(self, tmp_path):
        """The leak's second path: `venue` itself holds the source name.

        Real records contain venues of literally "local_corpus" / "openalex"
        because the source is written where a venue belongs when none exists.
        A test that only varies `source` would miss this entirely.
        """
        rec = _rec("s1", n_snip=1)
        rec["candidate_papers"][0]["venue"] = "local_corpus"
        rec["candidate_papers"][1]["venue"] = "openalex"
        text = self._render(tmp_path, [rec])
        assert "local_corpus" not in text
        assert "openalex" not in text

    def test_keeps_a_genuine_venue(self, tmp_path):
        """Scrubbing must not blank out real venues."""
        rec = _rec("s1", n_snip=1)
        rec["candidate_papers"][0]["venue"] = "NeurIPS"
        text = self._render(tmp_path, [rec])
        assert "NeurIPS" in text

    def test_includes_the_fill_instructions_and_schema(self, tmp_path):
        text = self._render(tmp_path, [_rec("s1")])
        assert "human_verdicts.jsonl" in text
        assert '"verdict"' in text and "idea_credible" in text
        # The field names must match what evaluate.py reads.
        assert "false_open" in text and "true_gap" in text

    def test_warns_that_a_passage_may_be_off_topic(self, tmp_path):
        text = self._render(tmp_path, [_rec("s1", n_snip=1)])
        assert "about something else entirely" in text

    def test_marks_retrieval_failure_as_not_absence_of_prior_art(self, tmp_path):
        rec = _rec("s1", n_snip=0)
        rec["query_failure"] = "502 from gateway"
        rec["candidate_papers"] = []
        text = self._render(tmp_path, [rec])
        assert "RETRIEVAL FAILED" in text
        assert "NOT absence of prior art" in text

    def test_states_candidate_list_quality(self, tmp_path):
        text = self._render(tmp_path, [_rec("s1", n_snip=1, n_cand=4)])
        assert "4 items" in text
        assert "1 with a quoted passage" in text

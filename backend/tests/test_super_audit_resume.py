"""Super-audit resume tests.

Pins the two defects found on 2026-09-16 when repairing the 21 failed audit
targets:

1. The target key omitted the claim, so several ideas from one topic collapsed
   onto a single entry and the resume re-ran 69 targets that had already
   succeeded ("1 kept, 90 to re-run" instead of "70 kept, 21 to re-run").
2. `--resume` unlinked the results file at start-up and rebuilt it at the end.
   Killing the run in between destroyed the previous results with nothing
   written in their place. Records are now appended and the file is never
   removed up front.
"""

from pathlib import Path

from eval.production_e2e.super_audit import _load_prior_records, _target_key


def _rec(system="full_v2", topic_id="pe2e-001", target_type="idea", claim="c",
         query_failure=None):
    return {
        "sample_id": topic_id, "system": system, "topic_id": topic_id,
        "target_type": target_type, "claim": claim,
        "query_failure": query_failure, "candidate_papers": [],
    }


class TestTargetKey:
    def test_same_topic_different_claims_are_distinct(self):
        """The core regression: one topic yields several ideas."""
        a = _rec(claim="Idea A: use gradient norms")
        b = _rec(claim="Idea B: use spectral norms")
        assert _target_key(a) != _target_key(b)

    def test_identical_targets_share_a_key(self):
        assert _target_key(_rec(claim="same")) == _target_key(_rec(claim="same"))

    def test_claim_whitespace_does_not_split_a_key(self):
        assert _target_key(_rec(claim="x y")) == _target_key(_rec(claim="  x y  "))

    def test_different_systems_are_distinct(self):
        assert _target_key(_rec(system="direct")) != _target_key(_rec(system="full_v2"))

    def test_idea_and_gap_are_distinct(self):
        assert _target_key(_rec(target_type="idea")) != _target_key(_rec(target_type="gap"))

    def test_missing_claim_does_not_raise(self):
        rec = _rec()
        rec.pop("claim")
        assert _target_key(rec)[3] == ""


class TestLoadPriorRecords:
    def test_reads_records(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            '{"sample_id": "a", "query_failure": null}\n'
            '{"sample_id": "b", "query_failure": "boom"}\n',
            encoding="utf-8")
        recs = _load_prior_records(p)
        assert [r["sample_id"] for r in recs] == ["a", "b"]

    def test_tolerates_a_truncated_final_line(self, tmp_path):
        """A crash mid-append leaves a partial JSON object; it must be skipped.

        Dropping it is correct: that target has no usable candidates and will be
        re-run, whereas raising here would make the whole file unreadable.
        """
        p = tmp_path / "r.jsonl"
        p.write_text(
            '{"sample_id": "a"}\n{"sample_id": "b", "trunca',
            encoding="utf-8")
        recs = _load_prior_records(p)
        assert [r["sample_id"] for r in recs] == ["a"]

    def test_missing_file_is_empty(self, tmp_path):
        assert _load_prior_records(tmp_path / "nope.jsonl") == []

    def test_blank_lines_are_ignored(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text('{"sample_id": "a"}\n\n\n', encoding="utf-8")
        assert len(_load_prior_records(p)) == 1


class TestResumePartition:
    """Reproduces the keep/re-run decision over a realistic record set."""

    def test_partition_keeps_successes_and_reruns_failures(self):
        records = [
            _rec(claim="a"),                       # success
            _rec(claim="b"),                       # success
            _rec(claim="c", query_failure="502"),  # failed -> re-run
        ]
        by_key = {}
        for r in records:
            by_key.setdefault(_target_key(r), []).append(r)

        kept, rerun = [], []
        for target in records:                     # simulate the next run's targets
            bucket = by_key.get(_target_key(target)) or []
            rec = next((r for r in bucket if not r.get("query_failure")), None)
            if rec is not None:
                kept.append(rec)
            else:
                rerun.append(target)

        assert len(kept) == 2
        assert len(rerun) == 1
        assert rerun[0]["claim"] == "c"

    def test_several_ideas_per_topic_are_all_kept(self):
        """Guards the exact bug: three ideas on one topic must all survive."""
        records = [_rec(topic_id="pe2e-014", claim=f"idea {i}") for i in range(3)]
        by_key = {}
        for r in records:
            by_key.setdefault(_target_key(r), []).append(r)

        kept = []
        for target in records:
            bucket = by_key.get(_target_key(target)) or []
            rec = next((r for r in bucket if not r.get("query_failure")), None)
            if rec is not None:
                kept.append(rec)
        assert len(kept) == 3, "key collision would have collapsed these to one"

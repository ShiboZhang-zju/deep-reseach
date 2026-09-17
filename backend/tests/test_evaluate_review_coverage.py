"""Review-coverage accounting tests for the headline aggregation.

The bug these guard against: a partially reviewed sheet produced numbers that
looked final. `human_verdicts.jsonl` ships one row PER SUBMISSION with
`verdict: ""` meaning "not judged yet", and every row de-anonymises to a real
topic_id — so a plain `verdict_map.get(...)` lookup finds a record for EVERY
topic, and an unjudged row is silently scored as "not false-open" and "not
credible".

Measured consequence with 9 of 24 topics judged: the table reported
`verdicted: 24` and a 4.2% false-open rate whose denominator was 24 rather than
3. Counting only judged verdicts changed that cell to 33.3% — the difference
between "no meaningful edge" and "the baseline opens false gaps while V2 does
not", i.e. it changes the decision the run exists to inform.

The rules pinned here:
1. `verdict: ""` (or missing) is NOT a verdict — exclude it from every
   denominator instead of counting it as a negative.
2. Credible-yield denominator is judged topics, never all topics, so partial
   review cannot masquerade as low yield.
3. Coverage is reported explicitly, and a complete review must be
   distinguishable from an incomplete one.
"""

import json

from eval.production_e2e.evaluate import main as evaluate_main


def _prediction(topic_id, decision="propose_idea", protocol_flag=None):
    return {"sample_id": topic_id, "topic_id": topic_id, "decision": decision,
            "protocol_flag": protocol_flag, "stratum": "narrow_mature"}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _run_eval(tmp_path, monkeypatch, preds, verdicts, n_topics=3):
    # Keep the topic pool small and deterministic: stub load_topics so the test
    # exercises the accounting, not the dataset.
    topics = [{"topic_id": f"t{i}", "topic": "x", "stratum": "narrow_mature"}
              for i in range(n_topics)]
    monkeypatch.setattr("eval.production_e2e.evaluate.load_topics", lambda: topics)

    run_dir = tmp_path / "preds"
    run_dir.mkdir()
    _write(run_dir / "predictions.jsonl", preds)
    # `load_system_predictions` reads cfg["system"] at the TOP level.
    (run_dir / "config.json").write_text(
        json.dumps({"system": "full_v2"}), encoding="utf-8")

    audit = tmp_path / "audit"
    audit.mkdir()
    _write(audit / "human_verdicts.jsonl", verdicts)
    (audit / "submission_mapping.json").write_text(json.dumps({
        "submissions": [
            {"submission_id": v["submission_id"], "system": "full_v2",
             "topic_id": v["topic_id"], "target_type": "idea"}
            for v in verdicts]}), encoding="utf-8")

    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "evaluate", "--runs", str(run_dir), "--audit-dir", str(audit),
        "--run-id", "t", "--output-dir", str(out)])
    evaluate_main()
    return json.loads((out / "t" / "metrics.json").read_text(encoding="utf-8"))


class TestUnjudgedVerdictsAreExcluded:
    def test_blank_verdict_is_not_counted_as_a_negative(self, tmp_path, monkeypatch):
        """The core regression: 1 judged of 3 must yield a 1-topic denominator."""
        preds = [_prediction("t0"), _prediction("t1"), _prediction("t2")]
        verdicts = [
            {"submission_id": "s0", "topic_id": "t0", "verdict": "false_open",
             "idea_credible": False, "novelty": 2, "feasibility": 4},
            {"submission_id": "s1", "topic_id": "t1", "verdict": "",
             "idea_credible": False, "novelty": None, "feasibility": None},
            {"submission_id": "s2", "topic_id": "t2", "verdict": "",
             "idea_credible": False, "novelty": None, "feasibility": None},
        ]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        fog = m["full_v2"]["false_open_gap"]["idea_level"]
        assert fog["verdicted"] == 1, "only judged verdicts may count"
        assert fog["rate"] == 1.0, "1 false_open out of 1 judged"

    def test_true_gap_counts_in_the_denominator(self, tmp_path, monkeypatch):
        preds = [_prediction("t0"), _prediction("t1")]
        verdicts = [
            {"submission_id": "s0", "topic_id": "t0", "verdict": "false_open",
             "idea_credible": False, "novelty": 1, "feasibility": 5},
            {"submission_id": "s1", "topic_id": "t1", "verdict": "true_gap",
             "idea_credible": True, "novelty": 3, "feasibility": 4},
        ]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        fog = m["full_v2"]["false_open_gap"]["idea_level"]
        assert fog["verdicted"] == 2
        assert fog["rate"] == 0.5

    def test_missing_verdict_row_is_also_excluded(self, tmp_path, monkeypatch):
        """A topic with no row at all is unjudged, not a negative."""
        preds = [_prediction("t0"), _prediction("t1")]
        verdicts = [{"submission_id": "s0", "topic_id": "t0",
                     "verdict": "false_open", "idea_credible": False,
                     "novelty": 2, "feasibility": 3}]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        assert m["full_v2"]["false_open_gap"]["idea_level"]["verdicted"] == 1


class TestCredibleYieldDenominator:
    def test_denominator_is_judged_topics_not_all_topics(self, tmp_path, monkeypatch):
        """Partial review must not look like low yield."""
        preds = [_prediction("t0"), _prediction("t1"), _prediction("t2")]
        verdicts = [
            {"submission_id": "s0", "topic_id": "t0", "verdict": "true_gap",
             "idea_credible": True, "novelty": 3, "feasibility": 4},
            {"submission_id": "s1", "topic_id": "t1", "verdict": "",
             "idea_credible": False, "novelty": None, "feasibility": None},
            {"submission_id": "s2", "topic_id": "t2", "verdict": "",
             "idea_credible": False, "novelty": None, "feasibility": None},
        ]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        ciy = m["full_v2"]["credible_idea_yield"]
        assert ciy["judged_topics"] == 1
        assert ciy["rate"] == 1.0, "1 credible of 1 judged, not of 3 topics"

    def test_reports_coverage_and_completeness(self, tmp_path, monkeypatch):
        preds = [_prediction("t0"), _prediction("t1")]
        verdicts = [
            {"submission_id": "s0", "topic_id": "t0", "verdict": "true_gap",
             "idea_credible": True, "novelty": 3, "feasibility": 3},
            {"submission_id": "s1", "topic_id": "t1", "verdict": "",
             "idea_credible": False, "novelty": None, "feasibility": None},
        ]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        cov = m["full_v2"]["review_coverage"]
        assert cov["topics_judged"] == 1
        assert cov["topics_total"] == 2
        assert cov["complete"] is False

    def test_complete_review_is_marked_complete(self, tmp_path, monkeypatch):
        preds = [_prediction("t0")]
        verdicts = [{"submission_id": "s0", "topic_id": "t0",
                     "verdict": "true_gap", "idea_credible": True,
                     "novelty": 3, "feasibility": 3}]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        assert m["full_v2"]["review_coverage"]["complete"] is True

    def test_all_unjudged_yields_pending_not_zero(self, tmp_path, monkeypatch):
        """No verdicts must read as "pending", never as a 0% result."""
        preds = [_prediction("t0"), _prediction("t1")]
        verdicts = [{"submission_id": "s0", "topic_id": "t0", "verdict": "",
                     "idea_credible": False, "novelty": None, "feasibility": None},
                    {"submission_id": "s1", "topic_id": "t1", "verdict": "",
                     "idea_credible": False, "novelty": None, "feasibility": None}]
        m = _run_eval(tmp_path, monkeypatch, preds, verdicts)
        assert m["full_v2"]["false_open_gap"]["idea_level"]["rate"] is None
        assert m["full_v2"]["credible_idea_yield"]["rate"] is None
        assert m["full_v2"]["novelty_mean"] is None

"""Internal validity-regression aggregator tests (pytest output parsing /
group mapping). The real suite is NOT executed here — only parsing of canned
`pytest -v` output."""
from eval.internal import research_validity_regression as ivr

CANNED_OUTPUT = """
tests/test_research_validity_regression.py::test_run7_label_effect_is_not_self_preference PASSED [ 16%]
tests/test_research_validity_regression.py::test_run8_temporal_decay_is_not_contamination_velocity PASSED [ 33%]
tests/test_research_validity_regression.py::test_run10_control_declaration_must_be_implemented PASSED [ 50%]
tests/test_research_validity_regression.py::test_run10_control_check_fails_closed_without_embeddings PASSED [ 66%]
tests/test_research_validity_regression.py::test_run10_unstable_retrieval_rejects_strong_verdict PASSED [ 83%]
tests/test_research_validity_regression.py::test_run10_diminishing_return_stop FAILED [100%]
"""


def test_parse_pytest_verbose_output():
    results = ivr.parse_pytest_verbose_output(CANNED_OUTPUT)
    assert len(results) == 6
    assert results["test_run7_label_effect_is_not_self_preference"] == "PASSED"
    assert results["test_run10_diminishing_return_stop"] == "FAILED"


def test_group_results_maps_runs_and_flags_run12_gap():
    results = ivr.parse_pytest_verbose_output(CANNED_OUTPUT)
    groups = ivr.group_results(results)
    assert groups["run7"] == {"status": "pass", "passed": 1, "total": 1,
                              "tests": groups["run7"]["tests"]}
    assert groups["run8"]["status"] == "pass"
    assert groups["run10"]["status"] == "fail"
    assert groups["run10"]["passed"] == 3
    assert groups["run10"]["total"] == 4
    # run12 has no unit test in the suite; it must be reported, not hidden.
    assert groups["run12"]["status"] == "no_unit_test"
    assert "E2E" in groups["run12"]["note"]


def test_render_summary_totals():
    results = ivr.parse_pytest_verbose_output(CANNED_OUTPUT)
    groups = ivr.group_results(results)
    result = {
        "suite": ivr.TEST_FILE,
        "groups": groups,
        "totals": {"total_tests": 6, "passed": 5, "failed": 1, "errors": 0, "skipped": 0},
        "group_summary": {"pass": 2, "known_groups": 3},
        "return_code": 1,
    }
    summary = ivr.render_summary(result)
    assert "5/6 PASS" in summary
    assert "no_unit_test" in summary

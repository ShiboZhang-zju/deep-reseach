"""Internal research-validity regression aggregator.

Runs the EXISTING pytest regression suite
(tests/test_research_validity_regression.py) and groups the results by the
E2E run that motivated each case (run7 / run8 / run10 / run12). This module
does not re-implement any case and does not mix internal results with
external benchmark scores.

Known coverage (as of eval-harness-v1): run7 x1, run8 x1, run10 x4 unit
tests exist; the run12 case (Granger / temporal predictive relation is not a
causal mechanism) is validated in E2E run artifacts but has no unit test in
the suite — reported honestly as "no_unit_test".
"""
from __future__ import annotations

import re
import subprocess
import sys

from eval.config import BACKEND_DIR, EVAL_HARNESS_VERSION

TEST_FILE = "tests/test_research_validity_regression.py"

RUN_PREFIXES: dict[str, str] = {
    "run7": "test_run7_",
    "run8": "test_run8_",
    "run10": "test_run10_",
    "run12": "test_run12_",
}

# pytest -v lines look like:
#   tests/test_x.py::test_run7_label_effect PASSED [ 16%]
_RESULT_RE = re.compile(r"::(?P<name>test_\S+?)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED)")


def parse_pytest_verbose_output(output: str) -> dict[str, str]:
    """Extract {test_name: status} from `pytest -v --tb=no` output."""
    return {match.group("name"): match.group("status")
            for match in _RESULT_RE.finditer(output)}


def group_results(results: dict[str, str]) -> dict:
    """Group flat test results by E2E run prefix."""
    groups: dict[str, dict] = {}
    for run_name, prefix in RUN_PREFIXES.items():
        matched = {name: status for name, status in results.items()
                   if name.startswith(prefix)}
        if not matched:
            note = ("no unit tests in the regression suite for this run; "
                    "validated via E2E run artifacts" if run_name == "run12"
                    else "no tests matched this group prefix")
            groups[run_name] = {
                "status": "no_unit_test", "passed": 0, "total": 0, "note": note,
            }
            continue
        passed = sum(1 for status in matched.values() if status == "PASSED")
        groups[run_name] = {
            "status": "pass" if passed == len(matched) else "fail",
            "passed": passed,
            "total": len(matched),
            "tests": matched,
        }
    return groups


def run_validity_regression(timeout_seconds: int = 900) -> dict:
    """Run the regression suite in a subprocess and return the grouped result."""
    command = [
        sys.executable, "-m", "pytest", TEST_FILE,
        "-v", "--tb=no", "--no-header", "-p", "no:cacheprovider",
    ]
    proc = subprocess.run(
        command,
        cwd=str(BACKEND_DIR),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout_seconds,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    results = parse_pytest_verbose_output(output)
    totals = {
        "total_tests": len(results),
        "passed": sum(1 for status in results.values() if status == "PASSED"),
        "failed": sum(1 for status in results.values() if status == "FAILED"),
        "errors": sum(1 for status in results.values() if status == "ERROR"),
        "skipped": sum(1 for status in results.values() if status == "SKIPPED"),
    }
    groups = group_results(results)
    groups_pass = sum(1 for g in groups.values() if g["status"] == "pass")
    groups_known = sum(1 for g in groups.values() if g["status"] in ("pass", "fail"))
    return {
        "benchmark": "internal",
        "task": "research_validity_regression",
        "mode": "pytest_aggregate",
        "suite": TEST_FILE,
        "groups": groups,
        "totals": totals,
        "group_summary": {"pass": groups_pass, "known_groups": groups_known},
        "return_code": proc.returncode,
        "metrics_version": EVAL_HARNESS_VERSION,
    }


def render_summary(result: dict) -> str:
    lines = [
        "# Internal research-validity regression",
        "",
        f"Suite: `{result['suite']}` (existing tests, aggregated — not re-implemented)",
        "",
        f"**Validity Regression: {result['totals']['passed']}/{result['totals']['total_tests']} PASS** "
        f"(groups: {result['group_summary']['pass']}/{result['group_summary']['known_groups']} known groups pass)",
        "",
        "| run | status | passed/total |",
        "|---|---|---|",
    ]
    for run_name, group in result["groups"].items():
        note = f" ({group['note']})" if group.get("note") else ""
        lines.append(f"| {run_name} | {group['status']}{note} | {group['passed']}/{group['total']} |")
    lines += ["", f"pytest return code: {result['return_code']}", ""]
    return "\n".join(lines)

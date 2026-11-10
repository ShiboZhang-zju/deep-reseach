"""Sub-stage timing collector (probe df744606).

phase_service records ONE duration per phase. That says which phase is slow and
nothing about why — a phase is a lens fan-out plus a candidate loop plus several
LLM calls, and the recorded number collapses all of them. StageTimings lets a
phase attribute its own wall clock to named sub-stages.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.timing import StageTimings  # noqa: E402


def test_stage_accumulates_repeated_names():
    t = StageTimings()
    for _ in range(3):
        with t.stage("lens"):
            time.sleep(0.01)
    out = t.as_dict()["lens"]
    assert out["count"] == 3
    assert out["ms"] > 0
    assert out["seconds"] > 0
    # avg_ms is per-invocation, not per-total.
    assert 0 < out["avg_ms"] <= out["ms"]


def test_stage_records_even_when_body_raises():
    t = StageTimings()
    try:
        with t.stage("boom"):
            raise RuntimeError("inner failure")
    except RuntimeError:
        pass
    # A failed sub-stage still costs time; dropping it would understate the phase.
    assert t.as_dict()["boom"]["count"] == 1


def test_add_records_externally_measured_duration():
    t = StageTimings()
    t.add("gap:abcd1234", 1.5)
    t.add("gap:abcd1234", 0.5)
    out = t.as_dict()["gap:abcd1234"]
    assert out["count"] == 2
    assert out["seconds"] == 2.0
    assert out["ms"] == 2000
    assert out["avg_ms"] == 1000


def test_add_clamps_negative_durations():
    t = StageTimings()
    t.add("clock_skew", -3.0)
    assert t.as_dict()["clock_skew"]["seconds"] == 0.0


def test_as_dict_sorted_by_total_descending():
    t = StageTimings()
    t.add("small", 0.1)
    t.add("large", 9.0)
    t.add("medium", 1.0)
    assert list(t.as_dict()) == ["large", "medium", "small"]


def test_empty_collector_serializes_to_empty_dict():
    assert StageTimings().as_dict() == {}

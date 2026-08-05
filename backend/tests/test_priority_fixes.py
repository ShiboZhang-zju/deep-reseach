"""Integration-ish tests for the priority fixes (no network, no DB).

Covers:
- constrained_retrieval_mode detection (high-priority #2)
- gap-audit admission threshold selection under constrained mode
- confidence_tier computation + feasibility WARN demotion (O1)
- paper source is_available gating (mid-priority #4)
"""

from types import SimpleNamespace

import pytest

from app.config import settings
from app.agent.steps.generate_interventions import (
    _evaluate_hard_gates,
    _compute_confidence_tier,
    InterventionSchema,
)


def _candidate(method, cost):
    return InterventionSchema(
        intervention_type="evaluation_protocol",
        failure_mechanism="state change failure",
        proposed_intervention=method,
        intermediate_effect="separates state changes",
        measurable_outcome="accuracy",
        implementation_cost=cost,
        mechanism_confidence=0.8,
    )


# --- high-priority #2: constrained retrieval mode ---

def test_constrained_mode_true_when_no_key_no_email(monkeypatch):
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "")
    monkeypatch.setattr(settings, "openalex_email", "")
    monkeypatch.setattr(settings, "crossref_email", "")
    monkeypatch.setattr(settings, "openalex_api_key", "")
    assert settings.constrained_retrieval_mode is True
    monkeypatch.undo()


def test_constrained_mode_false_with_s2_key(monkeypatch):
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "abc")
    assert settings.constrained_retrieval_mode is False
    monkeypatch.undo()


def test_constrained_mode_false_with_polite_email(monkeypatch):
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "")
    monkeypatch.setattr(settings, "openalex_email", "me@x.com")
    assert settings.constrained_retrieval_mode is False
    monkeypatch.undo()


def test_effective_s2_rate_reflects_key(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_s2_per_min", 20)
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "")
    assert settings.effective_s2_rate_per_min == 20
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "abc")
    assert settings.effective_s2_rate_per_min == settings.s2_rate_with_key
    monkeypatch.undo()


# --- O1: feasibility WARN demotion + tier ---

def test_feasibility_warn_not_fail_for_auxiliary_training():
    """Training mentioned only in the cost (not core) with no GPU -> WARN, not FAIL."""
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(
        gap, audit, ["e1", "e2"],
        _candidate("Add a fixed-budget evaluation protocol", "may need light training of a small probe"),
        contract,
    )
    assert result["gate_statuses"]["feasibility"] == "WARN"


def test_core_training_no_gpu_still_fails():
    gap = SimpleNamespace()
    audit = SimpleNamespace(audit_result="confirmed", remaining_delta="untested state changes")
    contract = SimpleNamespace(allow_model_training=False, allow_large_benchmark=False, gpu_available=False)
    result = _evaluate_hard_gates(
        gap, audit, ["e1", "e2"],
        _candidate("Fine-tune a large state model", "requires GPU training"),
        contract,
    )
    assert result["gate_statuses"]["feasibility"] == "FAIL"


def test_confidence_tier_a_when_all_pass_and_fulltext():
    gap = SimpleNamespace(provenance_status="complete")
    assert _compute_confidence_tier(gap, {"evidence": "PASS", "novelty": "PASS", "feasibility": "PASS"}) == "A"


def test_confidence_tier_b_when_warn_or_abstract():
    gap_full = SimpleNamespace(provenance_status="complete")
    assert _compute_confidence_tier(gap_full, {"evidence": "PASS", "novelty": "PASS", "feasibility": "WARN"}) == "B"
    gap_abs = SimpleNamespace(provenance_status="partial")
    assert _compute_confidence_tier(gap_abs, {"evidence": "PASS", "novelty": "PASS", "feasibility": "PASS"}) == "B"


def test_confidence_tier_c_when_any_fail():
    gap = SimpleNamespace(provenance_status="complete")
    assert _compute_confidence_tier(gap, {"evidence": "PASS", "novelty": "PASS", "feasibility": "FAIL"}) == "C"


# --- mid-priority #4: source availability gating ---

def test_ieee_core_unavailable_without_key(monkeypatch):
    from app.paper_sources.ieee import IeeeSource
    from app.paper_sources.core import CoreSource
    monkeypatch.setattr(settings, "ieee_api_key", "")
    monkeypatch.setattr(settings, "core_api_key", "")
    assert IeeeSource().is_available() is False
    assert CoreSource().is_available() is False
    monkeypatch.undo()


def test_free_sources_always_available():
    from app.paper_sources.arxiv import ArxivSource
    from app.paper_sources.openalex import OpenAlexSource
    assert ArxivSource().is_available() is True
    assert OpenAlexSource().is_available() is True


def test_search_service_loads_only_available_sources(monkeypatch):
    monkeypatch.setattr(settings, "ieee_api_key", "")
    monkeypatch.setattr(settings, "core_api_key", "")
    from app.services.search_service import SearchService
    svc = SearchService()
    names = {s.name for s in svc.sources}
    assert "ieee" not in names and "core" not in names
    assert "arxiv" in names and "openalex" in names
    health = svc.source_health()
    assert health["active_count"] == len(svc.sources)
    monkeypatch.undo()

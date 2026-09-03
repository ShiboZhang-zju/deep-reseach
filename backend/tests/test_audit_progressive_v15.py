"""Tests for the v15 progressive adversarial audit (P0-1/P0-2).

Progressive mode is flag-gated (settings.gap_audit_progressive, default OFF):
legacy v14 semantics must stay byte-for-byte while the flag is off, so these
tests only exercise the flag-on paths of

  P0-1  repeated/zero-new-neighbor stop BEFORE neighbor full-text extraction
  P0-2  wave-based full-text work admission, stopping at the verdict ceiling
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_audit_gaps import _seed_gap, temp_db  # noqa: E402,F401


def _previous_pass_audit(db, gap, paper_ids, query_ids=None,
                         recommended_action="more_search",
                         claimed_delta=None):
    """Latest PASS audit asking for more_search (the pre-fulltext stop's trigger)."""
    from app.db.repositories import gap_repo

    return gap_repo.create_gap_audit(
        db, gap_id=gap.id, task_id=gap.task_id,
        adversarial_queries=["old killer query"],
        audit_result="uncertain", neighbor_paper_ids=paper_ids,
        recommended_action=recommended_action, rejection_reason=None,
        audit_round=1,
        search_policy_version="gap-search-admission-v15",
        search_admission_status="PASS", search_admission_reasons=[],
        search_query_ids=query_ids or [],
        audited_claimed_delta=(claimed_delta if claimed_delta is not None
                               else gap.claimed_delta),
        failure_reason_codes=[], evidence_delta={},
    )


def test_stop_fires_when_more_search_recalled_nothing_new(temp_db):
    from app.agent.steps.audit_gaps import _progressive_no_new_neighbor_stop

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    paper = gap_repo_paper(db, gap)
    _previous_pass_audit(db, gap, [paper.id])
    db.commit()

    assert _progressive_no_new_neighbor_stop(db, gap, [paper]) == \
        "NO_NEW_NEIGHBOR_EVIDENCE"


def gap_repo_paper(db, gap):
    from app.db.models import Paper
    from app.db.models import TaskPaper

    paper = Paper(title="Stable neighbor", abstract="Agent memory compression.",
                  citation_count=3)
    db.add(paper)
    db.flush()
    db.add(TaskPaper(task_id=gap.task_id, paper_id=paper.id, discovered_round=1,
                     final_score=0.9, priority="high"))
    db.commit()
    return paper


def test_stop_spared_when_new_neighbor_or_new_claim(temp_db):
    from app.agent.steps.audit_gaps import _progressive_no_new_neighbor_stop

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    seen = gap_repo_paper(db, gap)
    fresh = gap_repo_paper(db, gap)
    _previous_pass_audit(db, gap, [seen.id])
    db.commit()

    # A genuinely new neighbor keeps the full audit alive.
    assert _progressive_no_new_neighbor_stop(db, gap, [seen, fresh]) is None


def test_stop_spared_when_claim_changed(temp_db):
    from app.agent.steps.audit_gaps import _progressive_no_new_neighbor_stop

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    paper = gap_repo_paper(db, gap)
    _previous_pass_audit(db, gap, [paper.id],
                         claimed_delta=gap.claimed_delta + " (moved)")
    db.commit()
    assert _progressive_no_new_neighbor_stop(db, gap, [paper]) is None


def test_stop_spared_when_previous_was_not_more_search(temp_db):
    from app.agent.steps.audit_gaps import _progressive_no_new_neighbor_stop

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    paper = gap_repo_paper(db, gap)
    _previous_pass_audit(db, gap, [paper.id], recommended_action="reject")
    db.commit()
    assert _progressive_no_new_neighbor_stop(db, gap, [paper]) is None


def test_count_verified_neighbors(temp_db):
    from app.agent.steps.audit_gaps import _count_verified_neighbors
    from app.db.models import EvidenceUnit

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    paper = gap_repo_paper(db, gap)
    assert _count_verified_neighbors(db, [paper]) == 0
    db.add(EvidenceUnit(task_id=gap.task_id, paper_id=paper.id,
                        evidence_type="limitation",
                        normalized_claim="x",
                        verification_status="verified"))
    db.commit()
    assert _count_verified_neighbors(db, [paper]) == 1


@pytest.mark.asyncio
async def test_pre_fulltext_stop_rejects_gap_without_crashing(temp_db, monkeypatch):
    """P0 regression: the pre-fulltext stop branch referenced the not-yet-bound
    `failure_codes` (first bound only after full-text extraction) and raised
    UnboundLocalError, which escaped audit_gap_candidates (it catches only
    TimeoutError) and failed the whole task instead of rejecting one gap."""
    from app.config import settings
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.db.repositories import gap_repo
    from test_audit_gaps import ConfirmingAuditLLM, _pin_admission_and_neighbors

    db = temp_db()
    task, gap, _ = _seed_gap(db)
    neighbor = _pin_admission_and_neighbors(monkeypatch, db, gap)
    # The trigger: the latest PASS audit asked more_search for the same claim,
    # and this round surfaced none of its neighbors' new siblings.
    _previous_pass_audit(db, gap, [neighbor.id])
    monkeypatch.setattr(settings, "gap_audit_progressive", True)

    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, ConfirmingAuditLLM(), task.id,
                                         perform_search=False)

    assert [item.recommended_action for item in results] == ["reject"]
    assert gap_repo.get_gap(db, gap.id).status == "rejected"
    audit = gap_repo.list_gap_audits(db, gap.id)[-1]
    assert "NO_NEW_NEIGHBOR_EVIDENCE" in (audit.failure_reason_codes_json or "")
    db.close()


def _state(gap):
    from app.agent.state import ResearchState

    return ResearchState(task_id=gap.task_id, contract_id=gap.contract_id,
                         current_round=2)


@pytest.mark.asyncio
async def test_wave_admission_stops_at_ceiling(temp_db, monkeypatch):
    """4 pending neighbors, wave=2: only wave 1 is extracted because the
    monkeypatched verified-count reaches the ceiling (1) after wave 1."""
    from app.config import settings
    from app.agent.steps import audit_gaps

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    neighbors = [gap_repo_paper(db, gap) for _ in range(4)]

    monkeypatch.setattr(settings, "gap_audit_progressive", True)
    monkeypatch.setattr(settings, "audit_neighbor_evidence_wave_size", 2)

    downloads: list[list[str]] = []
    extracts: list[str] = []

    async def fake_download(papers, task_id):
        downloads.append([p.id for p, _ in papers])
        return {}

    async def fake_extract(task_id, paper, tp, pdf_path, llm, audit_round,
                           semaphore):
        extracts.append(paper.id)
        return 1

    verified_sequence = iter([0, 1])

    def fake_count(db_, neighbors_):
        try:
            return next(verified_sequence)
        except StopIteration:
            return 1

    from app.agent.steps import extract_evidence

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fake_download)
    monkeypatch.setattr(extract_evidence, "_extract_from_paper_safe", fake_extract)
    monkeypatch.setattr(audit_gaps, "_count_verified_neighbors", fake_count)

    await audit_gaps._ensure_neighbor_evidence(
        db, _state(gap), None, gap.task_id, gap, neighbors, 2)

    assert len(downloads) == 1 and len(downloads[0]) == 2
    assert len(extracts) == 2


@pytest.mark.asyncio
async def test_wave_admission_skips_when_ceiling_already_met(temp_db, monkeypatch):
    """Re-audit with a stable neighbor set: verified neighbors already satisfy
    the ceiling -> zero downloads (the purest 'stop re-paying' case)."""
    from app.config import settings
    from app.agent.steps import audit_gaps

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    neighbors = [gap_repo_paper(db, gap) for _ in range(2)]

    monkeypatch.setattr(settings, "gap_audit_progressive", True)
    monkeypatch.setattr(audit_gaps, "_count_verified_neighbors",
                        lambda db_, ns: 1)

    async def fail_download(papers, task_id):
        raise AssertionError("must not download when the ceiling is already met")

    from app.agent.steps import extract_evidence

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fail_download)

    await audit_gaps._ensure_neighbor_evidence(
        db, _state(gap), None, gap.task_id, gap, neighbors, 2)


@pytest.mark.asyncio
async def test_legacy_mode_extracts_all_pending(temp_db, monkeypatch):
    """Flag off: one gather over every pending neighbor (v14 semantics)."""
    from app.config import settings
    from app.agent.steps import audit_gaps

    db = temp_db()
    _, gap, _ = _seed_gap(db)
    neighbors = [gap_repo_paper(db, gap) for _ in range(4)]

    monkeypatch.setattr(settings, "gap_audit_progressive", False)

    downloads: list[list[str]] = []

    async def fake_download(papers, task_id):
        downloads.append([p.id for p, _ in papers])
        return {}

    async def fake_extract(task_id, paper, tp, pdf_path, llm, audit_round,
                           semaphore):
        return 1

    from app.agent.steps import extract_evidence

    monkeypatch.setattr(extract_evidence, "_download_pdfs", fake_download)
    monkeypatch.setattr(extract_evidence, "_extract_from_paper_safe", fake_extract)

    await audit_gaps._ensure_neighbor_evidence(
        db, _state(gap), None, gap.task_id, gap, neighbors, 2)

    assert len(downloads) == 1 and len(downloads[0]) == 4

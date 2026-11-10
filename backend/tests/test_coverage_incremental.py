"""Incremental coverage matching (2026-10-30, probe df744606).

Coverage used to re-match the WHOLE evidence pool against every question each
round (measured 246 -> 398 -> 509 -> 667s, monotonic with pool size). Evidence
rows are immutable and question text never changes after decompose, so
(question, evidence) verdicts are stable: only evidence created after the
previous coverage snapshot needs an LLM judgment; settled pairs replay from
their links.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.models import _utcnow  # noqa: E402  (shared UTC stamper)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    config.set_main_option("script_location",
                          os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


def _seed(db, n_units):
    from app.db.models import EvidenceUnit, Paper, ResearchContract, ResearchQuestion, ResearchTask

    task = ResearchTask(user_input="incremental coverage", status="updating_coverage")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Coverage", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="How does pool growth affect matching cost?",
        question_type="evaluation", importance=0.9, status="open")
    db.add(question)
    db.flush()
    paper = Paper(title="Paper A", abstract="abstract")
    db.add(paper)
    db.flush()
    units = []
    for i in range(n_units):
        eu = EvidenceUnit(
            task_id=task.id, paper_id=paper.id, evidence_type="limitation",
            normalized_claim=f"claim {i}",
            extraction_method="pdf_fulltext", verification_status="verified")
        db.add(eu)
        units.append(eu)
    db.commit()
    return task, question, units


def _make_counting_matcher(seen):
    from app.agent.steps import update_coverage as module

    async def _match(llm, question, evidence):
        seen.append([e.id for e in evidence])
        # Everything matched as supporting at a fixed relevance.
        return [(i, "supports", 0.9) for i in range(len(evidence))]

    return _match


def _mark_coverage_phase_done(db, task_id, round_number):
    """Insert the PhaseRun that production writes around a coverage phase.

    The watermark is phase_runs.completed_at of the previous coverage phase —
    coverage_records.created_at cannot serve as one (it is upserted per
    (task, question, round) and stamped once, which pinned the watermark to
    round 1 and silently re-matched every later round in full).
    """
    from app.db.models import PhaseRun

    db.add(PhaseRun(
        task_id=task_id,
        phase_name=f"update_coverage_round_{round_number}",
        status="completed",
        completed_at=_utcnow(),
        round_number=round_number,
    ))
    db.commit()


@pytest.mark.asyncio
async def test_second_round_only_matches_new_evidence(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module
    from app.db.models import CoverageRecord, QuestionEvidenceLink

    db = temp_db()
    task, question, units = _seed(db, 4)
    seen_r1 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r1))
    state = ResearchState(task_id=task.id, current_round=1)
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=1)

    # Round 1 saw all 4 units (first run: no previous snapshot).
    assert len(seen_r1) == 1 and len(seen_r1[0]) == 4
    _mark_coverage_phase_done(db, task.id, 1)

    # Add NEW evidence after the round-1 coverage phase completed.
    from app.db.models import EvidenceUnit
    new_eu = EvidenceUnit(
        task_id=task.id, paper_id=units[0].paper_id, evidence_type="limitation",
        normalized_claim="brand new claim",
        extraction_method="pdf_fulltext", verification_status="verified")
    db.add(new_eu)
    db.commit()

    seen_r2 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r2))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=2)

    # Round 2 asked the LLM about ONLY the new unit...
    assert len(seen_r2) == 1 and seen_r2[0] == [new_eu.id]
    # ...while the links still cover all 5 units (4 replayed + 1 new).
    links = db.query(QuestionEvidenceLink).filter(
        QuestionEvidenceLink.question_id == question.id).all()
    assert len(links) == 5
    assert all(l.relation_type == "supports" for l in links)
    # Coverage snapshot exists for round 2 and reflects all 5 supporting units.
    cr = db.query(CoverageRecord).filter(
        CoverageRecord.question_id == question.id,
        CoverageRecord.round_number == 2).one()
    assert cr.supporting_evidence_count == 5
    db.close()


@pytest.mark.asyncio
async def test_no_new_evidence_means_zero_llm_calls(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module
    from app.db.models import CoverageRecord, QuestionEvidenceLink

    db = temp_db()
    task, question, units = _seed(db, 3)
    seen_r1 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r1))
    state = ResearchState(task_id=task.id, current_round=1)
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=1)
    r1_links = {(l.evidence_id, l.relation_type, l.relevance_score) for l in
                db.query(QuestionEvidenceLink).filter(
                    QuestionEvidenceLink.question_id == question.id).all()}
    cr1 = db.query(CoverageRecord).filter(
        CoverageRecord.question_id == question.id).one()
    _mark_coverage_phase_done(db, task.id, 1)

    seen_r2 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r2))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=2)

    # Zero LLM matching calls: nothing new to judge.
    assert seen_r2 == []
    # Links and coverage replay identically from the settled snapshot.
    r2_links = {(l.evidence_id, l.relation_type, l.relevance_score) for l in
                db.query(QuestionEvidenceLink).filter(
                    QuestionEvidenceLink.question_id == question.id).all()}
    assert r2_links == r1_links
    cr2 = db.query(CoverageRecord).filter(
        CoverageRecord.question_id == question.id,
        CoverageRecord.round_number == 2).one()
    assert cr2.coverage_score == cr1.coverage_score
    assert cr2.supporting_evidence_count == cr1.supporting_evidence_count
    db.close()


@pytest.mark.asyncio
async def test_escape_hatch_restores_full_rematching(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module
    from app.config import settings

    db = temp_db()
    task, question, units = _seed(db, 3)
    seen_r1 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r1))
    state = ResearchState(task_id=task.id, current_round=1)
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=1)

    monkeypatch.setattr(settings, "coverage_incremental_matching", False)
    seen_r2 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r2))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=2)

    # Full re-match: the LLM saw all 3 units again.
    assert len(seen_r2) == 1 and len(seen_r2[0]) == 3
    db.close()


@pytest.mark.asyncio
async def test_watermark_advances_with_each_coverage_phase(temp_db, monkeypatch):
    """Regression: r3 must not re-match r2's evidence.

    The first implementation derived the watermark from
    coverage_records.created_at, which is stamped once per (task, question)
    and upserted thereafter — so it stayed pinned at round 1 and every later
    round re-matched the entire pool (observed live: r6 reported reused=1804,
    byte-identical to r2, and cost 398s against r1's 268s). The watermark now
    comes from phase_runs.completed_at of the previous coverage phase.
    """
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module
    from app.db.models import EvidenceUnit

    db = temp_db()
    task, question, units = _seed(db, 2)
    state = ResearchState(task_id=task.id, current_round=1)

    seen = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=1)
    assert len(seen) == 1 and len(seen[0]) == 2
    _mark_coverage_phase_done(db, task.id, 1)

    r2_eu = EvidenceUnit(task_id=task.id, paper_id=units[0].paper_id,
                         evidence_type="limitation", normalized_claim="r2 claim",
                         extraction_method="pdf_fulltext", verification_status="verified")
    db.add(r2_eu)
    db.commit()

    seen_r2 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r2))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=2)
    assert len(seen_r2) == 1 and seen_r2[0] == [r2_eu.id]
    _mark_coverage_phase_done(db, task.id, 2)

    r3_eu = EvidenceUnit(task_id=task.id, paper_id=units[0].paper_id,
                         evidence_type="limitation", normalized_claim="r3 claim",
                         extraction_method="pdf_fulltext", verification_status="verified")
    db.add(r3_eu)
    db.commit()

    seen_r3 = []
    monkeypatch.setattr(module, "_llm_match_evidence", _make_counting_matcher(seen_r3))
    await module.update_coverage_matrix(db, state, object(), task.id, round_number=3)

    # Round 3 asks ONLY about its own new unit — r1/r2 units stay settled.
    assert len(seen_r3) == 1 and seen_r3[0] == [r3_eu.id]
    db.close()

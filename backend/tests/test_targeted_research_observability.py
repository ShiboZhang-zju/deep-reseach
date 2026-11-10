"""Regression: O2 remediation rounds must leave phase_runs + duration traces.

README §19 bottleneck #2: the primary search loop wraps extract/coverage in
phase_service.execute_phase (runner.py::_run_search_loop), but
run_targeted_research_round called the steps directly — remediation-round
evidence extraction produced no phase_runs record, no duration_ms, and had no
idempotence guard. These tests pin the wrapped behavior.
"""

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
    cfg = AlembicConfig()
    cfg.set_main_option('sqlalchemy.url', f'sqlite:///{db_path}')
    script_loc = os.path.join(os.path.dirname(__file__), "..", "alembic_migrations")
    cfg.set_main_option('script_location', script_loc)
    alembic_command.upgrade(cfg, 'head')
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    yield engine, Session
    engine.dispose()
    import gc
    gc.collect()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


async def _run_remediation(temp_db, monkeypatch, extract_ret=3, coverage_ret=0):
    """Drive run_targeted_research_round with all network/LLM steps mocked."""
    engine, Session = temp_db
    from app.db.models import ResearchTask
    from app.agent.state import ResearchState
    from app.agent.steps import targeted_research
    from app.config import settings

    db = Session()
    task_id = "obs-test-task-1"
    task = ResearchTask(id=task_id, user_input="agent memory", normalized_topic="agent memory")
    db.add(task)
    db.commit()

    state = ResearchState(task_id=task_id, normalized_topic="agent memory")

    async def fake_search(db, state, executions, task_id, round_num):
        return (5, [], [])

    async def fake_score(db, state, llm, task_id, round_num):
        return None

    async def fake_extract(db, state, llm, task_id, round_num):
        return extract_ret

    async def fake_coverage(db, state, llm, task_id, round_num):
        return coverage_ret

    monkeypatch.setattr(
        targeted_research, "save_search_query",
        lambda db, task_id, query, intent, question_id, source, round_num: SimpleNamespace(id=1),
    )
    monkeypatch.setattr(targeted_research, "search_and_save_papers", fake_search)
    monkeypatch.setattr(targeted_research, "score_papers", fake_score)
    monkeypatch.setattr(targeted_research, "extract_evidence_units", fake_extract)
    monkeypatch.setattr(targeted_research, "update_coverage_matrix", fake_coverage)

    result = await targeted_research.run_targeted_research_round(
        db, state, llm=None, task_id=task_id,
        reason="no_evidence_backed_gap_candidates",
    )
    return db, task_id, result, settings.max_rounds + 1


async def test_remediation_creates_phase_runs_with_duration(temp_db, monkeypatch):
    db, task_id, result, round_num = await _run_remediation(temp_db, monkeypatch)

    assert result.attempted is True
    assert result.new_evidence_count == 3

    from app.db.models import PhaseRun

    phases = {pr.phase_name: pr for pr in db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id).all()}
    ev_name = f"extract_evidence_round_{round_num}"
    cov_name = f"update_coverage_round_{round_num}"
    assert ev_name in phases, f"missing phase_runs record for remediation extraction: {sorted(phases)}"
    assert cov_name in phases, f"missing phase_runs record for remediation coverage: {sorted(phases)}"
    for name in (ev_name, cov_name):
        assert phases[name].status == "completed", f"{name}: {phases[name].status}"
        assert phases[name].input_version, f"{name}: empty input_version"
        assert phases[name].output_version, f"{name}: empty output_version"
    db.close()


async def test_remediation_phase_duration_trace_populated(temp_db, monkeypatch):
    db, task_id, result, round_num = await _run_remediation(temp_db, monkeypatch)
    assert result.attempted is True

    from app.db.models import AgentTrace

    duration_traces = db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id,
        AgentTrace.step_type == "phase_duration",
    ).all()
    trace_names = {t.step_name for t in duration_traces}
    assert f"extract_evidence_round_{round_num}" in trace_names
    assert f"update_coverage_round_{round_num}" in trace_names
    for t in duration_traces:
        assert t.duration_ms is not None and t.duration_ms >= 0, (
            f"phase_duration trace without duration_ms: {t.step_name}")
    db.close()


async def test_remediation_extraction_failure_is_non_fatal(temp_db, monkeypatch):
    """Extraction raising inside execute_phase must not break the round.

    execute_phase already marks the PhaseRun failed and re-raises; the
    remediation wrapper catches it non-fatally (same contract as before the
    change — remediation evidence failure never aborted the task).
    """
    engine, Session = temp_db
    from app.db.models import ResearchTask
    from app.agent.state import ResearchState
    from app.agent.steps import targeted_research

    db = Session()
    task_id = "obs-test-task-2"
    db.add(ResearchTask(id=task_id, user_input="x", normalized_topic="x"))
    db.commit()
    state = ResearchState(task_id=task_id, normalized_topic="x")

    async def boom(db, state, llm, task_id, round_num):
        raise RuntimeError("extraction exploded")

    async def fake_search(db, state, executions, task_id, round_num):
        return (5, [], [])

    monkeypatch.setattr(
        targeted_research, "save_search_query",
        lambda db, task_id, query, intent, question_id, source, round_num: SimpleNamespace(id=1),
    )
    monkeypatch.setattr(targeted_research, "search_and_save_papers", fake_search)
    monkeypatch.setattr(targeted_research, "score_papers", fake_search)
    monkeypatch.setattr(targeted_research, "extract_evidence_units", boom)

    result = await targeted_research.run_targeted_research_round(
        db, state, llm=None, task_id=task_id,
        reason="no_evidence_backed_gap_candidates",
    )
    assert result.attempted is True  # round completes; evidence failure is logged non-fatal

    from app.config import settings
    from app.db.models import PhaseRun
    ev_phase = db.query(PhaseRun).filter(
        PhaseRun.task_id == task_id,
        PhaseRun.phase_name == f"extract_evidence_round_{settings.max_rounds + 1}",
    ).one_or_none()
    assert ev_phase is not None
    assert ev_phase.status == "failed"
    db.close()

"""Coverage must reflect corroboration breadth, not evidence-unit count."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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


def _seed(db, *, paper_count, units_per_paper):
    """Create one question plus supporting evidence spread over N papers."""
    from app.db.models import (EvidenceUnit, Paper, ResearchContract, ResearchQuestion,
                               ResearchTask)

    task = ResearchTask(user_input="long horizon tool use", status="updating_coverage")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Tool Use", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="长任务中的错误累积如何被量化评估？",
        question_type="evaluation", importance=0.9, status="open")
    db.add(question)
    db.flush()
    for paper_index in range(paper_count):
        paper = Paper(title=f"Paper {paper_index}", abstract="abstract")
        db.add(paper)
        db.flush()
        for unit_index in range(units_per_paper):
            db.add(EvidenceUnit(
                task_id=task.id, paper_id=paper.id, evidence_type="limitation",
                normalized_claim=f"claim {paper_index}-{unit_index}",
                extraction_method="pdf_fulltext", verification_status="verified"))
    db.commit()
    return task, question


async def _run_with_all_supporting(db, monkeypatch, task_id):
    """Run the coverage update with every evidence unit counted as supporting."""
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module

    async def _all_supports(llm, question, evidence_index):
        return [(index, "supports", 0.9) for index in range(len(evidence_index))]

    monkeypatch.setattr(module, "_llm_match_evidence", _all_supports)
    state = ResearchState(task_id=task_id, current_round=1)
    return await module.update_coverage_matrix(db, state, object(), task_id, round_number=1)


@pytest.mark.asyncio
async def test_one_paper_cannot_saturate_a_question(temp_db, monkeypatch):
    """12 units from a single paper is weak corroboration, not full coverage.

    With the old unit-count formula (0.4 + supporting * 0.1) this scored a
    perfect 1.00, and a real round had 6 of 10 questions at 1.00 after 15 papers.
    """
    db = temp_db()
    task, question = _seed(db, paper_count=1, units_per_paper=12)

    deltas = await _run_with_all_supporting(db, monkeypatch, task.id)

    assert len(deltas) == 1
    assert deltas[0]["distinct_supporting_papers"] == 1
    assert deltas[0]["supporting"] == 12
    assert deltas[0]["new_coverage"] < 0.7, "a single paper must not mark a question covered"
    db.close()


@pytest.mark.asyncio
async def test_many_papers_do_reach_full_coverage(temp_db, monkeypatch):
    db = temp_db()
    task, question = _seed(db, paper_count=8, units_per_paper=1)

    deltas = await _run_with_all_supporting(db, monkeypatch, task.id)

    assert deltas[0]["distinct_supporting_papers"] == 8
    assert deltas[0]["new_coverage"] == 1.0
    assert deltas[0]["status"] == "covered"
    db.close()


@pytest.mark.asyncio
async def test_same_unit_count_scores_higher_when_spread_across_papers(temp_db, monkeypatch):
    """The score must separate 12-from-1 from 12-from-6, which used to tie at 1.00."""
    db_narrow = temp_db()
    task_narrow, _ = _seed(db_narrow, paper_count=1, units_per_paper=12)
    narrow = await _run_with_all_supporting(db_narrow, monkeypatch, task_narrow.id)

    db_broad = temp_db()
    task_broad, _ = _seed(db_broad, paper_count=6, units_per_paper=2)
    broad = await _run_with_all_supporting(db_broad, monkeypatch, task_broad.id)

    assert narrow[0]["supporting"] == broad[0]["supporting"] == 12
    assert broad[0]["new_coverage"] > narrow[0]["new_coverage"]
    db_narrow.close()
    db_broad.close()


async def _run_with_relation(db, monkeypatch, task_id, relation, relevance=0.9):
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module

    async def _matcher(llm, question, evidence_index):
        return [(index, relation, relevance) for index in range(len(evidence_index))]

    monkeypatch.setattr(module, "_llm_match_evidence", _matcher)
    state = ResearchState(task_id=task_id, current_round=1)
    return await module.update_coverage_matrix(db, state, object(), task_id, round_number=1)


@pytest.mark.asyncio
async def test_partially_answers_corroborates_a_question(temp_db, monkeypatch):
    """Coverage and gap mining must agree on what counts as support.

    The matcher's prompt defines `partially_answers` as "partially answers /
    points out a limitation", and the heuristic fallback maps every `limitation`
    unit to it. Coverage used to file it under background: a real run returned 21
    such links and zero `supports`, leaving all twelve questions at 0.15 while the
    same evidence was admissible for mining.
    """
    db = temp_db()
    task, _ = _seed(db, paper_count=6, units_per_paper=1)

    deltas = await _run_with_relation(db, monkeypatch, task.id, "partially_answers")

    assert deltas[0]["supporting"] == 6
    assert deltas[0]["distinct_supporting_papers"] == 6
    assert deltas[0]["new_coverage"] > 0.15
    db.close()


@pytest.mark.asyncio
async def test_weak_matches_do_not_corroborate(temp_db, monkeypatch):
    """Below the shared relevance floor, a link is background for both consumers."""
    db = temp_db()
    task, _ = _seed(db, paper_count=6, units_per_paper=1)

    deltas = await _run_with_relation(db, monkeypatch, task.id, "supports", relevance=0.2)

    assert deltas[0]["supporting"] == 0
    assert deltas[0]["distinct_supporting_papers"] == 0
    db.close()


def test_both_consumers_share_one_relation_definition():
    """The split that caused the 0.15-vs-1.00 discrepancy must not reappear."""
    from app.agent.evidence_relations import MIN_RELATION_RELEVANCE, SUPPORTING_RELATIONS
    from app.agent.steps import mine_gaps, update_coverage

    assert update_coverage.SUPPORTING_RELATIONS is SUPPORTING_RELATIONS
    assert mine_gaps.SUPPORTING_RELATIONS is SUPPORTING_RELATIONS
    assert update_coverage.MIN_RELATION_RELEVANCE == mine_gaps.MIN_RELATION_RELEVANCE
    assert MIN_RELATION_RELEVANCE == 0.5


@pytest.mark.asyncio
async def test_contradicting_evidence_lowers_coverage(temp_db, monkeypatch):
    """Refutation must subtract from coverage, never add to it.

    The formula used `+ distinct_contradicting * 0.05`, a sign error: an observed
    run returned coverage 0.95 for a question carrying 3 refuting papers, and a
    question refuted by 10 papers scored the same as one with a single weak
    match. `covered` is consumed downstream as permission to back a gap, so a
    contradicted question must not sail through on the strength of its rebuttals.
    """
    db_sup = temp_db()
    task_sup, _ = _seed(db_sup, paper_count=5, units_per_paper=1)
    sup_only = await _run_with_relation(db_sup, monkeypatch, task_sup.id, "supports")

    db_ref = temp_db()
    task_ref, _ = _seed(db_ref, paper_count=5, units_per_paper=1)
    # Same five papers corroborate, but three of them also refute.
    mixed = await _run_with_mixed(db_ref, monkeypatch, task_ref.id,
                                  support=2, contradict=3)

    assert mixed[0]["distinct_supporting_papers"] == 2
    assert mixed[0]["distinct_contradicting_papers"] == 3
    assert mixed[0]["new_coverage"] < sup_only[0]["new_coverage"], \
        "adding refuting papers must not raise coverage"
    db_sup.close()
    db_ref.close()


@pytest.mark.asyncio
async def test_purely_contradicted_question_is_not_confused_with_empty(temp_db, monkeypatch):
    """A fully refuted question carries information; it must beat an empty pool.

    Collapsing refutation onto the same score as "no evidence at all" is the
    "absence of evidence" error: the two facts are opposites and the downstream
    gap miner must be able to tell them apart.
    """
    db_ref = temp_db()
    task_ref, _ = _seed(db_ref, paper_count=3, units_per_paper=1)
    contradicted = await _run_with_relation(db_ref, monkeypatch, task_ref.id, "contradicts")

    assert contradicted[0]["distinct_supporting_papers"] == 0
    assert contradicted[0]["distinct_contradicting_papers"] == 3
    assert contradicted[0]["new_coverage"] > 0.0, \
        "three refuting papers are not the same finding as no evidence"
    db_ref.close()


@pytest.mark.asyncio
async def test_confidence_is_driven_by_corroboration_not_background_noise(temp_db, monkeypatch):
    """`confidence` must rest on the same papers the score rests on.

    It used `total_relevant * 0.15`, which counts background/weak matches that
    coverage excludes — so a question sat at coverage 0.15 while advertising
    confidence 0.60, vouching for a conclusion the score does not support.
    """
    db = temp_db()
    task, _ = _seed(db, paper_count=8, units_per_paper=1)
    deltas = await _run_with_relation(db, monkeypatch, task.id, "background")

    snap = _latest_snapshot(db, task.id)
    assert deltas[0]["supporting"] == 0
    assert snap.coverage_score == 0.15
    assert snap.confidence == 0.0, \
        "background-only links must not manufacture confidence"
    db.close()


def test_fallback_relation_keeps_comparison_as_partial_answer():
    """A failed LLM call must not demote measured comparisons to background.

    The heuristic fallback mapped `comparison` to `background`, silently pulling
    head-to-head results — the corroboration the extraction prompt asks for —
    out of `supporting`. A live run showed 16 comparison units and only 8
    background links, i.e. the downgrade was active in practice.
    """
    from app.agent.steps.update_coverage import _determine_relation

    class _Evidence:
        def __init__(self, evidence_type):
            self.evidence_type = evidence_type

    assert _determine_relation(_Evidence("comparison"), None) == "partially_answers"
    assert _determine_relation(_Evidence("limitation"), None) == "partially_answers"
    assert _determine_relation(_Evidence("negative_result"), None) == "contradicts"
    assert _determine_relation(_Evidence("future_work"), None) == "background"
    assert _determine_relation(_Evidence("result"), None) == "supports"


async def _run_with_mixed(db, monkeypatch, task_id, *, support, contradict):
    """Corroborate the first `support` papers and refute the next `contradict`."""
    from app.agent.state import ResearchState
    from app.agent.steps import update_coverage as module

    async def _matcher(llm, question, evidence_index):
        out = []
        for index in range(len(evidence_index)):
            out.append((index, "supports" if index < support else "contradicts", 0.9))
        return out

    monkeypatch.setattr(module, "_llm_match_evidence", _matcher)
    state = ResearchState(task_id=task_id, current_round=1)
    return await module.update_coverage_matrix(db, state, object(), task_id, round_number=1)


def _latest_snapshot(db, task_id):
    from app.db.models import CoverageRecord

    return (db.query(CoverageRecord)
            .filter(CoverageRecord.task_id == task_id)
            .order_by(CoverageRecord.created_at.desc())
            .first())

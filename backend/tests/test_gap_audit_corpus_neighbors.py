"""The audit may use corpus papers from the same questions as comparison material."""

import json
import os
import sys
import tempfile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
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


from app.agent.steps.audit_gaps import (GAP_SEARCH_POLICY_VERSION,
                                        collect_same_question_neighbors,
                                        evaluate_gap_search_admission,
                                        select_gap_specific_neighbors)
from app.db.models import (EvidenceUnit, Paper, QuestionEvidenceLink, ResearchQuestion,
                           SearchQueryPaper, SearchQueryRecord, TaskPaper)
from tests.test_audit_gaps import _seed_gap


def _attach_question_with_corpus(db, task, gap, paper_count, *, score=0.8):
    """Link the gap to a question whose evidence comes from `paper_count` papers."""
    question = ResearchQuestion(
        task_id=task.id, contract_id=gap.contract_id,
        question="长任务里的错误累积如何评估？", question_type="evaluation",
        importance=0.9, status="covered")
    db.add(question)
    db.flush()
    gap.question_ids_json = json.dumps([question.id])
    corpus_papers = []
    for index in range(paper_count):
        paper = Paper(title=f"Corpus {index}", abstract="corpus abstract")
        db.add(paper)
        db.flush()
        db.add(TaskPaper(task_id=task.id, paper_id=paper.id, final_score=score,
                         discovered_round=1))
        unit = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="comparison",
                            normalized_claim=f"corpus claim {index}",
                            verification_status="verified")
        db.add(unit)
        db.flush()
        db.add(QuestionEvidenceLink(question_id=question.id, evidence_id=unit.id,
                                    relation_type="supports", relevance_score=0.9))
        corpus_papers.append(paper)
    db.commit()
    return question, corpus_papers


def _add_queries(db, task, gap, *, families, papers, status="completed"):
    records = []
    for family in families:
        record = SearchQueryRecord(
            task_id=task.id, query_text=family, normalized_query_text=family,
            intent=f"gap_{family}", round_number=3, status=status, target_gap_id=gap.id,
            query_family=family, search_policy_version=GAP_SEARCH_POLICY_VERSION)
        db.add(record)
        records.append(record)
    db.flush()
    for index, record in enumerate(records):
        for paper in papers:
            db.add(SearchQueryPaper(query_id=record.id, paper_id=paper.id, rank=index,
                                    source="test", is_new_for_task=True))
    db.commit()
    return records


def test_corpus_papers_top_up_insufficient_retrieved_neighbours(temp_db):
    """Rate-limited sources must not make novelty undecidable when the corpus can help.

    Real run: three of four gaps sat at INSUFFICIENT_GAP_SPECIFIC_PAPERS with
    source_count=1 and two candidate papers, while 112 papers were already
    collected for the task.
    """
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    thin = [Paper(title="Only retrieved", abstract="q")]
    db.add_all(thin)
    db.flush()
    records = _add_queries(db, task, gap,
                           families=("exact_gap", "alternative_coverage", "claim_falsification"),
                           papers=thin)
    _attach_question_with_corpus(db, task, gap, paper_count=4)

    admission = evaluate_gap_search_admission(db, gap, [r.id for r in records])

    assert admission.status == "PASS"
    assert "INSUFFICIENT_GAP_SPECIFIC_PAPERS" not in admission.reason_codes
    assert len(admission.candidate_paper_ids) == 5
    db.close()


def test_corpus_papers_do_not_excuse_a_failed_search(temp_db):
    """Search-quality gates attest due diligence; the corpus cannot stand in."""
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    thin = [Paper(title="Only retrieved", abstract="q")]
    db.add_all(thin)
    db.flush()
    records = _add_queries(db, task, gap, families=("exact_gap",), papers=thin)
    _attach_question_with_corpus(db, task, gap, paper_count=6)

    admission = evaluate_gap_search_admission(db, gap, [r.id for r in records])

    assert admission.status == "UNKNOWN"
    assert "INSUFFICIENT_QUERY_FAMILIES" in admission.reason_codes
    db.close()


def test_gap_own_supporting_papers_are_never_neighbours(temp_db):
    """A claim cannot be checked against the evidence it was derived from."""
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    support_paper_ids = {
        db.get(EvidenceUnit, link.evidence_id).paper_id
        for link in __import__("app.db.repositories", fromlist=["gap_repo"]).gap_repo
        .list_gap_evidence(db, gap.id)
    }
    question = ResearchQuestion(
        task_id=task.id, contract_id=gap.contract_id, question="q", question_type="evaluation",
        importance=0.9, status="covered")
    db.add(question)
    db.flush()
    gap.question_ids_json = json.dumps([question.id])
    for paper_id in support_paper_ids:
        unit = EvidenceUnit(task_id=task.id, paper_id=paper_id, evidence_type="limitation",
                            normalized_claim="same paper again", verification_status="verified")
        db.add(unit)
        db.flush()
        db.add(QuestionEvidenceLink(question_id=question.id, evidence_id=unit.id,
                                    relation_type="supports", relevance_score=0.9))
    db.commit()

    assert collect_same_question_neighbors(db, gap) == []
    db.close()


def test_neighbour_selection_falls_back_to_corpus_when_retrieval_is_thin(temp_db):
    """Admission passing on corpus papers is useless if comparison cannot see them."""
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    thin = [Paper(title="Only retrieved", abstract="q")]
    db.add_all(thin)
    db.flush()
    records = _add_queries(db, task, gap,
                           families=("exact_gap", "alternative_coverage", "claim_falsification"),
                           papers=thin)
    _, corpus_papers = _attach_question_with_corpus(db, task, gap, paper_count=4)

    neighbours = select_gap_specific_neighbors(db, gap, [r.id for r in records])

    neighbour_ids = [paper.id for paper in neighbours]
    assert neighbour_ids[0] == thin[0].id, "retrieved papers keep priority"
    assert len(neighbour_ids) > 1
    assert set(neighbour_ids[1:]).issubset({paper.id for paper in corpus_papers})
    db.close()

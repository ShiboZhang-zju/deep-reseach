"""Regression tests for the gap-mining prompt/validation contract.

Observed in production (task 5e040ad5, 2026-08-11): gap mining admitted three
questions and 18 evidence units, the model returned two well-formed candidates,
and both were discarded as UNKNOWN_EVIDENCE_ID. None of the cited IDs were
fabricated — they were admitted evidence belonging to a *sibling* question.
The prompt pooled the admitted evidence of every passing question without ever
saying which question each unit belonged to, while validation demanded that all
cited evidence be linked to the questions the candidate named. The model was
being held to a constraint it was never shown, so gap=0/idea=0 was structural.
"""

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


class _GapLLM:
    """Emits one candidate whose citations are chosen by the test."""

    def __init__(self, question_ids, evidence_ids):
        self.question_ids = question_ids
        self.evidence_ids = evidence_ids
        self.prompts = []

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        self.prompts.append(messages[1]["content"])
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type="missing_evaluation",
            description="Fixed-budget state-change retention is never measured.",
            target_setting="Agent memory", observed_problem="State changes drop stored facts.",
            existing_coverage="Static question answering is measured.",
            missing_capability="State-change retention evaluation.",
            claimed_delta="Measure retention after a state change under one budget.",
            testable_hypothesis="Retention drops after a state change.",
            falsification_condition="A neighbour already reports this evaluation.",
            question_ids=list(self.question_ids),
            supporting_evidence_ids=list(self.evidence_ids),
        )])


def _seed_two_questions(db):
    """Two admissible questions, each backed by two papers with a limitation."""
    from app.db.models import (EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract,
                               ResearchQuestion, ResearchTask, TaskPaper)

    task = ResearchTask(user_input="agent memory")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()

    questions = []
    evidence_by_question = {}
    for index, text in enumerate(["Does memory survive state changes?",
                                  "Is retention measured under a fixed budget?"]):
        question = ResearchQuestion(task_id=task.id, contract_id=contract.id, question=text,
                                   question_type="failure", importance=0.9,
                                   status="partially_covered")
        db.add(question)
        db.flush()
        questions.append(question)

        units = []
        for offset, evidence_type in enumerate(["limitation", "comparison"]):
            paper = Paper(title=f"Paper {index}-{offset}", abstract="Memory evaluation")
            db.add(paper)
            db.flush()
            db.add(TaskPaper(task_id=task.id, paper_id=paper.id, discovered_round=1,
                             priority="high", final_score=0.9))
            unit = EvidenceUnit(
                task_id=task.id, paper_id=paper.id, evidence_type=evidence_type,
                normalized_claim=f"Claim {index}-{offset}",
                original_span="We do not evaluate memory after user state changes.",
                section="Limitations", page_number=4, page_start=4, page_end=4,
                span_start=10, span_end=66, source_chunk_hash=f"chunk-{index}-{offset}",
                verification_status="verified", extraction_confidence=0.9)
            db.add(unit)
            db.flush()
            db.add(QuestionEvidenceLink(question_id=question.id, evidence_id=unit.id,
                                        relation_type="supports", relevance_score=0.9))
            units.append(unit)
        evidence_by_question[question.id] = units

    db.commit()
    return task, contract, questions, evidence_by_question


def _last_trace(db, task_id, step_name):
    from app.db.models import AgentTrace

    return db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id, AgentTrace.step_name == step_name,
    ).order_by(AgentTrace.created_at.desc()).first()


@pytest.mark.asyncio
async def test_prompt_exposes_the_question_link_validation_enforces(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, questions, evidence = _seed_two_questions(db)
    own = evidence[questions[0].id]
    llm = _GapLLM([questions[0].id], [item.id for item in own])
    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2,
                          current_round=1)

    await mine_gap_candidates(db, state, llm, task.id)

    prompt = llm.prompts[0]
    assert "Linked questions:" in prompt
    for question in questions:
        assert f"Linked questions: {question.id}" in prompt, (
            "each evidence unit must disclose the question it is admitted for"
        )
    db.close()


@pytest.mark.asyncio
async def test_candidate_may_corroborate_with_sibling_question_evidence(temp_db):
    import json

    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, questions, evidence = _seed_two_questions(db)
    cited = [evidence[questions[0].id][0].id, evidence[questions[0].id][1].id,
             evidence[questions[1].id][0].id]
    llm = _GapLLM([questions[0].id], cited)
    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2,
                          current_round=1)

    created = await mine_gap_candidates(db, state, llm, task.id)

    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert trace["rejected_candidates"] == []
    assert len(created) == 1
    assert created[0].provenance_status == "complete"
    db.close()


@pytest.mark.asyncio
async def test_candidate_grounded_only_in_other_questions_is_rejected(temp_db):
    import json

    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, questions, evidence = _seed_two_questions(db)
    llm = _GapLLM([questions[0].id], [item.id for item in evidence[questions[1].id]])
    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2,
                          current_round=1)

    created = await mine_gap_candidates(db, state, llm, task.id)

    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert created == []
    assert [item["reason"] for item in trace["rejected_candidates"]] == [
        "EVIDENCE_NOT_LINKED_TO_CITED_QUESTION"]
    db.close()


@pytest.mark.asyncio
async def test_fabricated_evidence_id_is_reported_with_the_offending_ids(temp_db):
    import json

    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, questions, evidence = _seed_two_questions(db)
    llm = _GapLLM([questions[0].id],
                  [evidence[questions[0].id][0].id, "not-a-real-evidence-id"])
    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2,
                          current_round=1)

    created = await mine_gap_candidates(db, state, llm, task.id)

    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert created == []
    rejection = trace["rejected_candidates"][0]
    assert rejection["reason"] == "UNKNOWN_EVIDENCE_ID"
    assert rejection["unoffered_evidence_ids"] == ["not-a-real-evidence-id"], (
        "the trace must name the offending IDs so fabrication is distinguishable "
        "from a scope mismatch without re-running the model"
    )
    db.close()

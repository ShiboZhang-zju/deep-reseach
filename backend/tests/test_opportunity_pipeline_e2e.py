"""Deterministic offline E2E coverage for the opportunity pipeline."""

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
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass


class ScriptedLLM:
    def __init__(self):
        self.calls = []

    async def chat_json(self, messages, schema):
        self.calls.append(schema.__name__)
        if schema.__name__ == "GapCandidateList":
            from app.schemas.schemas import GapCandidateList, GapCandidateSchema
            question_id = next(line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Question ID:" in line)
            evidence_ids = [line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Evidence ID:" in line]
            return GapCandidateList(gaps=[GapCandidateSchema(
                gap_type="boundary_gap", description="Fixed budget state-change boundary remains unmeasured.",
                target_setting="Agent memory", observed_problem="State changes lose memory evidence.",
                existing_coverage="Generic question answering is measured.", missing_capability="State-change boundary evaluation.",
                claimed_delta="Measure state-change retention under a fixed budget.",
                testable_hypothesis="State-change accuracy is lower than stable-fact accuracy.",
                falsification_condition="A neighbor already reports the same evaluation.",
                question_ids=[question_id], supporting_evidence_ids=evidence_ids,
            )])
        if schema.__name__ == "GapAuditDecisionSchema":
            from app.agent.steps.audit_gaps import GapAuditDecisionSchema, NeighborAuditSchema
            import ast
            text = messages[1]["content"]
            evidence_id = ast.literal_eval(next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Supporting evidence IDs:")))[0]
            paper_id = next(line.split(": ", 1)[1] for line in text.splitlines() if "Paper ID:" in line)
            return GapAuditDecisionSchema(
                audit_result="confirmed", recommended_action="continue", remaining_delta="No state-change boundary evaluation.",
                evidence_for_gap_ids=[evidence_id], novelty_confidence=0.8, audit_confidence=0.8,
                comparisons=[NeighborAuditSchema(paper_id=paper_id, similarity_score=0.8, shared_problem="Memory", shared_mechanism="Compression", shared_evaluation="Generic QA", covered_claims=["generic QA"], uncovered_claims=["state-change boundary"], overlap_ratio=0.4, overlap_risk=0.3)],
            )
        if schema.__name__ == "InterventionList":
            from app.agent.steps.generate_interventions import InterventionList, InterventionSchema
            return InterventionList(interventions=[InterventionSchema(
                intervention_type="evaluation_protocol", failure_mechanism="Generic QA hides state changes.",
                proposed_intervention="Add a fixed-budget state-change evaluation.", intermediate_effect="Separates stable and changing states.",
                measurable_outcome="State-change accuracy.", implementation_cost="Low cost evaluation.", mechanism_confidence=0.8,
            )])
        if schema.__name__ == "MinimalExperimentSchema":
            from app.agent.steps.generate_minimal_experiments import MinimalExperimentSchema
            return MinimalExperimentSchema(title="State-change memory evaluation", summary="A minimal evaluation of state-change retention.", hypothesis="State changes reduce fixed-budget memory accuracy.", dataset="A small existing task subset", baselines="Existing memory baseline", metrics="State-change accuracy", controls=["same budget"], steps=["Construct stable and state-change examples", "Compare the same memory baseline"], success_condition="The difference is measurable.", falsification_condition="No difference appears.")
        raise AssertionError(f"Unexpected schema: {schema.__name__}")


@pytest.mark.asyncio
async def test_opportunity_pipeline_is_lineage_safe_and_idempotent(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.agent.steps.generate_interventions import generate_interventions
    from app.agent.steps.generate_minimal_experiments import generate_minimal_experiments
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract, ResearchIdea, ResearchQuestion, ResearchTask, TaskPaper

    db = temp_db()
    task = ResearchTask(user_input="agent memory")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active", version=1, input_hash="v1", gpu_available=False, allow_model_training=False)
    paper = Paper(title="Neighbor", abstract="Generic memory evaluation")
    db.add_all([contract, paper])
    db.flush()
    question = ResearchQuestion(task_id=task.id, contract_id=contract.id, question="Does memory fail after state changes?", question_type="failure", status="partially_covered")
    db.add(question)
    db.add(TaskPaper(task_id=task.id, paper_id=paper.id, discovered_round=1, priority="high", final_score=0.9))
    db.flush()
    evidence = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation", normalized_claim="State changes are not evaluated.", verification_status="verified", extraction_confidence=0.9)
    second_paper = Paper(title="Supporting Paper", abstract="Reports state-change limitation")
    db.add_all([evidence, second_paper])
    db.flush()
    second_evidence = EvidenceUnit(task_id=task.id, paper_id=second_paper.id, evidence_type="comparison", normalized_claim="Existing evaluations omit state-change boundary.", verification_status="verified", extraction_confidence=0.9)
    db.add(second_evidence)
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=evidence.id, relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id, relation_type="supports", relevance_score=0.9),
    ])
    db.commit()

    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2, current_round=1)
    llm = ScriptedLLM()
    gaps = await mine_gap_candidates(db, state, llm, task.id)
    audits = await audit_gap_candidates(db, state, llm, task.id, perform_search=False)
    interventions = await generate_interventions(db, state, llm, task.id)
    experiments = await generate_minimal_experiments(db, state, llm, task.id)

    idea = db.get(ResearchIdea, experiments.idea_ids[0])
    assert len(gaps) == len(audits) == len(interventions.passed_intervention_ids) == len(experiments.idea_ids) == 1
    assert idea.contract_id == contract.id
    assert idea.gap_id == gaps[0].id
    assert idea.intervention_id == interventions.passed_intervention_ids[0]
    assert idea.pipeline_version == 2
    assert idea.decision == "conditional_go"
    assert idea.final_score is None

    calls_before = list(llm.calls)
    assert await mine_gap_candidates(db, state, llm, task.id)
    assert len(llm.calls) == len(calls_before)
    db.close()

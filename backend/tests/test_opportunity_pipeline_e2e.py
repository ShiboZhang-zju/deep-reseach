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
            return MinimalExperimentSchema(title="State-change memory evaluation", summary="A minimal evaluation of state-change retention.", hypothesis="State changes reduce fixed-budget memory accuracy.", dataset="A small existing state-change task subset", baselines="Existing memory baseline", metrics="State-change accuracy", model_spec="3B inference-only model", dataset_provenance="Existing task subset; verify source availability", oracle="Executable held-out tests plus manual adjudication", statistical_analysis="Paired bootstrap confidence interval", resource_budget="One CPU run within 30 minutes", scenario_atoms=["state change"], controls=["same budget"], steps=["Construct stable and state-change examples", "Compare the same memory baseline"], success_condition="The difference is measurable.", falsification_condition="No difference appears.")
        if schema.__name__ == "IdeaScore":
            from app.schemas.schemas import IdeaScore
            return IdeaScore(novelty=0.8, feasibility=0.8, significance=0.8, evidence_support=0.8, differentiation=0.7, experimentability=0.9, potential_impact=0.8, risk=0.1, reason="Grounded in the confirmed state-change gap.")
        raise AssertionError(f"Unexpected schema: {schema.__name__}")


@pytest.mark.asyncio
async def test_opportunity_pipeline_is_lineage_safe_and_idempotent(temp_db, monkeypatch):
    from app.agent.state import ResearchState
    from app.agent.steps.audit_gaps import audit_gap_candidates
    from app.agent.steps.generate_interventions import generate_interventions
    from app.agent.steps.generate_minimal_experiments import generate_minimal_experiments
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract, ResearchIdea, ResearchQuestion, ResearchTask, SearchQueryPaper, TaskPaper

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
    evidence = EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation", normalized_claim="State changes are not evaluated.", original_span="We do not evaluate memory under changing user states.", section="Limitations", page_number=8, page_start=8, page_end=8, span_start=120, span_end=176, source_chunk_hash="e2e-fulltext-chunk-hash", verification_status="verified", extraction_confidence=0.9)
    second_paper = Paper(title="Supporting Paper", abstract="Reports state-change limitation")
    db.add_all([evidence, second_paper])
    db.flush()
    second_evidence = EvidenceUnit(task_id=task.id, paper_id=second_paper.id, evidence_type="comparison", normalized_claim="Existing evaluations omit state-change boundary.", verification_status="abstract_only", extraction_confidence=0.9)
    db.add(second_evidence)
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=evidence.id, relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id, relation_type="supports", relevance_score=0.9),
    ])
    # Neighbours the adversarial gap search is expected to surface. They must be
    # distinct from the papers backing the gap, otherwise admission reports
    # NO_EXTERNAL_NEIGHBOR. Three of them satisfy the strict (non-constrained)
    # admission threshold as well.
    neighbour_papers = [
        Paper(title=f"Adversarial Neighbor {index}", abstract="Generic memory benchmark without state changes")
        for index in range(3)
    ]
    db.add_all(neighbour_papers)
    db.commit()
    neighbour_ids = [item.id for item in neighbour_papers]

    async def _fake_gap_search(db, state, executions, task_id, round_number):
        """Stand in for the live search layer: mark every adversarial query
        completed and attach the neighbour papers, so the gap-search admission
        gate is exercised on realistic data instead of being short-circuited."""
        from app.db.repositories.search_query_repo import update_query_results
        for execution in executions:
            update_query_results(db, execution.query_id, len(neighbour_ids), 0, status="completed")
            for rank, paper_id in enumerate(neighbour_ids, start=1):
                db.add(SearchQueryPaper(query_id=execution.query_id, paper_id=paper_id,
                                        rank=rank, source="openalex", is_new_for_task=False))
        db.commit()

    monkeypatch.setattr("app.agent.steps.audit_gaps.search_and_save_papers", _fake_gap_search)

    state = ResearchState(task_id=task.id, contract_id=contract.id, pipeline_version=2, current_round=1)
    llm = ScriptedLLM()
    gaps = await mine_gap_candidates(db, state, llm, task.id)
    audits = await audit_gap_candidates(db, state, llm, task.id)
    from app.db.models import GapPhenomenonPlan
    db.add(GapPhenomenonPlan(task_id=task.id, contract_id=contract.id, gap_id=gaps[0].id,
                             phenomenon="State changes cause measurable memory degradation.",
                             mechanism_under_test="State-change retention under fixed budget.",
                             supports_gap_claim="State-change accuracy is lower than stable-fact accuracy.",
                             critical_unknown="Whether the drop exceeds baseline variance.",
                             expected_observation="Lower state-change accuracy.",
                             alternative_explanation="Task complexity differs.", comparator="H0 no accuracy gap; H1 lower state-change accuracy.",
                             oracle_experiment="Compare executable held-out tests.", kill_criterion="No meaningful accuracy gap.",
                             kill_criterion_basis="minimum_meaningful_effect: pre-registered effect.", measurement="state-change accuracy"))
    db.flush()
    interventions = await generate_interventions(db, state, llm, task.id)
    experiments = await generate_minimal_experiments(db, state, llm, task.id)

    assert [item.audit_result for item in audits] == ["confirmed"], (
        "gap audit must reach a decision once adversarial search admission passes"
    )
    assert len(gaps) == len(audits) == len(interventions.passed_intervention_ids) == len(experiments.idea_ids) == 1
    idea = db.get(ResearchIdea, experiments.idea_ids[0])
    assert idea.contract_id == contract.id
    assert idea.gap_id == gaps[0].id
    assert idea.intervention_id == interventions.passed_intervention_ids[0]
    assert idea.pipeline_version == 2
    assert idea.decision == "executable_candidate"
    assert idea.quality_reason_codes_json == "[]"
    assert idea.final_score is not None
    assert idea.final_score > 0.0

    calls_before = list(llm.calls)
    assert await mine_gap_candidates(db, state, llm, task.id)
    assert len(llm.calls) == len(calls_before)
    db.close()

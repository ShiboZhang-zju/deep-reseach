"""v7 Research Opportunity ontology regression tests (GAP_MINING v7).

Locks the three structural changes of opportunity-ontology-v7:

1. Seven opportunity types with DISTINCT evidence admission contracts — the
   old uniform ">=2 papers + limitation signal" rule structurally biased
   mining toward author-admitted limitations and could never surface untested
   assumptions, unexplained mechanisms, scaling blind spots or inter-paper
   contradictions, which are among the most valuable research opportunities.
2. Verified contradictions no longer block mining: they are admitted into
   the prompt pool ("Verified conflict" annotation) and mined as the
   contradiction opportunity type.
3. Lens fan-out: one LLM call per opportunity type; a candidate whose type
   does not match its lens call is rejected as TYPE_LENS_MISMATCH.
4. v7.1: the paper floor is per-type, not uniform — untested_assumption
   demands assumption prevalence (>=3 independent papers) and
   cross_domain_transfer demands two-sided evidence (source-domain mechanism
   + target-domain problem, declared explicitly).
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config
    from alembic import command
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


def _seed_question_with_evidence(db, evidence_specs):
    """One admissible question backed by the given evidence units.

    evidence_specs: list of (evidence_type, verification_status, fulltext,
    relation) tuples; each unit comes from its own paper so the two-paper
    floor is satisfiable from any two of them.
    """
    from app.db.models import (EvidenceUnit, Paper, QuestionEvidenceLink,
                               ResearchContract, ResearchQuestion, ResearchTask)

    task = ResearchTask(user_input="agent memory", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="Does retrieval similarity rank what actually matters?",
        question_type="method", importance=0.9, status="partially_covered",
    )
    db.add(question)
    db.flush()
    units = []
    for index, (evidence_type, status, fulltext, relation) in enumerate(evidence_specs):
        paper = Paper(title=f"Paper {index}")
        db.add(paper)
        db.flush()
        unit = EvidenceUnit(
            task_id=task.id, paper_id=paper.id, evidence_type=evidence_type,
            normalized_claim=f"Claim {index}: {evidence_type} observation",
            original_span=f"Original span {index}" if fulltext else None,
            section="Method" if evidence_type == "method" else "Results",
            page_number=4 if fulltext else None,
            span_start=10 if fulltext else None,
            span_end=60 if fulltext else None,
            source_chunk_hash=f"chunk-{index}" if fulltext else None,
            verification_status=status, extraction_confidence=0.9,
        )
        db.add(unit)
        db.flush()
        db.add(QuestionEvidenceLink(question_id=question.id, evidence_id=unit.id,
                                    relation_type=relation, relevance_score=0.9))
        units.append(unit)
    db.commit()
    return task, contract, question, units


class LensScriptedLLM:
    """One candidate per lens call; citations configured per opportunity type.

    Parses the lens type from the mining prompt ("Opportunity type for this
    call: X") and emits a candidate of exactly that type, so every lens is
    exercised in a single mining run. transfer_domains optionally fills the
    cross_domain_transfer two-sided fields; when omitted the transfer
    candidate lacks them (for the both-domains rejection test).
    """

    def __init__(self, citations_by_type, transfer_domains=None):
        self.citations_by_type = citations_by_type
        self.transfer_domains = transfer_domains
        self.lens_calls = []
        self.prompts = []

    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        assert schema.__name__ == "GapCandidateList", schema.__name__
        content = messages[1]["content"]
        self.prompts.append(content)
        gap_type = next(
            line.split(": ", 1)[1].split(" (")[0].strip()
            for line in content.splitlines()
            if line.startswith("Opportunity type for this call:"))
        question_id = next(line.split(": ", 1)[1].strip()
                           for line in content.splitlines() if "Question ID:" in line)
        self.lens_calls.append(gap_type)
        cited = self.citations_by_type.get(gap_type)
        if not cited:
            return GapCandidateList(gaps=[])
        extra = {}
        if gap_type == "cross_domain_transfer" and self.transfer_domains:
            extra["source_domain_evidence_ids"] = list(self.transfer_domains[0])
            extra["target_evidence_ids"] = list(self.transfer_domains[1])
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type=gap_type,
            description=f"{gap_type} opportunity in agent memory",
            target_setting="agent memory",
            observed_problem=f"observed problem specific to {gap_type}",
            existing_coverage="existing work covers the standard setting",
            missing_capability=f"capability missing for {gap_type}",
            claimed_delta=f"{gap_type}: establish the delta nobody has established",
            testable_hypothesis=f"a falsifiable hypothesis for {gap_type}",
            falsification_condition="a neighbour already covers this delta",
            question_ids=[question_id],
            supporting_evidence_ids=list(cited),
            **extra,
        )])


def _last_trace(db, task_id, step_name):
    from app.db.models import AgentTrace

    return db.query(AgentTrace).filter(
        AgentTrace.task_id == task_id, AgentTrace.step_name == step_name,
    ).order_by(AgentTrace.created_at.desc()).first()


@pytest.mark.asyncio
async def test_admission_contracts_are_type_specific(temp_db):
    """The same evidence pool must yield different outcomes per opportunity
    type: method+result evidence (no limitation anywhere) admits the
    assumption / contradiction / mechanism / scaling / transfer opportunities
    but NOT boundary or missing-evaluation ones, whose contracts genuinely
    demand a documented shortcoming. v7.1: the assumption admission is
    prevalence (>=3 papers sharing it), so the pool carries three independent
    papers.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import _OPPORTUNITY_CONTRACTS, mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("method", "verified", True, "supports"),
        ("result", "abstract_only", False, "supports"),
        ("method", "verified", True, "supports"),
    ])
    cited = [units[0].id, units[1].id, units[2].id]
    llm = LensScriptedLLM(
        {spec.gap_type: cited for spec in _OPPORTUNITY_CONTRACTS},
        transfer_domains=([units[0].id], [units[2].id]))
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    # Lens fan-out: one call per opportunity type, in ontology order.
    assert len(llm.lens_calls) == len(_OPPORTUNITY_CONTRACTS) == 7
    assert llm.lens_calls[0] == "boundary_gap"
    assert llm.lens_calls[1] == "missing_evaluation"
    assert {gap.gap_type for gap in gaps} == {
        "contradiction", "untested_assumption", "mechanism_gap",
        "scaling_regime", "cross_domain_transfer"}
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    rejections = {(item["lens"], item["reason"]) for item in trace["rejected_candidates"]}
    assert ("boundary_gap", "CANDIDATE_LACKS_LIMITATION_SIGNAL") in rejections
    assert ("missing_evaluation", "CANDIDATE_LACKS_LIMITATION_SIGNAL") in rejections
    db.close()


@pytest.mark.asyncio
async def test_assumption_requires_three_paper_prevalence(temp_db):
    """v7.1: the untested_assumption contract is assumption PREVALENCE, not
    the generic two-paper floor — two methods sharing a habit is coincidence,
    not a research opportunity. With only two independent papers behind the
    cited support the candidate is rejected with a prevalence-specific code.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("method", "verified", True, "supports"),
        ("result", "abstract_only", False, "supports"),
    ])
    llm = LensScriptedLLM({"untested_assumption": [units[0].id, units[1].id]})
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert gaps == []
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert [(item["lens"], item["reason"]) for item in trace["rejected_candidates"]] == [
        ("untested_assumption", "ASSUMPTION_NEEDS_BROADER_PREVALENCE")]
    db.close()


@pytest.mark.asyncio
async def test_transfer_requires_both_domain_evidence(temp_db):
    """v7.1: cross_domain_transfer is two-sided by contract — a documented
    source-domain mechanism AND a documented target-domain problem, both
    cited explicitly. Two arbitrary papers (or a one-sided citation) do not
    establish a transfer opportunity.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("method", "verified", True, "supports"),
        ("result", "verified", True, "supports"),
        ("comparison", "abstract_only", False, "supports"),
    ])
    cited = [units[0].id, units[1].id, units[2].id]
    # transfer_domains omitted: the candidate cites three papers but never
    # declares which side of the transfer each belongs to.
    llm = LensScriptedLLM({"cross_domain_transfer": cited})
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert gaps == []
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert [(item["lens"], item["reason"]) for item in trace["rejected_candidates"]] == [
        ("cross_domain_transfer", "TRANSFER_NEEDS_BOTH_DOMAIN_EVIDENCE")]
    db.close()


@pytest.mark.asyncio
async def test_missing_evaluation_accepts_future_work_signal(temp_db):
    """"we plan to evaluate X" is a documented unevaluation: future_work
    satisfies the missing_evaluation contract but not the boundary one.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import _OPPORTUNITY_CONTRACTS, mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("future_work", "verified", True, "supports"),
        ("method", "abstract_only", False, "supports"),
    ])
    cited = [units[0].id, units[1].id]
    llm = LensScriptedLLM({spec.gap_type: cited for spec in _OPPORTUNITY_CONTRACTS})
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert "missing_evaluation" in {gap.gap_type for gap in gaps}
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    rejections = {(item["lens"], item["reason"]) for item in trace["rejected_candidates"]}
    assert ("boundary_gap", "CANDIDATE_LACKS_LIMITATION_SIGNAL") in rejections
    db.close()


@pytest.mark.asyncio
async def test_contradiction_lens_mines_verified_conflict(temp_db):
    """The end-to-end contradiction path: a question carrying a verified
    conflicting link is admitted (not blocked), the conflicting unit is
    offered in the prompt with a "Verified conflict" annotation, and a
    contradiction candidate citing both sides of an inter-paper conflict is
    accepted.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import _OPPORTUNITY_CONTRACTS, mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        # Two supporting results that conflict with each other ...
        ("result", "verified", True, "supports"),
        ("result", "verified", True, "supports"),
        # ... plus a verified unit linked through a contradicts relation.
        ("comparison", "verified", True, "contradicts"),
    ])
    conflict_pair = [units[0].id, units[1].id]
    llm = LensScriptedLLM({"contradiction": conflict_pair})
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert [gap.gap_type for gap in gaps] == ["contradiction"]
    # The conflicting unit was offered and annotated, not hidden.
    contradiction_prompt = next(prompt for prompt in llm.prompts
                                if "Opportunity type for this call: contradiction" in prompt)
    assert units[2].id in contradiction_prompt
    assert "Verified conflict" in contradiction_prompt
    # Both sides of the conflict are linked to the created gap.
    from app.db.repositories import gap_repo
    linked = {link.evidence_id for link in gap_repo.list_gap_evidence(db, gaps[0].id)}
    assert set(conflict_pair) <= linked
    db.close()


@pytest.mark.asyncio
async def test_contradiction_requires_two_cited_claims(temp_db):
    """A contradiction is established by the conflicting claims themselves:
    citing a single unit cannot satisfy the contract, even from two papers.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import _OPPORTUNITY_CONTRACTS, mine_gap_candidates

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("result", "verified", True, "supports"),
        ("result", "verified", True, "supports"),
    ])
    llm = LensScriptedLLM({"contradiction": [units[0].id]})
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)

    gaps = await mine_gap_candidates(db, state, llm, task.id)

    assert gaps == []
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    assert [(item["lens"], item["reason"]) for item in trace["rejected_candidates"]] == [
        ("contradiction", "CONTRADICTION_NEEDS_CONFLICTING_CLAIM_PAIR")]
    db.close()


@pytest.mark.asyncio
async def test_wrong_type_candidate_is_rejected_as_lens_mismatch(temp_db):
    """A candidate whose type does not match its lens call is discarded
    without consuming an audit slot — the fan-out keeps lenses decorrelated.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import _OPPORTUNITY_CONTRACTS, mine_gap_candidates
    from app.schemas.schemas import GapCandidateList, GapCandidateSchema

    db = temp_db()
    task, contract, question, units = _seed_question_with_evidence(db, [
        ("result", "verified", True, "supports"),
        ("result", "verified", True, "supports"),
    ])

    class FixedTypeLLM:
        """Emits a contradiction candidate regardless of the requested lens."""

        async def chat_json(self, messages, schema):
            question_id = next(line.split(": ", 1)[1].strip()
                               for line in messages[1]["content"].splitlines()
                               if "Question ID:" in line)
            evidence_ids = [line.split(": ", 1)[1].strip()
                            for line in messages[1]["content"].splitlines()
                            if "Evidence ID:" in line]
            return GapCandidateList(gaps=[GapCandidateSchema(
                gap_type="contradiction",
                description="Longer context both helps and hurts recall.",
                target_setting="agent memory",
                observed_problem="two verified results conflict",
                existing_coverage="both sides are individually documented",
                missing_capability="explanation of the flip",
                claimed_delta="identify the variable that flips the conclusion",
                testable_hypothesis="context length above a boundary reverses the gain",
                falsification_condition="a neighbour already reconciles the two results",
                question_ids=[question_id],
                supporting_evidence_ids=evidence_ids,
            )])

    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=1)
    gaps = await mine_gap_candidates(db, state, FixedTypeLLM(), task.id)

    # Exactly one gap: the contradiction lens accepts its own type; the other
    # six calls' candidates are TYPE_LENS_MISMATCH.
    assert [gap.gap_type for gap in gaps] == ["contradiction"]
    trace = json.loads(_last_trace(db, task.id, "gap_mining_candidates").output_json)
    mismatches = [item for item in trace["rejected_candidates"]
                  if item["reason"] == "TYPE_LENS_MISMATCH"]
    assert len(mismatches) == len(_OPPORTUNITY_CONTRACTS) - 1
    assert {item["lens"] for item in mismatches} == {
        spec.gap_type for spec in _OPPORTUNITY_CONTRACTS} - {"contradiction"}
    db.close()


# --- Prompt evidence composition (type diversity + conflict slots) ------------


def _fake_evidence(item_id, paper_id, evidence_type):
    from types import SimpleNamespace

    return SimpleNamespace(
        id=item_id, paper_id=paper_id, evidence_type=evidence_type,
        verification_status="verified",
        extraction_confidence=0.8, normalized_claim="claim",
        original_span="span", source_chunk_hash="hash",
        page_number=3, page_start=None, span_start=10, span_end=40,
        section="results", conditions_json="{}",
    )


def test_prompt_evidence_interleaves_types_within_a_question():
    """v7: the shortlist must not let one abundant type (here: limitation)
    crowd every other type out of the per-question slots — the assumption and
    mechanism lenses mine method/result evidence that the old priority order
    would never have shown.
    """
    from app.agent.steps.mine_gaps import QuestionEvidenceAdmission, select_prompt_evidence

    evidence_by_id = {}
    ids = []
    for index in range(10):
        item = _fake_evidence(f"limit-{index:02d}", f"lp{index}", "limitation")
        evidence_by_id[item.id] = item
        ids.append(item.id)
    for index in range(4):
        item = _fake_evidence(f"method-{index:02d}", f"mp{index}", "method")
        evidence_by_id[item.id] = item
        ids.append(item.id)
    admission = QuestionEvidenceAdmission("q", "PASS", [], list(ids), list(ids))

    selected, _ = select_prompt_evidence(
        {"q": admission}, evidence_by_id, per_question=8, per_paper=2, total=40)

    types = {item.evidence_type for item in selected}
    assert types == {"limitation", "method"}, (
        "the per-question shortlist must span multiple evidence types, got "
        f"{sorted(types)}")


def test_verified_conflict_evidence_gets_reserved_prompt_slot():
    """v7: a verified conflicting unit is rare and high-signal; it must be
    offered even when the question's supporting pool alone fills the slots —
    otherwise the contradiction lens is asked to find a conflict it is never
    shown.
    """
    from app.agent.steps.mine_gaps import QuestionEvidenceAdmission, select_prompt_evidence

    evidence_by_id = {}
    supporting_ids = []
    for index in range(9):
        item = _fake_evidence(f"sup-{index:02d}", f"sp{index}", "comparison")
        evidence_by_id[item.id] = item
        supporting_ids.append(item.id)
    conflict = _fake_evidence("conflict-0", "cp0", "result")
    evidence_by_id[conflict.id] = conflict
    admission = QuestionEvidenceAdmission(
        "q", "PASS", [], supporting_ids + [conflict.id], list(supporting_ids),
        [], [conflict.id])

    selected, links = select_prompt_evidence(
        {"q": admission}, evidence_by_id, per_question=8, per_paper=2, total=40)

    selected_ids = {item.id for item in selected}
    assert conflict.id in selected_ids
    assert links[conflict.id] == [("q", "contradicts")]
    assert all(links[item.id] == [("q", "supports")]
               for item in selected if item.id != conflict.id)

"""Tests for lightweight evidence-backed gap mining."""

import os
import sys
import tempfile
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    config = Config()
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


class FakeGapLLM:
    async def chat_json(self, messages, schema):
        from app.schemas.schemas import GapCandidateList, GapCandidateSchema

        question_id = next(line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Question ID:" in line)
        evidence_ids = [line.split(": ", 1)[1] for line in messages[1]["content"].splitlines() if "Evidence ID:" in line]
        return GapCandidateList(gaps=[GapCandidateSchema(
            gap_type="boundary_gap",
            description="现有固定预算记忆方法在状态变化场景下缺少明确边界评测。",
            target_setting="固定 token 预算的 Agent Memory",
            observed_problem="状态变化发生后准确率下降，现有证据没有报告该边界。",
            existing_coverage="已有工作报告一般问答准确率和压缩结果。",
            missing_capability="对状态变化与冲突查询的边界评测。",
            claimed_delta="明确评估固定预算下状态变化证据保留能力。",
            testable_hypothesis="状态变化场景的性能低于稳定事实场景。",
            falsification_condition="近邻论文已在相同预算和场景下完成该评测。",
            question_ids=[question_id],
            supporting_evidence_ids=evidence_ids,
        )])


@pytest.mark.asyncio
async def test_mine_gap_creates_traceable_candidate(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract, ResearchQuestion, ResearchTask
    from app.db.repositories import gap_repo

    db = temp_db()
    task = ResearchTask(user_input="agent memory", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active", version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id,
        contract_id=contract.id,
        question="固定预算下状态变化会如何影响记忆系统？",
        question_type="failure",
        importance=0.9,
        searchability=0.8,
        status="partially_covered",
    )
    db.add(question)
    db.flush()
    first_paper = Paper(title="Evidence Paper A")
    second_paper = Paper(title="Evidence Paper B")
    db.add_all([first_paper, second_paper])
    db.flush()
    evidence = EvidenceUnit(
        task_id=task.id, paper_id=first_paper.id, evidence_type="limitation",
        normalized_claim="在状态变化后，固定预算记忆会遗漏关键历史证据。",
        original_span="状态变化后会遗漏关键证据", section="Limitations", page_number=8,
        span_start=10, span_end=22, source_chunk_hash="test-fulltext",
        verification_status="verified", extraction_confidence=0.8,
    )
    second_evidence = EvidenceUnit(
        task_id=task.id, paper_id=second_paper.id, evidence_type="comparison",
        normalized_claim="现有评测未覆盖状态变化边界。", verification_status="abstract_only",
    )
    db.add_all([evidence, second_evidence])
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=evidence.id, relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id, relation_type="supports", relevance_score=0.9),
    ])
    db.commit()
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)
    gaps = await mine_gap_candidates(db, state, FakeGapLLM(), task.id)

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.status == "candidate"
    assert gap.gap_type == "boundary_gap"
    assert gap.provenance_status == "complete"
    assert state.active_gap_ids == [gap.id]
    links = gap_repo.list_gap_evidence(db, gap.id)
    assert len(links) == 2
    assert {link.evidence_id for link in links} == {evidence.id, second_evidence.id}
    db.close()


@pytest.mark.asyncio
async def test_well_evidenced_covered_question_is_mined(temp_db):
    """A "covered" question is the best gap material, not a reason to skip it.

    Mining used to filter on status in (open, partially_covered), but
    update_coverage marks a question "covered" precisely when evidence has
    accumulated. On a real run that excluded 9 of 10 questions (191 evidence
    units) and mined only the weakest one at 0.15 coverage.
    """
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import (EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract,
                               ResearchQuestion, ResearchTask)

    db = temp_db()
    task = ResearchTask(user_input="agent memory", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Agent Memory", status="active",
                                version=1, input_hash="contract-v1")
    db.add(contract)
    db.flush()
    question = ResearchQuestion(
        task_id=task.id, contract_id=contract.id,
        question="固定预算下状态变化会如何影响记忆系统？",
        question_type="failure", importance=0.9, searchability=0.8,
        status="covered",
    )
    db.add(question)
    db.flush()
    first_paper = Paper(title="Evidence Paper A")
    second_paper = Paper(title="Evidence Paper B")
    db.add_all([first_paper, second_paper])
    db.flush()
    evidence = EvidenceUnit(
        task_id=task.id, paper_id=first_paper.id, evidence_type="limitation",
        normalized_claim="在状态变化后，固定预算记忆会遗漏关键历史证据。",
        original_span="状态变化后会遗漏关键证据", section="Limitations", page_number=8,
        span_start=10, span_end=22, source_chunk_hash="test-fulltext",
        verification_status="verified", extraction_confidence=0.8,
    )
    second_evidence = EvidenceUnit(
        task_id=task.id, paper_id=second_paper.id, evidence_type="comparison",
        normalized_claim="现有评测未覆盖状态变化边界。", verification_status="abstract_only",
    )
    db.add_all([evidence, second_evidence])
    db.flush()
    db.add_all([
        QuestionEvidenceLink(question_id=question.id, evidence_id=evidence.id,
                             relation_type="supports", relevance_score=0.9),
        QuestionEvidenceLink(question_id=question.id, evidence_id=second_evidence.id,
                             relation_type="supports", relevance_score=0.9),
    ])
    db.commit()
    state = ResearchState(task_id=task.id, contract_id=contract.id, current_round=2)

    gaps = await mine_gap_candidates(db, state, FakeGapLLM(), task.id)

    assert len(gaps) == 1, "a covered question with strong evidence must be mined"
    assert gaps[0].provenance_status == "complete"
    db.close()


@pytest.mark.asyncio
async def test_mine_gap_rejects_hallucinated_ids(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, Paper, QuestionEvidenceLink, ResearchContract, ResearchQuestion, ResearchTask
    from app.schemas.schemas import GapCandidateList, GapCandidateSchema

    class BadLLM:
        async def chat_json(self, messages, schema):
            return GapCandidateList(gaps=[GapCandidateSchema(
                gap_type="missing_evaluation",
                description="无效 ID 不得被写入缺口控制面。",
                target_setting="test",
                observed_problem="test",
                existing_coverage="test",
                missing_capability="test",
                claimed_delta="test",
                testable_hypothesis="test",
                falsification_condition="test",
                question_ids=["unknown-question"],
                supporting_evidence_ids=["unknown-evidence"],
            )])

    db = temp_db()
    task = ResearchTask(user_input="test", status="mining_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="test", status="active", version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    db.add(ResearchQuestion(task_id=task.id, contract_id=contract.id, question="test question", question_type="evaluation"))
    db.add(EvidenceUnit(task_id=task.id, paper_id="paper-1", evidence_type="metric", normalized_claim="test metric"))
    db.commit()

    gaps = await mine_gap_candidates(db, ResearchState(task_id=task.id, contract_id=contract.id), BadLLM(), task.id)
    assert gaps == []
    db.close()

@pytest.mark.asyncio
async def test_mine_gap_without_links_never_calls_llm(temp_db):
    from app.agent.state import ResearchState
    from app.agent.steps.mine_gaps import mine_gap_candidates
    from app.db.models import EvidenceUnit, Paper, ResearchContract, ResearchQuestion, ResearchTask

    class FailingIfCalledLLM:
        async def chat_json(self, messages, schema):
            raise AssertionError("LLM must not be called")

    db = temp_db()
    task = ResearchTask(user_input="test")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="test", status="active", version=1, input_hash="v1")
    paper = Paper(title="Unlinked Paper")
    db.add_all([contract, paper])
    db.flush()
    db.add_all([
        ResearchQuestion(task_id=task.id, contract_id=contract.id, question="question", question_type="failure", status="open"),
        EvidenceUnit(task_id=task.id, paper_id=paper.id, evidence_type="limitation", normalized_claim="unlinked evidence", verification_status="verified"),
    ])
    db.commit()
    result = await mine_gap_candidates(db, ResearchState(task_id=task.id, contract_id=contract.id), FailingIfCalledLLM(), task.id)
    assert result == []
    db.close()


def test_admission_rejects_single_paper_and_missing_span():
    from types import SimpleNamespace
    from app.agent.steps.mine_gaps import evaluate_gap_mining_admission

    question = SimpleNamespace(id="q", status="open")
    evidence = SimpleNamespace(id="e", paper_id="p", evidence_type="limitation", verification_status="abstract_only")
    link = SimpleNamespace(evidence_id="e", relation_type="supports", relevance_score=0.9)
    result = evaluate_gap_mining_admission(question, [link], {"e": evidence})
    assert result.status == "UNKNOWN"
    assert "INSUFFICIENT_INDEPENDENT_PAPERS" in result.reason_codes
    assert "NO_FULLTEXT_LOCATABLE_EVIDENCE" in result.reason_codes


def test_admission_blocks_verified_contradiction():
    from types import SimpleNamespace
    from app.agent.steps.mine_gaps import evaluate_gap_mining_admission

    question = SimpleNamespace(id="q", status="open")
    fulltext = lambda item_id, paper_id, kind: SimpleNamespace(
        id=item_id, paper_id=paper_id, evidence_type=kind, verification_status="verified",
        original_span="span", source_chunk_hash="hash", page_number=1, page_start=1,
        span_start=0, span_end=4,
    )
    evidence = {"a": fulltext("a", "p1", "limitation"), "b": fulltext("b", "p2", "comparison"), "c": fulltext("c", "p3", "comparison")}
    links = [
        SimpleNamespace(evidence_id="a", relation_type="supports", relevance_score=0.9),
        SimpleNamespace(evidence_id="b", relation_type="supports", relevance_score=0.9),
        SimpleNamespace(evidence_id="c", relation_type="contradicts", relevance_score=0.9),
    ]
    result = evaluate_gap_mining_admission(question, links, evidence)
    assert result.status == "UNKNOWN"
    assert "UNRESOLVED_VERIFIED_CONTRADICTION" in result.reason_codes


# --- Bounded evidence selection for the mining prompt -------------------------
#
# The admitted pool grows with every search round while the context window does
# not. A real run went from 9 admitted units in round 1 to 194 in round 4, whose
# prompt the backend rejected outright (40961 tokens against a 40960 window) and
# the whole task was failed.


@dataclass
class _FakeEvidence:
    id: str
    paper_id: str
    evidence_type: str = "comparison"
    verification_status: str = "verified"
    extraction_confidence: float = 0.8
    normalized_claim: str = "claim"
    original_span: str = "span"
    source_chunk_hash: str = "hash"
    page_number: int = 3
    page_start: int | None = None
    span_start: int = 10
    span_end: int = 40
    section: str = "results"
    conditions_json: str = "{}"


def _passing_admission(question_id: str, evidence_ids: list[str]):
    from app.agent.steps.mine_gaps import QuestionEvidenceAdmission

    return QuestionEvidenceAdmission(
        question_id, "PASS", [], list(evidence_ids), list(evidence_ids),
    )


def test_prompt_evidence_is_bounded_and_shared_across_questions():
    from app.agent.steps.mine_gaps import select_prompt_evidence

    evidence_by_id = {}
    admissions = {}
    for question_index in range(4):
        ids = []
        for evidence_index in range(30):
            item = _FakeEvidence(
                id=f"q{question_index}-e{evidence_index:02d}",
                paper_id=f"q{question_index}-p{evidence_index % 10}",
                evidence_type="limitation" if evidence_index == 0 else "comparison",
            )
            evidence_by_id[item.id] = item
            ids.append(item.id)
        admissions[f"q{question_index}"] = _passing_admission(f"q{question_index}", ids)

    selected, links = select_prompt_evidence(admissions, evidence_by_id,
                                             per_question=8, per_paper=2, total=20)

    assert len(selected) == 20
    # Every admitted question is represented, so a global cap truncates the long
    # tail instead of silently dropping whole questions from the prompt.
    assert {item.id.split("-")[0] for item in selected} == {"q0", "q1", "q2", "q3"}
    # A gap needs a documented shortcoming, so each question's limitation unit is
    # offered first.
    assert {f"q{index}-e00" for index in range(4)} <= {item.id for item in selected}
    assert all(links[item.id] for item in selected)


def test_per_paper_cap_keeps_independent_papers_in_view():
    from app.agent.steps.mine_gaps import select_prompt_evidence

    evidence_by_id = {}
    ids = []
    for index in range(10):
        item = _FakeEvidence(
            id=f"e{index:02d}",
            # Nine units from one paper and one from another: without a per-paper
            # cap the prompt could offer a single paper, and no candidate could
            # then satisfy the two-paper gate.
            paper_id="p1" if index < 9 else "p2",
            evidence_type="limitation" if index in (0, 9) else "comparison",
        )
        evidence_by_id[item.id] = item
        ids.append(item.id)

    selected, _ = select_prompt_evidence({"q": _passing_admission("q", ids)}, evidence_by_id,
                                         per_question=4, per_paper=2, total=10)

    assert len({item.paper_id for item in selected}) == 2
    assert len(selected) == 3


def test_prompt_evidence_selection_is_deterministic():
    from app.agent.steps.mine_gaps import select_prompt_evidence

    evidence_by_id = {}
    ids = []
    for index in range(12):
        item = _FakeEvidence(id=f"e{index:02d}", paper_id=f"p{index % 4}")
        evidence_by_id[item.id] = item
        ids.append(item.id)
    admissions = {"q1": _passing_admission("q1", ids[:6]),
                  "q2": _passing_admission("q2", ids[6:])}

    first = [item.id for item in select_prompt_evidence(admissions, evidence_by_id, total=5)[0]]
    second = [item.id for item in select_prompt_evidence(admissions, evidence_by_id, total=5)[0]]

    assert first == second

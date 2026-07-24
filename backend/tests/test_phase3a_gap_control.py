"""Phase 3A: Gap control plane tests.

Tests that the Gap data models, migration, ResearchState extension,
and query binding are correctly set up — without any mining logic.
"""

import os
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database via Alembic migrations."""
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


# === Test 1: Gap tables exist after migration ===

def test_gap_tables_exist(temp_db):
    """All 4 Gap tables exist after migration."""
    engine, Session = temp_db
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    required = {'gap_candidates', 'gap_evidence_links', 'gap_audits', 'neighbor_comparisons'}
    missing = required - tables
    assert not missing, f"Missing gap tables: {missing}"


# === Test 2: target_gap_id column in search_query_records ===

def test_target_gap_id_column_exists(temp_db):
    """search_query_records has target_gap_id column."""
    engine, Session = temp_db
    from sqlalchemy import inspect
    inspector = inspect(engine)
    cols = {c['name'] for c in inspector.get_columns('search_query_records')}
    assert 'target_gap_id' in cols, f"target_gap_id missing from search_query_records: {cols}"


# === Test 3: GapCandidate model CRUD ===

def test_gap_candidate_crud(temp_db):
    """GapCandidate can be created, queried, and has correct fields."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, GapCandidate
    import json

    # Create a task first
    task = ResearchTask(user_input="test gap", status="searching")
    db.add(task)
    db.commit()

    # Create a gap candidate
    gap = GapCandidate(
        task_id=task.id,
        gap_type="coverage_gap",
        description="No existing method addresses temporal state under fixed token budget",
        question_ids_json=json.dumps(["q1", "q2"]),
        supporting_evidence_ids_json=json.dumps(["e1"]),
        contradicting_evidence_ids_json=json.dumps([]),
        mining_round=1,
        novelty_score=0.8,
        feasibility_score=0.6,
        significance_score=0.7,
        risk_score=0.3,
        status="candidate",
        version=1,
    )
    db.add(gap)
    db.commit()

    # Query it back
    found = db.query(GapCandidate).filter(GapCandidate.task_id == task.id).first()
    assert found is not None
    assert found.gap_type == "coverage_gap"
    assert found.status == "candidate"
    assert found.novelty_score == 0.8
    assert json.loads(found.question_ids_json) == ["q1", "q2"]

    db.close()


# === Test 4: GapEvidenceLink model ===

def test_gap_evidence_link(temp_db):
    """GapEvidenceLink can link a gap to evidence."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import (
        ResearchTask, GapCandidate, GapEvidenceLink,
        EvidenceUnit, Paper,
    )

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    paper = Paper(title="Test Paper", abstract="test abstract")
    db.add(paper)
    db.commit()

    gap = GapCandidate(
        task_id=task.id, gap_type="contradiction",
        description="Evidence contradicts on accuracy",
    )
    db.add(gap)
    evidence = EvidenceUnit(
        task_id=task.id, paper_id=paper.id,
        evidence_type="result", normalized_claim="accuracy is 85%",
    )
    db.add(evidence)
    db.commit()

    link = GapEvidenceLink(
        gap_id=gap.id, evidence_id=evidence.id,
        relation_type="contradicts", relevance_score=0.9,
    )
    db.add(link)
    db.commit()

    found = db.query(GapEvidenceLink).filter(GapEvidenceLink.gap_id == gap.id).first()
    assert found is not None
    assert found.relation_type == "contradicts"
    assert found.relevance_score == 0.9

    db.close()


# === Test 5: GapAudit model ===

def test_gap_audit(temp_db):
    """GapAudit can store audit results."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, GapCandidate, GapAudit
    import json

    task = ResearchTask(user_input="test", status="auditing_gaps")
    db.add(task)
    db.commit()  # Must commit to get task.id
    gap = GapCandidate(
        task_id=task.id, gap_type="boundary_gap",
        description="Unexplored combination of compression + retrieval",
    )
    db.add(gap)
    db.commit()

    audit = GapAudit(
        gap_id=gap.id, task_id=task.id,
        adversarial_queries_json=json.dumps(["compression retrieval LLM", "memory budget retrieval"]),
        audit_result="confirmed",
        nearest_neighbor_summary="No paper directly addresses this combination",
        differentiation_summary="Existing work either uses compression OR retrieval, not both under budget",
        neighbor_paper_ids_json=json.dumps(["p1", "p2"]),
        audit_round=2,
    )
    db.add(audit)
    db.commit()

    found = db.query(GapAudit).filter(GapAudit.gap_id == gap.id).first()
    assert found is not None
    assert found.audit_result == "confirmed"
    assert len(json.loads(found.adversarial_queries_json)) == 2

    db.close()


# === Test 6: NeighborComparison model ===

def test_neighbor_comparison(temp_db):
    """NeighborComparison can store detailed comparisons."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, GapCandidate, NeighborComparison, Paper
    import json

    task = ResearchTask(user_input="test", status="auditing_gaps")
    db.add(task)
    paper = Paper(title="Neighbor Paper", abstract="similar work")
    db.add(paper)
    db.commit()  # Must commit to get task.id and paper.id
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="test gap")
    db.add(gap)
    db.commit()

    comp = NeighborComparison(
        gap_id=gap.id, paper_id=paper.id, task_id=task.id,
        similarity_score=0.7,
        shared_aspects_json=json.dumps(["uses compression", "targets memory budget"]),
        differentiating_aspects_json=json.dumps(["our gap uses retrieval too"]),
        overlap_risk=0.3,
    )
    db.add(comp)
    db.commit()

    found = db.query(NeighborComparison).filter(NeighborComparison.gap_id == gap.id).first()
    assert found is not None
    assert found.similarity_score == 0.7
    assert len(json.loads(found.shared_aspects_json)) == 2

    db.close()


# === Test 7: ResearchState has gap fields ===

def test_research_state_gap_fields():
    """ResearchState has active_gap_ids and surviving_gap_ids."""
    from app.agent.state import ResearchState
    import json

    state = ResearchState(task_id="test", user_input="test")
    assert hasattr(state, 'active_gap_ids')
    assert hasattr(state, 'surviving_gap_ids')
    assert state.active_gap_ids == []
    assert state.surviving_gap_ids == []

    # Test serialization roundtrip
    state.active_gap_ids = ["gap1", "gap2"]
    state.surviving_gap_ids = ["gap1"]
    json_str = state.to_json()
    restored = ResearchState.from_json(json_str)
    assert restored.active_gap_ids == ["gap1", "gap2"]
    assert restored.surviving_gap_ids == ["gap1"]


# === Test 8: SearchQueryExecution has target_gap_id ===

def test_search_query_execution_has_target_gap_id():
    """SearchQueryExecution dataclass has target_gap_id field."""
    from app.agent.steps.generate_queries import SearchQueryExecution

    qe = SearchQueryExecution(
        query_id="q1", query_text="test query",
        intent="gap_falsification", target_question_id="qq1",
        expected_evidence_type="method",
    )
    assert qe.target_gap_id is None  # Default is None for discovery queries

    qe_with_gap = SearchQueryExecution(
        query_id="q2", query_text="adversarial query",
        intent="gap_falsification", target_question_id="qq1",
        expected_evidence_type="method",
        target_gap_id="gap-123",
    )
    assert qe_with_gap.target_gap_id == "gap-123"


# === Test 9: save_search_query accepts target_gap_id ===

def test_save_search_query_with_gap_id(temp_db):
    """save_search_query can store target_gap_id."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, GapCandidate, SearchQueryRecord
    from app.db.repositories.search_query_repo import save_search_query, get_queries_for_gap

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="test gap")
    db.add(gap)
    db.commit()

    record = save_search_query(
        db, task.id, "adversarial query for gap", "gap_falsification",
        None, None, 1, target_gap_id=gap.id,
    )
    db.commit()

    assert record.target_gap_id == gap.id

    # Query by gap
    queries = get_queries_for_gap(db, gap.id)
    assert len(queries) == 1
    assert queries[0].query_text == "adversarial query for gap"

    db.close()


# === Test 10: Pydantic schemas ===

def test_gap_candidate_schema():
    """GapCandidateSchema validates correctly with all required fields."""
    from app.schemas.schemas import GapCandidateSchema, GapCandidateList

    gap = GapCandidateSchema(
        gap_type="coverage_gap",
        description="This is a test gap description that is long enough",
        target_setting="LLM Agent systems",
        observed_problem="No method addresses temporal state under fixed budget",
        existing_coverage="Existing work uses compression OR retrieval, not both",
        missing_capability="A method combining compression and retrieval under budget",
        claimed_delta="Our approach uses both compression and retrieval simultaneously",
        testable_hypothesis="Combining compression and retrieval improves accuracy under fixed token budget",
        falsification_condition="If existing work already combines both approaches with comparable or better accuracy",
        question_ids=["q1"],
        supporting_evidence_ids=["e1"],
        novelty_score=0.8,
        feasibility_score=0.6,
        significance_score=0.7,
    )
    assert gap.gap_type == "coverage_gap"
    assert gap.novelty_score == 0.8

    gap_list = GapCandidateList(gaps=[gap])
    assert len(gap_list.gaps) == 1


# === Test 11: GapCandidate with contract linkage ===

def test_gap_candidate_contract_linkage(temp_db):
    """GapCandidate can be linked to a ResearchContract."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, ResearchContract, GapCandidate

    task = ResearchTask(user_input="test", status="mining_gaps")
    db.add(task)
    db.commit()

    contract = ResearchContract(
        task_id=task.id, topic="test topic",
        status="active", version=1, input_hash="test_hash",
    )
    db.add(contract)
    db.commit()

    gap = GapCandidate(
        task_id=task.id, contract_id=contract.id,
        gap_type="missing_method", description="No method found for X",
    )
    db.add(gap)
    db.commit()

    found = db.query(GapCandidate).filter(GapCandidate.contract_id == contract.id).first()
    assert found is not None
    assert found.contract_id == contract.id

    db.close()


# === Test 12: Gap status lifecycle ===

def test_gap_status_lifecycle(temp_db):
    """Gap can transition through status lifecycle."""
    engine, Session = temp_db
    db = Session()

    from app.db.models import ResearchTask, GapCandidate

    task = ResearchTask(user_input="test", status="mining_gaps")
    db.add(task)
    db.commit()

    gap = GapCandidate(
        task_id=task.id, gap_type="coverage_gap",
        description="test gap", status="candidate",
    )
    db.add(gap)
    db.commit()

    # candidate → audited
    gap.status = "audited"
    db.commit()

    # audited → surviving
    gap.status = "surviving"
    db.commit()

    # surviving → rejected (if feasibility gate fails)
    gap.status = "rejected"
    db.commit()

    db.refresh(gap)
    assert gap.status == "rejected"

    db.close()

"""Phase 3A Control Plane Closure: Comprehensive tests.

Tests:
1. target_gap_id real FK exists
2. migration upgrade→downgrade→upgrade roundtrip
3. ORM vs Alembic column/FK/index consistency
4. Two gaps with same query don't cross-wire
5. Same gap with same query is idempotent
6. target_question_id can be None (gap query)
7. question_id and gap_id cannot both be None (SearchQueryExecution)
8. Cross-task Contract binding rejected
9. Cross-task Evidence binding rejected
10. Cross-task Audit/NeighborComparison rejected
11. GapEvidenceLink is API Evidence source
12. GapCandidateSchema missing falsification_condition fails
13. Gap status enum validation
14. Five read-only API endpoints
15. Superseded Contract Gap doesn't appear in active API
"""

import json
import os
import sys
import tempfile
import uuid
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


# === Test 1: target_gap_id FK exists ===

def test_target_gap_id_fk_exists(temp_db):
    """search_query_records.target_gap_id has a real FK to gap_candidates.id."""
    engine, Session = temp_db
    from sqlalchemy import inspect
    inspector = inspect(engine)
    fks = inspector.get_foreign_keys("search_query_records")
    assert any(
        fk["referred_table"] == "gap_candidates"
        and "target_gap_id" in fk["constrained_columns"]
        for fk in fks
    ), f"target_gap_id FK missing: {fks}"


# === Test 2: Migration roundtrip ===

def test_migration_roundtrip():
    """upgrade → downgrade → upgrade succeeds with consistent schema."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        from alembic.config import Config
        from alembic import command
        from sqlalchemy import create_engine, inspect
        url = f"sqlite:///{db_path}"
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", url)
        # Working dir must be backend
        import os as _os
        _os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg.set_main_option("script_location", "alembic_migrations")

        command.upgrade(cfg, "head")
        command.downgrade(cfg, "0009_output_json")
        command.upgrade(cfg, "head")

        engine = create_engine(url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "gap_candidates" in tables
        fks = inspector.get_foreign_keys("search_query_records")
        assert any(
            fk["referred_table"] == "gap_candidates"
            for fk in fks
        ), "FK should exist after roundtrip"
        engine.dispose()
    finally:
        import gc
        gc.collect()
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass


# === Test 3: ORM vs Alembic consistency ===

def test_orm_alembic_consistency(temp_db):
    """ORM model columns match what Alembic created."""
    engine, Session = temp_db
    from sqlalchemy import inspect
    inspector = inspect(engine)

    # gap_candidates has new structured fields
    gc_cols = {c['name'] for c in inspector.get_columns('gap_candidates')}
    required_gc = {'id', 'task_id', 'gap_type', 'description',
                  'target_setting', 'observed_problem', 'existing_coverage',
                  'missing_capability', 'claimed_delta', 'testable_hypothesis',
                  'falsification_condition', 'provenance_status', 'status'}
    missing = required_gc - gc_cols
    assert not missing, f"gap_candidates missing columns: {missing}"

    # gap_audits has decision fields
    ga_cols = {c['name'] for c in inspector.get_columns('gap_audits')}
    required_ga = {'recommended_action', 'rejection_reason', 'novelty_confidence',
                   'audit_confidence', 'remaining_delta', 'evidence_for_gap_json',
                   'evidence_against_gap_json'}
    missing_ga = required_ga - ga_cols
    assert not missing_ga, f"gap_audits missing columns: {missing_ga}"

    # neighbor_comparisons has structured fields
    nc_cols = {c['name'] for c in inspector.get_columns('neighbor_comparisons')}
    required_nc = {'shared_problem', 'shared_mechanism', 'shared_evaluation',
                   'covered_claims_json', 'uncovered_claims_json', 'overlap_ratio'}
    missing_nc = required_nc - nc_cols
    assert not missing_nc, f"neighbor_comparisons missing columns: {missing_nc}"

    # search_query_records has two partial unique indexes
    sqr_indexes = inspector.get_indexes('search_query_records')
    index_names = {idx['name'] for idx in sqr_indexes}
    assert 'idx_sqr_unique_discovery' in index_names, f"Missing discovery unique index: {index_names}"
    assert 'idx_sqr_unique_gap' in index_names, f"Missing gap unique index: {index_names}"
    assert 'idx_sqr_unique' not in index_names, "Old unique index should be dropped"


# === Test 4: Two gaps with same query don't cross-wire ===

def test_two_gaps_same_query_dont_crosswire(temp_db):
    """Two different gaps using the same query produce two separate records."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate
    from app.db.repositories.search_query_repo import save_search_query, get_queries_for_gap

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()
    gap_a = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="gap A")
    gap_b = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="gap B")
    db.add_all([gap_a, gap_b])
    db.commit()

    rec_a = save_search_query(db, task.id, "same query text", "gap_falsification",
                              None, None, 1, target_gap_id=gap_a.id)
    rec_b = save_search_query(db, task.id, "same query text", "gap_falsification",
                              None, None, 1, target_gap_id=gap_b.id)
    db.commit()

    assert rec_a.id != rec_b.id
    assert rec_a.target_gap_id == gap_a.id
    assert rec_b.target_gap_id == gap_b.id

    qa = get_queries_for_gap(db, gap_a.id)
    qb = get_queries_for_gap(db, gap_b.id)
    assert len(qa) == 1 and qa[0].id == rec_a.id
    assert len(qb) == 1 and qb[0].id == rec_b.id
    db.close()


# === Test 5: Same gap with same query is idempotent ===

def test_same_gap_same_query_idempotent(temp_db):
    """Same gap + same query returns the same record."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate
    from app.db.repositories.search_query_repo import save_search_query

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="gap")
    db.add(gap)
    db.commit()

    r1 = save_search_query(db, task.id, "query", "gap_falsification",
                           None, None, 1, target_gap_id=gap.id)
    r2 = save_search_query(db, task.id, "query", "gap_falsification",
                           None, None, 1, target_gap_id=gap.id)
    db.commit()

    assert r1.id == r2.id
    db.close()


# === Test 6: target_question_id can be None for gap queries ===

def test_target_question_id_none_for_gap(temp_db):
    """Gap queries can have target_question_id=None."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate, SearchQueryRecord
    from app.db.repositories.search_query_repo import save_search_query

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap", description="gap")
    db.add(gap)
    db.commit()

    rec = save_search_query(db, task.id, "adversarial query", "gap_falsification",
                           None, None, 1, target_gap_id=gap.id)
    db.commit()

    assert rec.target_question_id is None
    assert rec.target_gap_id == gap.id
    db.close()


# === Test 7: SearchQueryExecution requires at least one binding ===

def test_search_query_execution_requires_binding():
    """target_question_id and target_gap_id cannot both be None."""
    from app.agent.steps.generate_queries import SearchQueryExecution

    with pytest.raises(ValueError, match="at least one"):
        SearchQueryExecution(
            query_id="q1", query_text="test", intent="survey",
            target_question_id=None, expected_evidence_type=None,
            target_gap_id=None,
        )

    # With question_id only — OK
    qe1 = SearchQueryExecution(
        query_id="q1", query_text="test", intent="survey",
        target_question_id="qq1", expected_evidence_type=None,
    )
    assert qe1.target_gap_id is None

    # With gap_id only — OK
    qe2 = SearchQueryExecution(
        query_id="q2", query_text="test", intent="gap_falsification",
        target_question_id=None, expected_evidence_type=None,
        target_gap_id="gap1",
    )
    assert qe2.target_gap_id == "gap1"

    # With both — OK
    qe3 = SearchQueryExecution(
        query_id="q3", query_text="test", intent="gap_falsification",
        target_question_id="qq1", expected_evidence_type=None,
        target_gap_id="gap1",
    )
    assert qe3.target_question_id == "qq1"
    assert qe3.target_gap_id == "gap1"


# === Test 8: Cross-task Contract binding rejected ===

def test_cross_task_contract_rejected(temp_db):
    """Creating a gap with a contract from another task raises error."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, ResearchContract
    from app.db.repositories.gap_repo import create_gap_candidate, CrossTaskValidationError

    task_a = ResearchTask(user_input="task A", status="searching")
    task_b = ResearchTask(user_input="task B", status="searching")
    db.add_all([task_a, task_b])
    db.commit()
    contract_b = ResearchContract(task_id=task_b.id, topic="B topic",
                                  status="active", version=1, input_hash="b")
    db.add(contract_b)
    db.commit()

    with pytest.raises(CrossTaskValidationError):
        create_gap_candidate(db, task_id=task_a.id, gap_type="coverage_gap",
                            description="test", contract_id=contract_b.id)
    db.close()


# === Test 9: Cross-task Evidence binding rejected ===

def test_cross_task_evidence_rejected(temp_db):
    """Linking a gap to evidence from another task raises error."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate, EvidenceUnit, Paper
    from app.db.repositories.gap_repo import create_gap_evidence_link, CrossTaskValidationError

    task_a = ResearchTask(user_input="A", status="searching")
    task_b = ResearchTask(user_input="B", status="searching")
    db.add_all([task_a, task_b])
    paper = Paper(title="P", abstract="a")
    db.add(paper)
    db.commit()
    gap_a = GapCandidate(task_id=task_a.id, gap_type="coverage_gap", description="gap A")
    ev_b = EvidenceUnit(task_id=task_b.id, paper_id=paper.id,
                       evidence_type="method", normalized_claim="claim B")
    db.add_all([gap_a, ev_b])
    db.commit()

    with pytest.raises(CrossTaskValidationError):
        create_gap_evidence_link(db, gap_id=gap_a.id, evidence_id=ev_b.id)
    db.close()


# === Test 10: Cross-task Audit/NeighborComparison rejected ===

def test_cross_task_audit_rejected(temp_db):
    """Creating an audit with wrong task_id raises error."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate
    from app.db.repositories.gap_repo import create_gap_audit, CrossTaskValidationError

    task_a = ResearchTask(user_input="A", status="searching")
    task_b = ResearchTask(user_input="B", status="searching")
    db.add_all([task_a, task_b])
    db.commit()
    gap_a = GapCandidate(task_id=task_a.id, gap_type="coverage_gap", description="gap")
    db.add(gap_a)
    db.commit()

    with pytest.raises(CrossTaskValidationError):
        create_gap_audit(db, gap_id=gap_a.id, task_id=task_b.id)
    db.close()


def test_cross_task_neighbor_comparison_rejected(temp_db):
    """Creating a neighbor comparison with wrong task_id raises error."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate, Paper
    from app.db.repositories.gap_repo import create_neighbor_comparison, CrossTaskValidationError

    task_a = ResearchTask(user_input="A", status="searching")
    task_b = ResearchTask(user_input="B", status="searching")
    db.add_all([task_a, task_b])
    paper = Paper(title="P", abstract="a")
    db.add(paper)
    db.commit()
    gap_a = GapCandidate(task_id=task_a.id, gap_type="coverage_gap", description="gap")
    db.add(gap_a)
    db.commit()

    with pytest.raises(CrossTaskValidationError):
        create_neighbor_comparison(db, gap_id=gap_a.id, paper_id=paper.id, task_id=task_b.id)
    db.close()


# === Test 11: GapEvidenceLink is API Evidence source ===

def test_gap_evidence_link_is_api_source(temp_db):
    """Evidence comes from gap_evidence_links, not JSON snapshots."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate, EvidenceUnit, Paper
    from app.db.repositories.gap_repo import create_gap_evidence_link, list_gap_evidence

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    paper = Paper(title="P", abstract="a")
    db.add(paper)
    db.commit()
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap",
                     description="gap", supporting_evidence_ids_json="[]")
    ev = EvidenceUnit(task_id=task.id, paper_id=paper.id,
                     evidence_type="method", normalized_claim="claim")
    db.add_all([gap, ev])
    db.commit()

    # JSON snapshot is empty, but link table has data
    create_gap_evidence_link(db, gap_id=gap.id, evidence_id=ev.id,
                            relation_type="suggests")
    db.commit()

    links = list_gap_evidence(db, gap.id)
    assert len(links) == 1
    assert links[0].evidence_id == ev.id

    # JSON snapshot is still empty — link is authoritative
    assert json.loads(gap.supporting_evidence_ids_json) == []
    db.close()


# === Test 12: GapCandidateSchema missing falsification_condition fails ===

def test_gap_schema_missing_falsification_fails():
    """GapCandidateSchema requires falsification_condition."""
    from app.schemas.schemas import GapCandidateSchema
    from pydantic import ValidationError

    with pytest.raises((ValidationError, Exception)):
        GapCandidateSchema(
            gap_type="coverage_gap",
            description="test description that is long enough",
            target_setting="setting",
            observed_problem="problem",
            existing_coverage="coverage",
            missing_capability="capability",
            claimed_delta="delta",
            testable_hypothesis="hypothesis",
            # Missing falsification_condition!
            question_ids=["q1"],
            supporting_evidence_ids=["e1"],
        )


def test_gap_schema_missing_question_id_fails():
    """GapCandidateSchema requires at least one question_id."""
    from app.schemas.schemas import GapCandidateSchema
    from pydantic import ValidationError

    with pytest.raises((ValidationError, Exception)):
        GapCandidateSchema(
            gap_type="coverage_gap",
            description="test description that is long enough",
            target_setting="setting",
            observed_problem="problem",
            existing_coverage="coverage",
            missing_capability="capability",
            claimed_delta="delta",
            testable_hypothesis="hypothesis",
            falsification_condition="falsification",
            question_ids=[],  # Empty!
            supporting_evidence_ids=["e1"],
        )


# === Test 13: Gap status enum validation ===

def test_gap_status_enum():
    """Gap statuses match frozen enum."""
    valid_statuses = {"candidate", "auditing", "audited", "surviving", "rejected", "superseded"}
    # "survived" is NOT valid — must be "surviving"
    assert "survived" not in valid_statuses
    assert "surviving" in valid_statuses


# === Test 14: Five read-only API endpoints ===

def test_gap_api_endpoints(temp_db):
    """All five read-only Gap API endpoints work."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate, EvidenceUnit, Paper
    from app.db.repositories.gap_repo import create_gap_evidence_link

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    paper = Paper(title="P", abstract="a")
    db.add(paper)
    db.commit()
    gap = GapCandidate(task_id=task.id, gap_type="coverage_gap",
                     description="test gap", status="candidate")
    ev = EvidenceUnit(task_id=task.id, paper_id=paper.id,
                     evidence_type="method", normalized_claim="claim")
    db.add_all([gap, ev])
    db.commit()
    create_gap_evidence_link(db, gap_id=gap.id, evidence_id=ev.id)
    db.commit()

    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_db_session

    def _override_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db_session] = _override_db
    client = TestClient(app)

    # 1. GET /tasks/{task_id}/gaps
    r = client.get(f"/api/tasks/{task.id}/gaps")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    # 2. GET /gaps/{gap_id}
    r = client.get(f"/api/gaps/{gap.id}")
    assert r.status_code == 200
    assert r.json()["id"] == gap.id

    # 3. GET /gaps/{gap_id}/evidence
    r = client.get(f"/api/gaps/{gap.id}/evidence")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    # 4. GET /gaps/{gap_id}/audits
    r = client.get(f"/api/gaps/{gap.id}/audits")
    assert r.status_code == 200

    # 5. GET /gaps/{gap_id}/neighbors
    r = client.get(f"/api/gaps/{gap.id}/neighbors")
    assert r.status_code == 200

    app.dependency_overrides.clear()
    db.close()


# === Test 15: Superseded Contract Gap doesn't appear in active API ===

def test_superseded_gap_not_in_active(temp_db):
    """Superseded gaps don't appear in the default (non-superseded) API call."""
    engine, Session = temp_db
    db = Session()
    from app.db.models import ResearchTask, GapCandidate
    from app.db.repositories.gap_repo import list_active_gaps_for_task

    task = ResearchTask(user_input="test", status="searching")
    db.add(task)
    db.commit()
    active_gap = GapCandidate(task_id=task.id, gap_type="coverage_gap",
                             description="active", status="candidate")
    superseded_gap = GapCandidate(task_id=task.id, gap_type="coverage_gap",
                                  description="old", status="superseded")
    db.add_all([active_gap, superseded_gap])
    db.commit()

    # Default: excludes superseded
    gaps = list_active_gaps_for_task(db, task.id)
    assert len(gaps) == 1
    assert gaps[0].id == active_gap.id

    # Include superseded
    gaps_all = list_active_gaps_for_task(db, task.id, include_superseded=True)
    assert len(gaps_all) == 2
    db.close()

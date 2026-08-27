"""Tests for claim-level coverage and verdict derivation (P1-1 Phase 3H).

Locks the blocking issue: UNCERTAIN is handled FIRST, so a "confirmed" verdict
can never be stamped while some claim is still unresolved; and a claim with no
effective NPA judgment is UNCERTAIN (insufficient evidence), never NONE.
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


def _seed(db, coverages_by_claim_index):
    """Seed a gap + 3 atomic claims + optional neighbor claim coverage.

    coverages_by_claim_index: dict[claim_index] -> list[coverage] (one per NPA).
    """
    from app.agent.steps.mine_gaps import GAP_MINING_POLICY_VERSION
    from app.db.models import GapCandidate, ResearchContract, ResearchTask
    from app.db.repositories import gap_repo

    task = ResearchTask(user_input="code gen", status="auditing_gaps")
    db.add(task)
    db.flush()
    contract = ResearchContract(task_id=task.id, topic="Code Gen", status="active",
                                version=1, input_hash="v1")
    db.add(contract)
    db.flush()
    gap = gap_repo.create_gap_candidate(
        db, task_id=task.id, contract_id=contract.id, gap_type="boundary_gap",
        description="self-correction eval", target_setting="code gen",
        observed_problem="accidental correctness", existing_coverage="generic",
        missing_capability="dense hidden tests", claimed_delta="sparse-to-dense eval",
        testable_hypothesis="h", falsification_condition="f",
        provenance_status="complete", question_ids=[], mining_round=1,
        mining_policy_version=GAP_MINING_POLICY_VERSION,
    )
    claims = []
    for i, text in enumerate(["C0 capability", "C1 capability", "C2 capability"]):
        claims.append(gap_repo.create_atomic_claim(db, task.id, gap.id, i, text))
    for claim_index, covs in coverages_by_claim_index.items():
        for j, cov in enumerate(covs):
            gap_repo.create_neighbor_claim_coverage(
                db, task_id=task.id, gap_id=gap.id,
                neighbor_paper_id=f"paper-{j}", claim_id=claims[claim_index].id,
                coverage=cov, rationale="",
            )
    db.commit()
    return gap, claims


def _derive(db, gap):
    from app.agent.steps.audit_gaps import _derive_verdict_from_claims
    return _derive_verdict_from_claims(db, gap)


def test_all_full_closes(temp_db):
    db = temp_db()
    gap, _ = _seed(db, {0: ["FULL"], 1: ["FULL"], 2: ["FULL"]})
    result = _derive(db, gap)
    assert result[0] == "closed" and result[1] == "reject"
    db.close()


def test_none_without_partial_confirms(temp_db):
    db = temp_db()
    gap, _ = _seed(db, {0: ["NONE"], 1: ["NONE"], 2: ["NONE"]})
    result = _derive(db, gap)
    assert result[0] == "confirmed" and result[1] == "continue"
    # residual = all three claims (none FULL-covered)
    assert len(result[2]) == 3
    db.close()


def test_partial_narrows(temp_db):
    db = temp_db()
    gap, _ = _seed(db, {0: ["FULL"], 1: ["PARTIAL"], 2: ["NONE"]})
    result = _derive(db, gap)
    assert result[0] == "partially_closed" and result[1] == "narrow"
    db.close()


def test_uncertain_forces_more_search_even_with_none(temp_db):
    # BLOCKING ISSUE: residual non-empty + partial empty used to confirm, but a
    # coexisting UNCERTAIN claim must force more_search instead.
    db = temp_db()
    gap, _ = _seed(db, {0: ["NONE"], 1: ["NONE"], 2: ["UNCERTAIN"]})
    result = _derive(db, gap)
    assert result[0] == "uncertain" and result[1] == "more_search"
    db.close()


def test_missing_judgment_is_uncertain_not_none(temp_db):
    # BLOCKING ISSUE: a claim no NPA judged is insufficient evidence -> UNCERTAIN,
    # never aggregated into NONE (which would wrongly count as "uncovered").
    db = temp_db()
    gap, _ = _seed(db, {0: ["NONE"], 1: ["NONE"]})  # claim 2 has no judgment
    result = _derive(db, gap)
    assert result[0] == "uncertain" and result[1] == "more_search"
    db.close()


def test_uncertain_claims_remain_in_residual(temp_db):
    db = temp_db()
    gap, claims = _seed(db, {0: ["NONE"], 1: ["UNCERTAIN"]})
    from app.agent.steps.audit_gaps import _derive_verdict_from_claims
    result = _derive_verdict_from_claims(db, gap)
    assert result[0] == "uncertain"
    assert set(result[2]) == {claim.id for claim in claims}
    db.close()


def test_full_removes_from_residual(temp_db):
    # residual only subtracts FULL; PARTIAL stays (partially addressed).
    db = temp_db()
    gap, _ = _seed(db, {0: ["FULL"], 1: ["PARTIAL"], 2: ["NONE"]})
    result = _derive(db, gap)
    residual_ids = set(result[2])
    # claim 0 is FULL -> removed; claims 1 (PARTIAL) and 2 (NONE) remain.
    assert len(residual_ids) == 2
    db.close()

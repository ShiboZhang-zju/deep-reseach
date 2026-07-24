import os
import sys
import tempfile

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_lineage_columns_migrate():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        config = Config()
        config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
        config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
        command.upgrade(config, "head")
        engine = create_engine(f"sqlite:///{path}")
        inspector = inspect(engine)
        assert {"contract_id", "gap_id", "intervention_id", "pipeline_version"}.issubset({column["name"] for column in inspector.get_columns("research_ideas")})
        engine.dispose()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_intervention_recovery_is_scoped_to_contract():
    from app.db.models import Base, GapCandidate, InterventionCandidate, ResearchContract, ResearchTask
    from app.db.repositories.intervention_repo import list_interventions_for_task
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    task = ResearchTask(user_input="test")
    db.add(task)
    db.flush()
    old_contract = ResearchContract(task_id=task.id, topic="old", status="superseded", version=1, input_hash="old")
    active_contract = ResearchContract(task_id=task.id, topic="new", status="active", version=2, input_hash="new")
    db.add_all([old_contract, active_contract])
    db.flush()
    old_gap = GapCandidate(task_id=task.id, contract_id=old_contract.id, gap_type="boundary_gap", description="old")
    new_gap = GapCandidate(task_id=task.id, contract_id=active_contract.id, gap_type="boundary_gap", description="new")
    db.add_all([old_gap, new_gap])
    db.flush()
    old_item = InterventionCandidate(task_id=task.id, gap_id=old_gap.id, contract_id=old_contract.id, intervention_type="evaluation", failure_mechanism="old", proposed_intervention="old", intermediate_effect="old", measurable_outcome="old", status="passed")
    new_item = InterventionCandidate(task_id=task.id, gap_id=new_gap.id, contract_id=active_contract.id, intervention_type="evaluation", failure_mechanism="new", proposed_intervention="new", intermediate_effect="new", measurable_outcome="new", status="passed")
    db.add_all([old_item, new_item])
    db.commit()
    recovered = list_interventions_for_task(db, task.id, active_contract.id, [new_gap.id])
    assert [item.id for item in recovered] == [new_item.id]
    db.close()
    engine.dispose()

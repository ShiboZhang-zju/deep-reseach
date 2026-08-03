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
    config.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "alembic_migrations"))
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{path}")
    yield sessionmaker(bind=engine)
    engine.dispose()
    try: os.unlink(path)
    except PermissionError: pass

from app.agent.steps.audit_gaps import GAP_SEARCH_POLICY_VERSION, audit_gap_candidates, evaluate_gap_search_admission, select_gap_specific_neighbors
from app.agent.state import ResearchState
from app.db.models import Paper, SearchQueryPaper, SearchQueryRecord
from tests.test_audit_gaps import _seed_gap

class FailingIfCalledLLM:
    async def chat_json(self, messages, schema):
        raise AssertionError("Audit LLM must not be called")

@pytest.mark.asyncio
async def test_search_admission_blocks_llm_without_gap_papers(temp_db):
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    state = ResearchState(task_id=task.id, contract_id=gap.contract_id, current_round=2)
    results = await audit_gap_candidates(db, state, FailingIfCalledLLM(), task.id, perform_search=False)
    assert results[0].audit_result == "uncertain"

def test_gap_specific_neighbor_selection_excludes_task_only_paper(temp_db):
    db = temp_db()
    task, gap, _ = _seed_gap(db)
    queried = [Paper(title=f"Queried {index}", abstract="q") for index in range(3)]
    db.add_all(queried)
    db.flush()
    records = []
    for family in ("exact_gap", "alternative_coverage", "claim_falsification"):
        record = SearchQueryRecord(task_id=task.id, query_text=family, normalized_query_text=family, intent=f"gap_{family}", round_number=3, status="completed", target_gap_id=gap.id, query_family=family, search_policy_version=GAP_SEARCH_POLICY_VERSION)
        db.add(record)
        records.append(record)
    db.flush()
    for index, record in enumerate(records):
        for paper in queried:
            db.add(SearchQueryPaper(query_id=record.id, paper_id=paper.id, rank=index, source="test", is_new_for_task=True))
    db.commit()
    admission = evaluate_gap_search_admission(db, gap, [record.id for record in records])
    assert admission.status == "PASS"
    assert {paper.id for paper in select_gap_specific_neighbors(db, gap, admission.completed_query_ids)} == {paper.id for paper in queried}
    db.close()
"""Legacy database migration test — verifies data preservation.

Creates a pre-Phase-0 schema with data, runs bootstrap, verifies preservation.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_legacy_db_migration_preserves_data():
    """Legacy database with old schema can be migrated and data preserved."""
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
    from sqlalchemy import create_engine, text, inspect

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{db_path}"

    try:
        # Step 1: Create legacy schema (just base tables, no alembic_version)
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Create minimal old tables
            conn.execute(text("""
                CREATE TABLE research_tasks (
                    id TEXT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    normalized_topic TEXT,
                    status TEXT DEFAULT 'pending',
                    current_round INTEGER DEFAULT 0,
                    max_rounds INTEGER DEFAULT 5,
                    stop_reason TEXT,
                    state_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    citation_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE task_papers (
                    id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES research_tasks(id),
                    paper_id TEXT REFERENCES papers(id),
                    discovered_round INTEGER NOT NULL,
                    priority TEXT,
                    final_score REAL
                )
            """))

            # Insert test data
            conn.execute(text("""
                INSERT INTO research_tasks (id, user_input, status)
                VALUES ('legacy-task-1', 'legacy research input', 'done')
            """))
            conn.execute(text("""
                INSERT INTO papers (id, title, year)
                VALUES ('legacy-paper-1', 'Legacy Paper Title', 2023)
            """))
            conn.execute(text("""
                INSERT INTO task_papers (id, task_id, paper_id, discovered_round, priority, final_score)
                VALUES ('legacy-tp-1', 'legacy-task-1', 'legacy-paper-1', 1, 'high', 0.85)
            """))
            conn.commit()

        # Verify data exists
        with engine.connect() as conn:
            task_count = conn.execute(text("SELECT COUNT(*) FROM research_tasks")).fetchone()[0]
            paper_count = conn.execute(text("SELECT COUNT(*) FROM papers")).fetchone()[0]
            tp_count = conn.execute(text("SELECT COUNT(*) FROM task_papers")).fetchone()[0]

        assert task_count == 1
        assert paper_count == 1
        assert tp_count == 1

        engine.dispose()

        # Step 2: Run bootstrap
        # Add scripts to path
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        sys.path.insert(0, scripts_dir)
        from bootstrap_db import bootstrap
        success = bootstrap(db_url)
        assert success, "Bootstrap should succeed"

        # Step 3: Verify data preserved
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Original data preserved
            task_count_after = conn.execute(text("SELECT COUNT(*) FROM research_tasks")).fetchone()[0]
            paper_count_after = conn.execute(text("SELECT COUNT(*) FROM papers")).fetchone()[0]
            tp_count_after = conn.execute(text("SELECT COUNT(*) FROM task_papers")).fetchone()[0]

            assert task_count_after == 1, f"Tasks should be preserved: {task_count_after}"
            assert paper_count_after == 1, f"Papers should be preserved: {paper_count_after}"
            assert tp_count_after == 1, f"TaskPapers should be preserved: {tp_count_after}"

            # Verify actual data content
            task = conn.execute(text("SELECT user_input FROM research_tasks WHERE id='legacy-task-1'")).fetchone()
            assert task is not None
            assert task[0] == 'legacy research input'

        # Step 4: Verify new tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        required_new = [
            'research_contracts', 'research_questions', 'evidence_units',
            'coverage_records', 'search_query_records', 'phase_runs',
        ]
        for t in required_new:
            assert t in tables, f"New table {t} should exist after migration"

        # Verify alembic_version exists
        assert 'alembic_version' in tables, "alembic_version table should exist"

        engine.dispose()
        print("Legacy migration test passed — all data preserved, new tables created")

    finally:
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

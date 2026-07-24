"""Database bootstrap script — handles fresh and legacy databases.

Usage:
    python scripts/bootstrap_db.py [--database-url sqlite:///path/to/db]

For fresh databases: Runs alembic upgrade head from scratch
For legacy databases: Detects, stamps, upgrades, verifies
"""

import os
import sys
import shutil
from datetime import datetime
from sqlalchemy import create_engine, inspect, text

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_schema_revision(engine) -> str | None:
    """Detect the approximate schema revision based on existing tables and columns."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if not tables:
        return None  # Fresh database

    if 'alembic_version' in tables:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
            return result[0] if result else None

    has_phase_runs = 'phase_runs' in tables
    has_contracts = 'research_contracts' in tables
    has_evidence = 'evidence_units' in tables
    has_search_queries = 'search_query_records' in tables

    if has_contracts:
        contract_cols = {c['name'] for c in inspector.get_columns('research_contracts')}
        has_versioning = 'version' in contract_cols and 'input_hash' in contract_cols
    else:
        has_versioning = False

    if has_evidence:
        eu_cols = {c['name'] for c in inspector.get_columns('evidence_units')}
        has_span = 'span_start' in eu_cols
    else:
        has_span = False

    if 'coverage_records' in tables:
        cr_cols = {c['name'] for c in inspector.get_columns('coverage_records')}
        has_round = 'round_number' in cr_cols
    else:
        has_round = False

    if has_search_queries:
        sqr_cols = {c['name'] for c in inspector.get_columns('search_query_records')}
        has_normalized = 'normalized_query_text' in sqr_cols
    else:
        has_normalized = False

    # Check for base tables (pre-Phase-1 schema)
    has_base_tables = {'research_tasks', 'papers', 'task_papers'}.issubset(tables)

    if has_normalized:
        return '0007_query_norm'
    elif has_search_queries and has_span and has_round:
        return '0006_searchquery_ev2'
    elif has_evidence:
        return '0005_evidence_coverage'
    elif has_versioning:
        return '0004_contract_versioning'
    elif has_contracts:
        return '0003_contract_questions'
    elif has_phase_runs:
        return '0002_phase_runs'
    elif has_base_tables:
        # Phase 2.2A Final Closure: Verify minimum column manifest before
        # stamping 0001_baseline. Don't trust table names alone — verify
        # that critical columns exist so ORM reads won't crash.
        required_manifest = {
            'research_tasks': {'id', 'user_input', 'status'},
            'papers': {'id', 'title'},
            'task_papers': {'id', 'task_id', 'paper_id', 'discovered_round'},
        }
        manifest_ok = True
        for tbl, required_cols in required_manifest.items():
            if tbl not in tables:
                manifest_ok = False
                break
            actual_cols = {c['name'] for c in inspector.get_columns(tbl)}
            missing = required_cols - actual_cols
            if missing:
                print(f"  Manifest check failed: {tbl} missing columns: {missing}")
                manifest_ok = False
                break

        if manifest_ok:
            return '0001_baseline'
        else:
            # Unknown schema — base tables exist but columns don't match
            return None
    else:
        # (#11) Unknown schema — cannot safely identify
        return None


def count_rows(engine, table_name: str) -> int:
    """Count rows in a table."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
            return result[0] if result else 0
    except Exception:
        return 0


def _add_missing_columns(database_url: str):
    """Add missing columns to existing legacy tables.

    When a legacy database has tables created by Base.metadata.create_all()
    or manual SQL, but some columns are missing (because they were added in
    later migrations), this function adds them via ALTER TABLE.
    """
    engine = create_engine(database_url)
    inspector = inspect(engine)

    # Define expected columns per table (matching current ORM models)
    expected_columns = {
        'research_tasks': [
            ('stop_reason', 'TEXT'),
        ],
        'papers': [
            ('authors_json', 'TEXT'),
            ('arxiv_id', 'TEXT'),
            ('semantic_scholar_id', 'TEXT'),
            ('openalex_id', 'TEXT'),
            ('url', 'TEXT'),
            ('pdf_url', 'TEXT'),
            ('sources_json', 'TEXT'),
            ('raw_json', 'TEXT'),
            ('normalized_title', 'TEXT'),
            ('title_hash', 'TEXT'),
        ],
        'task_papers': [
            ('relevance_score', 'REAL'),
            ('authority_score', 'REAL'),
            ('recency_score', 'REAL'),
            ('novelty_score', 'REAL'),
            ('idea_potential_score', 'REAL'),
            ('reason', 'TEXT'),
            ('summary', 'TEXT'),
            ('created_at', 'DATETIME'),
        ],
        'research_rounds': [
            ('queries_json', 'TEXT'),
            ('duplicate_rate', 'REAL'),
            ('knowledge_gaps_json', 'TEXT'),
        ],
        'research_ideas': [
            ('motivation', 'TEXT'),
            ('method_sketch', 'TEXT'),
            ('expected_contribution', 'TEXT'),
            ('novelty', 'REAL'),
            ('feasibility', 'REAL'),
            ('significance', 'REAL'),
            ('evidence_support', 'REAL'),
            ('differentiation', 'REAL'),
            ('experimentability', 'REAL'),
            ('potential_impact', 'REAL'),
            ('risk', 'REAL'),
            ('related_paper_ids_json', 'TEXT'),
            ('user_selected', 'BOOLEAN DEFAULT 0'),
            ('idea_status', "TEXT DEFAULT 'active'"),
        ],
    }

    with engine.connect() as conn:
        for table_name, columns in expected_columns.items():
            if table_name not in inspector.get_table_names():
                continue

            existing_cols = {c['name'] for c in inspector.get_columns(table_name)}
            for col_name, col_type in columns:
                if col_name not in existing_cols:
                    print(f"  Adding column {table_name}.{col_name} ({col_type})")
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    ))
                    existing_cols.add(col_name)
        conn.commit()

    engine.dispose()


def backup_database(db_path: str) -> str:
    """Create a timestamped backup of the database file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def bootstrap(database_url: str = None) -> bool:
    """Bootstrap the database — fresh or legacy."""
    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command

    if not database_url:
        from app.config import settings
        database_url = settings.database_url

    engine = create_engine(database_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    has_alembic_version = 'alembic_version' in tables

    print(f"Database URL: {database_url}")
    print(f"Tables found: {len(tables)}")
    print(f"Has alembic_version: {has_alembic_version}")

    data_before = {}
    for t in ['research_tasks', 'papers', 'task_papers', 'reports', 'research_ideas']:
        if t in tables:
            data_before[t] = count_rows(engine, t)
            print(f"  {t}: {data_before[t]} rows")

    engine.dispose()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = AlembicConfig()
    cfg.set_main_option('sqlalchemy.url', database_url)
    cfg.set_main_option('script_location', os.path.join(backend_dir, 'alembic_migrations'))

    if not tables:
        print("\n=== Fresh database ===")
        print("Running alembic upgrade head...")
        alembic_command.upgrade(cfg, 'head')
        print("Fresh database created successfully.")

    elif not has_alembic_version:
        print("\n=== Legacy database detected ===")
        if database_url.startswith('sqlite:///'):
            db_path = database_url.replace('sqlite:///', '')
            if os.path.exists(db_path):
                backup = backup_database(db_path)
                print(f"Backup created: {backup}")

        engine = create_engine(database_url)
        detected = detect_schema_revision(engine)
        engine.dispose()

        if detected:
            print(f"Detected schema revision: {detected}")
            print(f"Stamping to {detected}...")
            alembic_command.stamp(cfg, detected)
        else:
            # (#11) Cannot safely identify schema — must fail, not default to baseline
            print("ERROR: Cannot identify schema revision from existing tables.")
            print("Tables exist but don't match any known schema pattern.")
            print("Refusing to stamp baseline — this could corrupt data.")
            print("Manual intervention required: inspect the database and determine")
            print("the correct migration revision to stamp.")
            return False

        # Add missing columns to existing tables before upgrade
        # (Alembic migrations assume tables match the revision they were stamped at,
        # but legacy DBs may have partial columns)
        print("Checking for missing columns in existing tables...")
        _add_missing_columns(database_url)

        print("Running alembic upgrade head...")
        alembic_command.upgrade(cfg, 'head')
        print("Legacy database upgraded successfully.")

    else:
        print("\n=== Database with alembic_version ===")
        print("Running alembic upgrade head...")
        alembic_command.upgrade(cfg, 'head')
        print("Database upgraded to head.")

    engine = create_engine(database_url)
    print("\n=== Verification ===")
    all_ok = True
    for t, count_before in data_before.items():
        count_after = count_rows(engine, t)
        status = "OK" if count_after == count_before else "MISMATCH"
        if count_after != count_before:
            all_ok = False
        print(f"  {t}: {count_before} -> {count_after} [{status}]")

    inspector = inspect(engine)
    required_new = ['research_contracts', 'research_questions', 'evidence_units',
                    'coverage_records', 'search_query_records', 'phase_runs']
    for t in required_new:
        exists = t in inspector.get_table_names()
        print(f"  {t}: {'exists' if exists else 'MISSING'}")
        if not exists:
            all_ok = False

    engine.dispose()

    if all_ok:
        print("\n✓ Bootstrap completed successfully.")
    else:
        print("\n✗ Bootstrap completed with issues.")

    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-url', default=None)
    args = parser.parse_args()
    success = bootstrap(args.database_url)
    sys.exit(0 if success else 1)

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
    else:
        return '0001_baseline'


def count_rows(engine, table_name: str) -> int:
    """Count rows in a table."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
            return result[0] if result else 0
    except Exception:
        return 0


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
            print("Could not detect revision, stamping to 0001_baseline")
            alembic_command.stamp(cfg, '0001_baseline')

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

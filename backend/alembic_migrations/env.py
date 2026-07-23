# Alembic env.py — configured to use Deep Research models (P2-10)
"""Alembic environment configuration.

Reads models from app.db.models so autogenerate works.
Database URL is loaded from app.config.settings (which reads .env).
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Add backend/ to path so we can import app.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.db.session import Base  # noqa: E402
import app.db.models  # noqa: E402, F401  # register all models with Base.metadata

# this is the Alembic Config object
config = context.config

# Use the sqlalchemy.url from alembic.ini by default.
# If the test/migration sets a custom URL via cfg.set_main_option, respect it.
# Only fall back to settings.database_url if the URL is still the default from alembic.ini.
current_url = config.get_main_option("sqlalchemy.url")
if current_url and "deep_research.db" not in current_url:
    # Custom URL set (e.g., test temp DB) — respect it
    pass
else:
    # Use the URL from .env settings
    config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite-friendly: use ALTER TABLE via batch mode
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode (connect to DB and execute)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite-friendly
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

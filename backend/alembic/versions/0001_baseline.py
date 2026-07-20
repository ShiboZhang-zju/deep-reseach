"""baseline — mark existing schema (P2-10)

This is a baseline migration. It does NOT create tables (they already exist via
Base.metadata.create_all in main.py). Future schema changes should use
`alembic revision --autogenerate -m "description"`.

Revision ID: 0001_baseline
Revises: 
Create Date: 2026-07-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Baseline migration — no-op. Existing schema is already in place."""
    # To generate a real migration for future changes, run:
    #   alembic revision --autogenerate -m "add new column"
    #
    # This baseline assumes the database already has all tables from
    # Base.metadata.create_all(). If starting fresh, run:
    #   alembic stamp head   (marks current DB as up-to-date without running SQL)
    pass


def downgrade() -> None:
    """No-op baseline downgrade."""
    pass

"""Phase 2.2A Final Closure: Add output_json column to phase_runs

Stores the complete JSON payload of phase outputs so that
RoundSearchResult (and other phase results) can be restored
exactly without approximate reconstruction from secondary tables.

Revision ID: 0009_output_json
Revises: 0008_sqp
Create Date: 2026-07-24
"""
from alembic import op
import sqlalchemy as sa


revision = '0009_output_json'
down_revision = '0008_sqp'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'phase_runs',
        sa.Column('output_json', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('phase_runs', 'output_json')

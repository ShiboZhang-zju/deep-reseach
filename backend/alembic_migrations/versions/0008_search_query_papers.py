"""Phase 2.2A Closure: Add search_query_papers table

Revision ID: 0008_sqp
Revises: 0007_query_norm
Create Date: 2026-07-23
"""
from alembic import op
import sqlalchemy as sa


revision = '0008_sqp'
down_revision = '0007_query_norm'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'search_query_papers',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('query_id', sa.String(), sa.ForeignKey('search_query_records.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('rank', sa.Integer(), server_default='0'),
        sa.Column('source', sa.Text(), server_default='unknown'),
        sa.Column('is_new_for_task', sa.Boolean(), server_default='1'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_sqp_query', 'search_query_papers', ['query_id'])
    op.create_index('idx_sqp_paper', 'search_query_papers', ['paper_id'])
    op.create_index('idx_sqp_unique', 'search_query_papers',
                    ['query_id', 'paper_id', 'source'], unique=True)


def downgrade() -> None:
    op.drop_table('search_query_papers')

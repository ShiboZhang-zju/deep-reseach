"""Phase 2.2A: Add normalized_query_text, completed_at, unique constraint to search_query_records

Revision ID: 0007_query_norm
Revises: 0006_searchquery_ev2
Create Date: 2026-07-23
"""
from alembic import op
import sqlalchemy as sa


revision = '0007_query_norm'
down_revision = '0006_searchquery_ev2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('search_query_records') as batch_op:
        batch_op.add_column(sa.Column('normalized_query_text', sa.Text(), server_default=''))
        batch_op.add_column(sa.Column('completed_at', sa.DateTime()))

    op.create_index(
        'idx_sqr_unique',
        'search_query_records',
        ['task_id', 'round_number', 'normalized_query_text', 'target_question_id'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('idx_sqr_unique', table_name='search_query_records')

    with op.batch_alter_table('search_query_records') as batch_op:
        batch_op.drop_column('completed_at')
        batch_op.drop_column('normalized_query_text')

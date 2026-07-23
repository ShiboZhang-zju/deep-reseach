"""Phase 1.5: Add versioning fields to research_contracts and research_questions

Revision ID: 0004_contract_versioning
Revises: 0003_contract_questions
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa


revision = '0004_contract_versioning'
down_revision = '0003_contract_questions'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add versioning fields to research_contracts
    with op.batch_alter_table('research_contracts') as batch_op:
        batch_op.add_column(sa.Column('version', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('input_hash', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('superseded_at', sa.DateTime()))
        batch_op.add_column(sa.Column('source_feedback_ids_json', sa.Text(), server_default='[]'))

    # Add versioning fields to research_questions
    with op.batch_alter_table('research_questions') as batch_op:
        batch_op.add_column(sa.Column('version', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('superseded_at', sa.DateTime()))

    op.create_index('idx_rq_contract', 'research_questions', ['contract_id'])


def downgrade() -> None:
    op.drop_index('idx_rq_contract', table_name='research_questions')

    with op.batch_alter_table('research_questions') as batch_op:
        batch_op.drop_column('superseded_at')
        batch_op.drop_column('version')

    with op.batch_alter_table('research_contracts') as batch_op:
        batch_op.drop_column('source_feedback_ids_json')
        batch_op.drop_column('superseded_at')
        batch_op.drop_column('input_hash')
        batch_op.drop_column('version')

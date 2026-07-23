"""Phase 2.1: Add SearchQueryRecord, update EvidenceUnit and CoverageRecord

Revision ID: 0006_searchquery_ev2
Revises: 0005_evidence_coverage
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa


revision = '0006_searchquery_ev2'
down_revision = '0005_evidence_coverage'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to evidence_units
    with op.batch_alter_table('evidence_units') as batch_op:
        batch_op.add_column(sa.Column('page_start', sa.Integer()))
        batch_op.add_column(sa.Column('page_end', sa.Integer()))
        batch_op.add_column(sa.Column('span_start', sa.Integer()))
        batch_op.add_column(sa.Column('span_end', sa.Integer()))
        batch_op.add_column(sa.Column('source_chunk_hash', sa.Text()))
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()))

    op.create_index('idx_eu_chunk_hash', 'evidence_units', ['source_chunk_hash'])
    op.create_index('idx_eu_task_paper_hash', 'evidence_units', ['task_id', 'paper_id', 'source_chunk_hash'])

    # Add round_number to coverage_records
    with op.batch_alter_table('coverage_records') as batch_op:
        batch_op.add_column(sa.Column('round_number', sa.Integer(), nullable=False, server_default='0'))

    # Drop old unique index and create new one with round_number
    try:
        op.drop_index('idx_cr_task_question', table_name='coverage_records')
    except Exception:
        pass
    op.create_index('idx_cr_task_question_round', 'coverage_records',
                    ['task_id', 'question_id', 'round_number'])

    # Create search_query_records table
    op.create_table(
        'search_query_records',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('intent', sa.Text(), nullable=False),
        sa.Column('target_question_id', sa.String(), sa.ForeignKey('research_questions.id')),
        sa.Column('expected_evidence_type', sa.Text()),
        sa.Column('round_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending'),
        sa.Column('result_count', sa.Integer(), server_default='0'),
        sa.Column('new_paper_count', sa.Integer(), server_default='0'),
        sa.Column('evidence_unit_count', sa.Integer(), server_default='0'),
        sa.Column('execution_error', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_sqr_task', 'search_query_records', ['task_id'])
    op.create_index('idx_sqr_intent', 'search_query_records', ['intent'])
    op.create_index('idx_sqr_round', 'search_query_records', ['round_number'])
    op.create_index('idx_sqr_question', 'search_query_records', ['target_question_id'])


def downgrade() -> None:
    op.drop_table('search_query_records')
    try:
        op.drop_index('idx_cr_task_question_round', table_name='coverage_records')
    except Exception:
        pass
    op.create_index('idx_cr_task_question', 'coverage_records', ['task_id', 'question_id'], unique=True)

    with op.batch_alter_table('coverage_records') as batch_op:
        batch_op.drop_column('round_number')

    op.drop_index('idx_eu_task_paper_hash', table_name='evidence_units')
    op.drop_index('idx_eu_chunk_hash', table_name='evidence_units')

    with op.batch_alter_table('evidence_units') as batch_op:
        batch_op.drop_column('updated_at')
        batch_op.drop_column('source_chunk_hash')
        batch_op.drop_column('span_end')
        batch_op.drop_column('span_start')
        batch_op.drop_column('page_end')
        batch_op.drop_column('page_start')

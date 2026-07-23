"""Phase 2: Add evidence_units, question_evidence_links, paper_roles, coverage_records

Revision ID: 0005_evidence_coverage
Revises: 0004_contract_versioning
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa


revision = '0005_evidence_coverage'
down_revision = '0004_contract_versioning'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # evidence_units
    op.create_table(
        'evidence_units',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('evidence_type', sa.Text(), nullable=False),
        sa.Column('normalized_claim', sa.Text(), nullable=False),
        sa.Column('original_span', sa.Text()),
        sa.Column('section', sa.Text()),
        sa.Column('page_number', sa.Integer()),
        sa.Column('conditions_json', sa.Text(), server_default='{}'),
        sa.Column('dataset_name', sa.Text()),
        sa.Column('metric_name', sa.Text()),
        sa.Column('result_value', sa.Text()),
        sa.Column('extraction_method', sa.Text(), server_default='llm'),
        sa.Column('extraction_confidence', sa.Float(), server_default='0.5'),
        sa.Column('verification_status', sa.Text(), server_default='unverified'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_eu_task', 'evidence_units', ['task_id'])
    op.create_index('idx_eu_paper', 'evidence_units', ['paper_id'])
    op.create_index('idx_eu_type', 'evidence_units', ['evidence_type'])
    op.create_index('idx_eu_verification', 'evidence_units', ['verification_status'])

    # question_evidence_links
    op.create_table(
        'question_evidence_links',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('question_id', sa.String(), sa.ForeignKey('research_questions.id'), nullable=False),
        sa.Column('evidence_id', sa.String(), sa.ForeignKey('evidence_units.id'), nullable=False),
        sa.Column('relation_type', sa.Text(), server_default='supports'),
        sa.Column('relevance_score', sa.Float(), server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_qel_question', 'question_evidence_links', ['question_id'])
    op.create_index('idx_qel_evidence', 'question_evidence_links', ['evidence_id'])

    # paper_roles
    op.create_table(
        'paper_roles',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='0.5'),
        sa.Column('reason', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_pr_task', 'paper_roles', ['task_id'])
    op.create_index('idx_pr_paper', 'paper_roles', ['paper_id'])
    op.create_index('idx_pr_role', 'paper_roles', ['role'])

    # coverage_records
    op.create_table(
        'coverage_records',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('question_id', sa.String(), sa.ForeignKey('research_questions.id'), nullable=False),
        sa.Column('coverage_score', sa.Float(), server_default='0.0'),
        sa.Column('confidence', sa.Float(), server_default='0.0'),
        sa.Column('supporting_evidence_count', sa.Integer(), server_default='0'),
        sa.Column('contradicting_evidence_count', sa.Integer(), server_default='0'),
        sa.Column('direct_neighbor_count', sa.Integer(), server_default='0'),
        sa.Column('unresolved_aspects_json', sa.Text(), server_default='[]'),
        sa.Column('unavailable_reason', sa.Text()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_cr_task', 'coverage_records', ['task_id'])
    op.create_index('idx_cr_question', 'coverage_records', ['question_id'])
    op.create_index('idx_cr_task_question', 'coverage_records', ['task_id', 'question_id'], unique=True)


def downgrade() -> None:
    op.drop_table('coverage_records')
    op.drop_table('paper_roles')
    op.drop_table('question_evidence_links')
    op.drop_table('evidence_units')

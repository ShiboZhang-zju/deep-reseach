"""Phase 3A: Gap control plane — gap_candidates, gap_evidence_links, gap_audits, neighbor_comparisons

Revision ID: 0010_gap_tables
Revises: 0009_output_json
Create Date: 2026-07-24
"""
from alembic import op
import sqlalchemy as sa


revision = '0010_gap_tables'
down_revision = '0009_output_json'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # gap_candidates (must exist before search_query_records FK)
    op.create_table(
        'gap_candidates',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('contract_id', sa.String(), sa.ForeignKey('research_contracts.id')),
        sa.Column('gap_type', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('question_ids_json', sa.Text(), server_default='[]'),
        sa.Column('supporting_evidence_ids_json', sa.Text(), server_default='[]'),
        sa.Column('contradicting_evidence_ids_json', sa.Text(), server_default='[]'),
        sa.Column('mining_round', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('novelty_score', sa.Float()),
        sa.Column('feasibility_score', sa.Float()),
        sa.Column('significance_score', sa.Float()),
        sa.Column('risk_score', sa.Float()),
        sa.Column('status', sa.String(), nullable=False, server_default='candidate'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('superseded_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_gc_task', 'gap_candidates', ['task_id'])
    op.create_index('idx_gc_contract', 'gap_candidates', ['contract_id'])
    op.create_index('idx_gc_status', 'gap_candidates', ['status'])
    op.create_index('idx_gc_task_status', 'gap_candidates', ['task_id', 'status'])

    # gap_evidence_links
    op.create_table(
        'gap_evidence_links',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('gap_id', sa.String(), sa.ForeignKey('gap_candidates.id'), nullable=False),
        sa.Column('evidence_id', sa.String(), sa.ForeignKey('evidence_units.id'), nullable=False),
        sa.Column('relation_type', sa.Text(), server_default='suggests'),
        sa.Column('relevance_score', sa.Float(), server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_gel_gap', 'gap_evidence_links', ['gap_id'])
    op.create_index('idx_gel_evidence', 'gap_evidence_links', ['evidence_id'])
    op.create_index('idx_gel_unique', 'gap_evidence_links', ['gap_id', 'evidence_id'], unique=True)

    # gap_audits
    op.create_table(
        'gap_audits',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('gap_id', sa.String(), sa.ForeignKey('gap_candidates.id'), nullable=False),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('adversarial_queries_json', sa.Text(), server_default='[]'),
        sa.Column('audit_result', sa.String(), nullable=False, server_default='pending'),
        sa.Column('nearest_neighbor_summary', sa.Text()),
        sa.Column('differentiation_summary', sa.Text()),
        sa.Column('neighbor_paper_ids_json', sa.Text(), server_default='[]'),
        sa.Column('audit_round', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_ga_gap', 'gap_audits', ['gap_id'])
    op.create_index('idx_ga_task', 'gap_audits', ['task_id'])
    op.create_index('idx_ga_result', 'gap_audits', ['audit_result'])

    # neighbor_comparisons
    op.create_table(
        'neighbor_comparisons',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('gap_id', sa.String(), sa.ForeignKey('gap_candidates.id'), nullable=False),
        sa.Column('paper_id', sa.String(), sa.ForeignKey('papers.id'), nullable=False),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('similarity_score', sa.Float(), server_default='0.0'),
        sa.Column('shared_aspects_json', sa.Text(), server_default='[]'),
        sa.Column('differentiating_aspects_json', sa.Text(), server_default='[]'),
        sa.Column('overlap_risk', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_nc_gap', 'neighbor_comparisons', ['gap_id'])
    op.create_index('idx_nc_paper', 'neighbor_comparisons', ['paper_id'])
    op.create_index('idx_nc_unique', 'neighbor_comparisons', ['gap_id', 'paper_id'], unique=True)

    # Phase 3A: Add target_gap_id to search_query_records (gap-driven query binding)
    # Use batch mode for SQLite (ALTER TABLE with FK requires copy-and-move)
    with op.batch_alter_table('search_query_records') as batch_op:
        batch_op.add_column(sa.Column('target_gap_id', sa.String(), nullable=True))
    op.create_index('idx_sqr_gap', 'search_query_records', ['target_gap_id'])


def downgrade() -> None:
    op.drop_table('neighbor_comparisons')
    op.drop_table('gap_audits')
    op.drop_table('gap_evidence_links')
    op.drop_table('gap_candidates')

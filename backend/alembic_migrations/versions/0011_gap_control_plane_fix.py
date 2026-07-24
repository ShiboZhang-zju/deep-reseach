"""Phase 3A Closure: FK fix, structured gap fields, query unique constraints

Revision ID: 0011_gap_closure
Revises: 0010_gap_tables
Create Date: 2026-07-24

Changes:
- Add real FK on search_query_records.target_gap_id → gap_candidates.id
- Drop old unique index idx_sqr_unique; replace with two partial unique indexes
- Add structured fields to gap_candidates (falsifiable contract)
- Add decision fields to gap_audits
- Add structured comparison fields to neighbor_comparisons
"""
from alembic import op
import sqlalchemy as sa


revision = '0011_gap_closure'
down_revision = '0010_gap_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add real FK on search_query_records.target_gap_id
    with op.batch_alter_table('search_query_records') as batch_op:
        batch_op.create_foreign_key(
            'fk_sqr_target_gap',
            'gap_candidates',
            ['target_gap_id'],
            ['id'],
        )

    # 2. Drop old unique index, create two partial unique indexes
    op.drop_index('idx_sqr_unique', table_name='search_query_records')

    # Discovery queries: target_gap_id IS NULL
    op.create_index(
        'idx_sqr_unique_discovery',
        'search_query_records',
        ['task_id', 'round_number', 'normalized_query_text', 'target_question_id'],
        unique=True,
        sqlite_where=sa.text('target_gap_id IS NULL'),
    )

    # Gap queries: target_gap_id IS NOT NULL
    op.create_index(
        'idx_sqr_unique_gap',
        'search_query_records',
        ['task_id', 'round_number', 'normalized_query_text', 'target_gap_id'],
        unique=True,
        sqlite_where=sa.text('target_gap_id IS NOT NULL'),
    )

    # 3. Add structured fields to gap_candidates
    with op.batch_alter_table('gap_candidates') as batch_op:
        batch_op.add_column(sa.Column('target_setting', sa.Text()))
        batch_op.add_column(sa.Column('observed_problem', sa.Text()))
        batch_op.add_column(sa.Column('existing_coverage', sa.Text()))
        batch_op.add_column(sa.Column('missing_capability', sa.Text()))
        batch_op.add_column(sa.Column('claimed_delta', sa.Text()))
        batch_op.add_column(sa.Column('testable_hypothesis', sa.Text()))
        batch_op.add_column(sa.Column('falsification_condition', sa.Text()))
        batch_op.add_column(sa.Column('provenance_status', sa.Text(), server_default='partial'))

    # 4. Add decision fields to gap_audits
    with op.batch_alter_table('gap_audits') as batch_op:
        batch_op.add_column(sa.Column('evidence_for_gap_json', sa.Text(), server_default='[]'))
        batch_op.add_column(sa.Column('evidence_against_gap_json', sa.Text(), server_default='[]'))
        batch_op.add_column(sa.Column('remaining_delta', sa.Text()))
        batch_op.add_column(sa.Column('novelty_confidence', sa.Float()))
        batch_op.add_column(sa.Column('audit_confidence', sa.Float()))
        batch_op.add_column(sa.Column('recommended_action', sa.Text(), server_default='continue'))
        batch_op.add_column(sa.Column('rejection_reason', sa.Text()))

    # 5. Add structured comparison fields to neighbor_comparisons
    with op.batch_alter_table('neighbor_comparisons') as batch_op:
        batch_op.add_column(sa.Column('shared_problem', sa.Text()))
        batch_op.add_column(sa.Column('shared_mechanism', sa.Text()))
        batch_op.add_column(sa.Column('shared_evaluation', sa.Text()))
        batch_op.add_column(sa.Column('covered_claims_json', sa.Text(), server_default='[]'))
        batch_op.add_column(sa.Column('uncovered_claims_json', sa.Text(), server_default='[]'))
        batch_op.add_column(sa.Column('overlap_ratio', sa.Float(), server_default='0.0'))


def downgrade() -> None:
    # Reverse neighbor_comparisons
    with op.batch_alter_table('neighbor_comparisons') as batch_op:
        batch_op.drop_column('overlap_ratio')
        batch_op.drop_column('uncovered_claims_json')
        batch_op.drop_column('covered_claims_json')
        batch_op.drop_column('shared_evaluation')
        batch_op.drop_column('shared_mechanism')
        batch_op.drop_column('shared_problem')

    # Reverse gap_audits
    with op.batch_alter_table('gap_audits') as batch_op:
        batch_op.drop_column('rejection_reason')
        batch_op.drop_column('recommended_action')
        batch_op.drop_column('audit_confidence')
        batch_op.drop_column('novelty_confidence')
        batch_op.drop_column('remaining_delta')
        batch_op.drop_column('evidence_against_gap_json')
        batch_op.drop_column('evidence_for_gap_json')

    # Reverse gap_candidates
    with op.batch_alter_table('gap_candidates') as batch_op:
        batch_op.drop_column('provenance_status')
        batch_op.drop_column('falsification_condition')
        batch_op.drop_column('testable_hypothesis')
        batch_op.drop_column('claimed_delta')
        batch_op.drop_column('missing_capability')
        batch_op.drop_column('existing_coverage')
        batch_op.drop_column('observed_problem')
        batch_op.drop_column('target_setting')

    # Reverse unique indexes
    op.drop_index('idx_sqr_unique_gap', table_name='search_query_records')
    op.drop_index('idx_sqr_unique_discovery', table_name='search_query_records')

    # Restore old unique index
    op.create_index(
        'idx_sqr_unique', 'search_query_records',
        ['task_id', 'round_number', 'normalized_query_text', 'target_question_id'],
        unique=True,
    )

    # Drop FK and target_gap_id column.
    # NOTE: 0010's downgrade doesn't drop target_gap_id (a bug in 0010).
    # We cannot modify 0010 (already pushed), so we drop the column here
    # as a supplementary fix to make the full downgrade path work.
    with op.batch_alter_table('search_query_records') as batch_op:
        batch_op.drop_constraint('fk_sqr_target_gap', type_='foreignkey')
        batch_op.drop_index('idx_sqr_gap')
        batch_op.drop_column('target_gap_id')

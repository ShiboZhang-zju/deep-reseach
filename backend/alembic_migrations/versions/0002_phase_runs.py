"""Phase 0: Add phase_runs table

Revision ID: 0002_phase_runs
Revises: 0001_baseline
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0002_phase_runs'
down_revision = '0001_baseline'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'phase_runs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('phase_name', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('attempt_count', sa.Integer(), server_default='0'),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('input_version', sa.String()),
        sa.Column('output_version', sa.String()),
        sa.Column('error_message', sa.Text()),
        sa.Column('round_number', sa.Integer()),
        sa.Column('output_summary', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_phase_runs_task', 'phase_runs', ['task_id'])
    op.create_index('idx_phase_runs_task_phase', 'phase_runs', ['task_id', 'phase_name'])
    op.create_index('idx_phase_runs_status', 'phase_runs', ['status'])


def downgrade() -> None:
    op.drop_index('idx_phase_runs_status', table_name='phase_runs')
    op.drop_index('idx_phase_runs_task_phase', table_name='phase_runs')
    op.drop_index('idx_phase_runs_task', table_name='phase_runs')
    op.drop_table('phase_runs')

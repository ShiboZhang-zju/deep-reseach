"""Phase 1: Add research_contracts and research_questions tables

Revision ID: 0003_contract_questions
Revises: 0002_phase_runs
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0003_contract_questions'
down_revision = '0002_phase_runs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # research_contracts
    op.create_table(
        'research_contracts',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('topic', sa.Text(), nullable=False),
        sa.Column('target_problem', sa.Text()),
        sa.Column('target_setting', sa.Text()),
        sa.Column('desired_output', sa.Text()),
        sa.Column('novelty_bar', sa.Text(), server_default='conference'),
        sa.Column('preferred_directions_json', sa.Text(), server_default='[]'),
        sa.Column('excluded_directions_json', sa.Text(), server_default='[]'),
        sa.Column('gpu_available', sa.Boolean()),
        sa.Column('max_gpu_hours', sa.Float()),
        sa.Column('max_api_budget', sa.Float()),
        sa.Column('max_runtime_minutes', sa.Integer()),
        sa.Column('allow_large_benchmark', sa.Boolean(), server_default='1'),
        sa.Column('allow_model_training', sa.Boolean(), server_default='1'),
        sa.Column('experiment_preferences_json', sa.Text(), server_default='{}'),
        sa.Column('key_terms_json', sa.Text(), server_default='[]'),
        sa.Column('time_scope_start', sa.Integer()),
        sa.Column('time_scope_end', sa.Integer()),
        sa.Column('status', sa.Text(), server_default='active'),
        sa.Column('confidence', sa.Float(), server_default='0.5'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_contracts_task', 'research_contracts', ['task_id'])

    # research_questions
    op.create_table(
        'research_questions',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), sa.ForeignKey('research_tasks.id'), nullable=False),
        sa.Column('contract_id', sa.String(), sa.ForeignKey('research_contracts.id')),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('question_type', sa.Text(), nullable=False),
        sa.Column('importance', sa.Float(), server_default='0.5'),
        sa.Column('searchability', sa.Float(), server_default='0.5'),
        sa.Column('status', sa.Text(), server_default='open'),
        sa.Column('axis_name', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp()),
    )
    op.create_index('idx_rq_task', 'research_questions', ['task_id'])
    op.create_index('idx_rq_status', 'research_questions', ['status'])
    op.create_index('idx_rq_task_type', 'research_questions', ['task_id', 'question_type'])


def downgrade() -> None:
    op.drop_index('idx_rq_task_type', table_name='research_questions')
    op.drop_index('idx_rq_status', table_name='research_questions')
    op.drop_index('idx_rq_task', table_name='research_questions')
    op.drop_table('research_questions')
    op.drop_index('idx_contracts_task', table_name='research_contracts')
    op.drop_table('research_contracts')

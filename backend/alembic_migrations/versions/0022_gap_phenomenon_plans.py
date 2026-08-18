"""Add gap phenomenon validation plans.

Before any intervention or method is designed, a surviving gap's underlying
empirical claim must be pinned down: what phenomenon it rests on, the cheapest
experiment that could falsify it, and the kill criterion below which the
phenomenon is too small to be worth a method paper. This gates method generation
on "the problem is real and measurable", not "the story is plausible".
"""

from alembic import op
import sqlalchemy as sa


revision = "0022_gap_phenomenon_plans"
down_revision = "0021_gap_nearest_prior_art"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("gap_phenomenon_plans"):
        return
    op.create_table(
        "gap_phenomenon_plans",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("contract_id", sa.String(), nullable=True),
        sa.Column("gap_id", sa.String(), nullable=False),
        sa.Column("phenomenon", sa.Text(), nullable=False),
        sa.Column("critical_unknown", sa.Text(), nullable=True),
        sa.Column("oracle_experiment", sa.Text(), nullable=True),
        sa.Column("kill_criterion", sa.Text(), nullable=True),
        sa.Column("measurement", sa.Text(), nullable=True),
        sa.Column("pipeline_version", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_gpp_task", "gap_phenomenon_plans", ["task_id"])
    op.create_index("idx_gpp_gap", "gap_phenomenon_plans", ["gap_id"])


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("gap_phenomenon_plans"):
        op.drop_table("gap_phenomenon_plans")

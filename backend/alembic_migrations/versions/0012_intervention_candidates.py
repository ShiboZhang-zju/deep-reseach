"""Add lightweight intervention candidates.

Revision ID: 0012_interventions
Revises: 0011_gap_closure
Create Date: 2026-07-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0012_interventions"
down_revision = "0011_gap_closure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intervention_candidates",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("research_tasks.id"), nullable=False),
        sa.Column("gap_id", sa.String(), sa.ForeignKey("gap_candidates.id"), nullable=False),
        sa.Column("intervention_type", sa.Text(), nullable=False),
        sa.Column("failure_mechanism", sa.Text(), nullable=False),
        sa.Column("proposed_intervention", sa.Text(), nullable=False),
        sa.Column("intermediate_effect", sa.Text(), nullable=False),
        sa.Column("measurable_outcome", sa.Text(), nullable=False),
        sa.Column("required_components_json", sa.Text(), server_default="[]"),
        sa.Column("dependency_paper_ids_json", sa.Text(), server_default="[]"),
        sa.Column("implementation_cost", sa.Text()),
        sa.Column("mechanism_confidence", sa.Float()),
        sa.Column("evidence_gate", sa.Text(), server_default="UNKNOWN"),
        sa.Column("novelty_gate", sa.Text(), server_default="UNKNOWN"),
        sa.Column("feasibility_gate", sa.Text(), server_default="UNKNOWN"),
        sa.Column("gate_rationale_json", sa.Text(), server_default="{}"),
        sa.Column("status", sa.Text(), server_default="candidate"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("idx_ic_task", "intervention_candidates", ["task_id"])
    op.create_index("idx_ic_gap", "intervention_candidates", ["gap_id"])
    op.create_index("idx_ic_status", "intervention_candidates", ["status"])


def downgrade() -> None:
    op.drop_index("idx_ic_status", table_name="intervention_candidates")
    op.drop_index("idx_ic_gap", table_name="intervention_candidates")
    op.drop_index("idx_ic_task", table_name="intervention_candidates")
    op.drop_table("intervention_candidates")

"""Add contract and intervention lineage for opportunity recovery.

Revision ID: 0013_opportunity_lineage
Revises: 0012_interventions
Create Date: 2026-07-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0013_opportunity_lineage"
down_revision = "0012_interventions"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if _has_table("intervention_candidates"):
        with op.batch_alter_table("intervention_candidates") as batch_op:
            batch_op.add_column(sa.Column("contract_id", sa.String()))
        op.create_index("idx_ic_contract", "intervention_candidates", ["contract_id"])

    if _has_table("research_ideas"):
        with op.batch_alter_table("research_ideas") as batch_op:
            batch_op.add_column(sa.Column("contract_id", sa.String()))
            batch_op.add_column(sa.Column("gap_id", sa.String()))
            batch_op.add_column(sa.Column("intervention_id", sa.String()))
            batch_op.add_column(sa.Column("pipeline_version", sa.Integer()))
        op.create_index("idx_ideas_contract", "research_ideas", ["contract_id"])
        op.create_index("idx_ideas_intervention", "research_ideas", ["intervention_id"])


def downgrade() -> None:
    if _has_table("research_ideas"):
        op.drop_index("idx_ideas_intervention", table_name="research_ideas")
        op.drop_index("idx_ideas_contract", table_name="research_ideas")
        with op.batch_alter_table("research_ideas") as batch_op:
            batch_op.drop_column("pipeline_version")
            batch_op.drop_column("intervention_id")
            batch_op.drop_column("gap_id")
            batch_op.drop_column("contract_id")
    if _has_table("intervention_candidates"):
        op.drop_index("idx_ic_contract", table_name="intervention_candidates")
        with op.batch_alter_table("intervention_candidates") as batch_op:
            batch_op.drop_column("contract_id")

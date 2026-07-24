"""Add gap mining policy version."""

from alembic import op
import sqlalchemy as sa

revision = "0014_gap_mining_policy"
down_revision = "0013_opportunity_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("gap_candidates"):
        with op.batch_alter_table("gap_candidates") as batch_op:
            batch_op.add_column(sa.Column("mining_policy_version", sa.Text(), server_default=""))


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("gap_candidates"):
        with op.batch_alter_table("gap_candidates") as batch_op:
            batch_op.drop_column("mining_policy_version")

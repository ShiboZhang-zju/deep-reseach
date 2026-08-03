"""Add gap search admission lineage."""

from alembic import op
import sqlalchemy as sa

revision = "0015_gap_search_admission"
down_revision = "0014_gap_mining_policy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("search_query_records") as batch_op:
        batch_op.add_column(sa.Column("query_family", sa.Text(), server_default=""))
        batch_op.add_column(sa.Column("search_policy_version", sa.Text(), server_default=""))
    with op.batch_alter_table("gap_audits") as batch_op:
        batch_op.add_column(sa.Column("search_policy_version", sa.Text(), server_default=""))
        batch_op.add_column(sa.Column("search_admission_status", sa.Text(), server_default="UNKNOWN"))
        batch_op.add_column(sa.Column("search_admission_reasons_json", sa.Text(), server_default="[]"))
        batch_op.add_column(sa.Column("search_query_ids_json", sa.Text(), server_default="[]"))


def downgrade() -> None:
    with op.batch_alter_table("gap_audits") as batch_op:
        batch_op.drop_column("search_query_ids_json")
        batch_op.drop_column("search_admission_reasons_json")
        batch_op.drop_column("search_admission_status")
        batch_op.drop_column("search_policy_version")
    with op.batch_alter_table("search_query_records") as batch_op:
        batch_op.drop_column("search_policy_version")
        batch_op.drop_column("query_family")

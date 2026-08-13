"""Add distinct supporting/contradicting paper counts to coverage_records.

Coverage score is driven by how many *distinct papers* speak to a question, not
how many evidence units were extracted, but that breadth signal was only ever
written into a trace JSON blob — unqueryable for later analysis or the frontend.
"""

from alembic import op
import sqlalchemy as sa


revision = "0018_coverage_breadth"
down_revision = "0017_audited_claim"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "coverage_records"):
        return
    with op.batch_alter_table("coverage_records") as batch_op:
        if not _has_column(bind, "coverage_records", "distinct_supporting_papers"):
            batch_op.add_column(sa.Column("distinct_supporting_papers", sa.Integer(), server_default="0"))
        if not _has_column(bind, "coverage_records", "distinct_contradicting_papers"):
            batch_op.add_column(sa.Column("distinct_contradicting_papers", sa.Integer(), server_default="0"))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "coverage_records"):
        return
    with op.batch_alter_table("coverage_records") as batch_op:
        if _has_column(bind, "coverage_records", "distinct_supporting_papers"):
            batch_op.drop_column("distinct_supporting_papers")
        if _has_column(bind, "coverage_records", "distinct_contradicting_papers"):
            batch_op.drop_column("distinct_contradicting_papers")

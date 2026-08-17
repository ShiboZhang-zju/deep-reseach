"""Add an open-access flag to papers.

The flag is harvested for free from OpenAlex's search response and used by
scoring to deprioritise paywalled papers with no OA route, so they don't enter
the evidence pool only to fail PDF fetch later.
"""

from alembic import op
import sqlalchemy as sa


revision = "0019_paper_is_oa"
down_revision = "0018_coverage_breadth"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "papers"):
        return
    with op.batch_alter_table("papers") as batch_op:
        if not _has_column(bind, "papers", "is_oa"):
            batch_op.add_column(sa.Column("is_oa", sa.Boolean(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "papers"):
        return
    with op.batch_alter_table("papers") as batch_op:
        if _has_column(bind, "papers", "is_oa"):
            batch_op.drop_column("is_oa")

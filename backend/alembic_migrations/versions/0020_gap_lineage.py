"""Add gap lineage columns (canonical_gap_id, parent_gap_id).

Narrowing a gap rewrites its claimed delta, which historically happened in
place — the previous claim was overwritten and the audit history was the only
record of the old version. Two semantically identical gaps could then surface
as sibling rows with conflicting audit verdicts (one rejected, one "novel"),
because nothing tied the narrowed version back to its origin.

canonical_gap_id points at the lineage root (NULL on the root itself), and
parent_gap_id points at the immediately superseded version. The report then
shows one row per canonical gap (its latest non-superseded version).
"""

from alembic import op
import sqlalchemy as sa


revision = "0020_gap_lineage"
down_revision = "0019_paper_is_oa"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "gap_candidates"):
        return
    with op.batch_alter_table("gap_candidates") as batch_op:
        if not _has_column(bind, "gap_candidates", "canonical_gap_id"):
            batch_op.add_column(sa.Column("canonical_gap_id", sa.String(), nullable=True))
        if not _has_column(bind, "gap_candidates", "parent_gap_id"):
            batch_op.add_column(sa.Column("parent_gap_id", sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "gap_candidates"):
        return
    with op.batch_alter_table("gap_candidates") as batch_op:
        if _has_column(bind, "gap_candidates", "canonical_gap_id"):
            batch_op.drop_column("canonical_gap_id")
        if _has_column(bind, "gap_candidates", "parent_gap_id"):
            batch_op.drop_column("parent_gap_id")

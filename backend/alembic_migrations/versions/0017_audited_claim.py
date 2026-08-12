"""Record which claim an audit judged, so a re-audit can tell new input apart.

Without it, "the audit input is unchanged" cannot be distinguished from "the
gap's claim was just narrowed and needs a fresh verdict": both look identical
from the audit table.
"""

from alembic import op
import sqlalchemy as sa


revision = "0017_audited_claim"
down_revision = "0016_confidence_tier"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "gap_audits") and not _has_column(bind, "gap_audits", "audited_claimed_delta"):
        with op.batch_alter_table("gap_audits") as batch_op:
            batch_op.add_column(sa.Column("audited_claimed_delta", sa.Text()))


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "gap_audits") and _has_column(bind, "gap_audits", "audited_claimed_delta"):
        with op.batch_alter_table("gap_audits") as batch_op:
            batch_op.drop_column("audited_claimed_delta")

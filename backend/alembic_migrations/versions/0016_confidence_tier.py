"""O1: graded output — add confidence_tier to interventions and ideas."""

from alembic import op
import sqlalchemy as sa


revision = "0016_confidence_tier"
down_revision = "0015_gap_search_admission"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    # These tables are created by the Phase 3A opportunity-pipeline migrations
    # or by ORM create_all. On a legacy DB that predates them, they may be
    # absent at this point; guard so the migration is safe either way.
    for table in ("intervention_candidates", "research_ideas"):
        if _has_table(bind, table) and not _has_column(bind, table, "confidence_tier"):
            with op.batch_alter_table(table) as batch_op:
                batch_op.add_column(sa.Column("confidence_tier", sa.Text(), server_default="C"))


def downgrade() -> None:
    bind = op.get_bind()
    for table in ("research_ideas", "intervention_candidates"):
        if _has_table(bind, table) and _has_column(bind, table, "confidence_tier"):
            with op.batch_alter_table(table) as batch_op:
                batch_op.drop_column("confidence_tier")

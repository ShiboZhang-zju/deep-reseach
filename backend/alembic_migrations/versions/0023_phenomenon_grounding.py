"""Ground the phenomenon plan in the gap's mechanism and a non-arbitrary criterion.

The first cut of the phenomenon gate emitted a kill criterion like "FPR < 10%"
with no source — a plausible-sounding but arbitrary threshold. These columns
force the plan to name the mechanism under test, the specific gap claim a
passing experiment would support, the H0/H1 comparator, an alternative
explanation, and the basis (prior literature / baseline variance / minimum
meaningful effect) behind the kill threshold.
"""

from alembic import op
import sqlalchemy as sa


revision = "0023_phenomenon_grounding"
down_revision = "0022_gap_phenomenon_plans"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "gap_phenomenon_plans"):
        return
    cols = {
        "mechanism_under_test": sa.Text(),
        "supports_gap_claim": sa.Text(),
        "expected_observation": sa.Text(),
        "alternative_explanation": sa.Text(),
        "comparator": sa.Text(),
        "kill_criterion_basis": sa.Text(),
    }
    with op.batch_alter_table("gap_phenomenon_plans") as batch_op:
        for name, typ in cols.items():
            if not _has_column(bind, "gap_phenomenon_plans", name):
                batch_op.add_column(sa.Column(name, typ, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "gap_phenomenon_plans"):
        return
    with op.batch_alter_table("gap_phenomenon_plans") as batch_op:
        for name in ("mechanism_under_test", "supports_gap_claim",
                     "expected_observation", "alternative_explanation",
                     "comparator", "kill_criterion_basis"):
            if _has_column(bind, "gap_phenomenon_plans", name):
                batch_op.drop_column(name)

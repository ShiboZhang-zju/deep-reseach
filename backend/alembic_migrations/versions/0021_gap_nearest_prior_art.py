"""Add nearest-prior-art provenance to gap candidates.

A surviving gap's novelty used to be reduced to a single novelty_confidence
float, which hides which paper is the closest known prior work and how much
retrieval confidence backs the "novel" claim. These four columns let the report
state, per surviving gap: the closest prior work, the residual (uncovered) gap,
and how confident the adversarial search is that it found the real prior art.
"""

from alembic import op
import sqlalchemy as sa


revision = "0021_gap_nearest_prior_art"
down_revision = "0020_gap_lineage"
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
        if not _has_column(bind, "gap_candidates", "nearest_prior_art_paper_id"):
            batch_op.add_column(sa.Column("nearest_prior_art_paper_id", sa.String(), nullable=True))
        if not _has_column(bind, "gap_candidates", "nearest_prior_art_title"):
            batch_op.add_column(sa.Column("nearest_prior_art_title", sa.String(), nullable=True))
        if not _has_column(bind, "gap_candidates", "residual_gap"):
            batch_op.add_column(sa.Column("residual_gap", sa.Text(), nullable=True))
        if not _has_column(bind, "gap_candidates", "search_confidence"):
            batch_op.add_column(sa.Column("search_confidence", sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_table(bind, "gap_candidates"):
        return
    with op.batch_alter_table("gap_candidates") as batch_op:
        for col in ("nearest_prior_art_paper_id", "nearest_prior_art_title",
                    "residual_gap", "search_confidence"):
            if _has_column(bind, "gap_candidates", col):
                batch_op.drop_column(col)

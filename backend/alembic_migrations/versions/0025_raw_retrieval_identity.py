"""P1-1 raw recall diagnostic: minimal raw-retrieval identity persistence.

Adds the `search_raw_results` table so the overlap diagnostic can reconstruct
the recall waterfall (raw -> canonical -> post-filter -> final Top-K) per query
family. Stores only the minimal audit fields (external id / doi / arxiv id /
title / year / raw rank), not the full API payload.
"""

from alembic import op
import sqlalchemy as sa


revision = "0025_raw_retrieval_identity"
down_revision = "0024_p11_coverage_saturation_npa"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def upgrade() -> None:
    bind = op.get_bind()

    if not _has_table(bind, "search_raw_results"):
        op.create_table(
            "search_raw_results",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("query_id", sa.String(), nullable=False),
            sa.Column("source", sa.Text(), nullable=True),
            sa.Column("raw_rank", sa.Integer(), nullable=True),
            sa.Column("external_paper_id", sa.Text(), nullable=True),
            sa.Column("doi", sa.Text(), nullable=True),
            sa.Column("arxiv_id", sa.Text(), nullable=True),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("year", sa.Integer(), nullable=True),
            sa.Column("canonical_paper_id", sa.String(), nullable=True),
            sa.Column("retrieved_at", sa.DateTime(), nullable=True),
        )
        op.create_index("idx_srr_query", "search_raw_results", ["query_id"])
        op.create_index("idx_srr_query_src_rank", "search_raw_results",
                        ["query_id", "source", "raw_rank"], unique=True)
        op.create_index("idx_srr_canonical", "search_raw_results",
                        ["canonical_paper_id"])


def downgrade() -> None:
    bind = op.get_bind()
    if _has_table(bind, "search_raw_results"):
        op.drop_table("search_raw_results")

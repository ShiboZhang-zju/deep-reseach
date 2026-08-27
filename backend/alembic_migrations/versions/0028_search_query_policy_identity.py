"""Align search query uniqueness with policy-aware idempotency.

Revision ID: 0028_search_query_policy_identity
Revises: 0027_gap_paper_relevance
Create Date: 2026-08-24

The application treats search_policy_version as part of a query's identity,
while the previous partial indexes did not. Rebuild both indexes so a policy
revision can persist a fresh query without colliding with an older audit.
"""
from alembic import op
import sqlalchemy as sa


revision = "0028_search_query_policy_identity"
down_revision = "0027_gap_paper_relevance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Older databases may have been created before 0011 or may have had the
    # indexes removed by a partial migration. SQLite supports IF EXISTS, which
    # keeps this policy migration replayable on those valid legacy schemas.
    op.execute("DROP INDEX IF EXISTS idx_sqr_unique_discovery")
    op.execute("DROP INDEX IF EXISTS idx_sqr_unique_gap")

    op.create_index(
        "idx_sqr_unique_discovery",
        "search_query_records",
        [
            "task_id",
            "round_number",
            "normalized_query_text",
            "target_question_id",
            "search_policy_version",
        ],
        unique=True,
        sqlite_where=sa.text("target_gap_id IS NULL"),
    )
    op.create_index(
        "idx_sqr_unique_gap",
        "search_query_records",
        [
            "task_id",
            "round_number",
            "normalized_query_text",
            "target_gap_id",
            "search_policy_version",
        ],
        unique=True,
        sqlite_where=sa.text("target_gap_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_sqr_unique_gap", table_name="search_query_records")
    op.drop_index("idx_sqr_unique_discovery", table_name="search_query_records")

    op.create_index(
        "idx_sqr_unique_discovery",
        "search_query_records",
        ["task_id", "round_number", "normalized_query_text", "target_question_id"],
        unique=True,
        sqlite_where=sa.text("target_gap_id IS NULL"),
    )
    op.create_index(
        "idx_sqr_unique_gap",
        "search_query_records",
        ["task_id", "round_number", "normalized_query_text", "target_gap_id"],
        unique=True,
        sqlite_where=sa.text("target_gap_id IS NOT NULL"),
    )

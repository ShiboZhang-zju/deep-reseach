"""Add explicit scoring status for research ideas.

Revision ID: 0030_idea_quality_status
Revises: 0029_gap_audit_diagnostics
Create Date: 2026-08-25
"""
from alembic import op
import sqlalchemy as sa


revision = "0030_idea_quality_status"
down_revision = "0029_gap_audit_diagnostics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if "research_ideas" not in table_names:
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("research_ideas")}
    if "score_status" not in columns:
        op.add_column(
            "research_ideas",
            sa.Column("score_status", sa.Text(), nullable=True, server_default="unscored"),
        )
    if "score_error" not in columns:
        op.add_column(
            "research_ideas",
            sa.Column("score_error", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("research_ideas", "score_error")
    op.drop_column("research_ideas", "score_status")

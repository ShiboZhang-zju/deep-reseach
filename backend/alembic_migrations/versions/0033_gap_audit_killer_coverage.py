"""Persist killer-work and search-coverage snapshots on gap audits.

The audit verdict is a SURVIVED_CURRENT_AUDIT statement, not a probability of
novelty: killer_work_json records what paper would kill the gap (and whether
the final adversarial search found it), search_coverage_json snapshots the
mechanical coverage facts the verdict rests on.

Revision ID: 0033_gap_audit_killer_coverage
Revises: 0032_idea_experiment_quality_contract
Create Date: 2026-08-28
"""
from alembic import op
import sqlalchemy as sa

revision = "0033_gap_audit_killer_coverage"
down_revision = "0032_idea_experiment_quality_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gap_audits",
        sa.Column("killer_work_json", sa.Text(), nullable=True, server_default="{}"),
    )
    op.add_column(
        "gap_audits",
        sa.Column("search_coverage_json", sa.Text(), nullable=True, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("gap_audits", "search_coverage_json")
    op.drop_column("gap_audits", "killer_work_json")

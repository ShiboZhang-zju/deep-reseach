"""Persist structured Gap audit failure reasons and evidence deltas.

Revision ID: 0029_gap_audit_diagnostics
Revises: 0028_search_query_policy_identity
Create Date: 2026-08-25
"""
from alembic import op
import sqlalchemy as sa


revision = "0029_gap_audit_diagnostics"
down_revision = "0028_search_query_policy_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gap_audits",
        sa.Column("failure_reason_codes_json", sa.Text(), nullable=True, server_default="[]"),
    )
    op.add_column(
        "gap_audits",
        sa.Column("evidence_delta_json", sa.Text(), nullable=True, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("gap_audits", "evidence_delta_json")
    op.drop_column("gap_audits", "failure_reason_codes_json")

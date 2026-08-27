"""Record claim-salvage diagnostics for partial gap audits.

Revision ID: 0031_gap_claim_salvage
Revises: 0030_idea_quality_status
Create Date: 2026-08-25
"""
from alembic import op
import sqlalchemy as sa


revision = "0031_gap_claim_salvage"
down_revision = "0030_idea_quality_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The audit explanation is already persisted in GapAudit's JSON diagnostics;
    # this no-op revision makes the salvage policy explicit and invalidates old
    # PhaseRun/audit input identities on upgrade.
    op.execute("SELECT 1")


def downgrade() -> None:
    op.execute("SELECT 1")

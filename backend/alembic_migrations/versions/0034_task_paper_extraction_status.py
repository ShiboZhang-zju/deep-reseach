"""task_papers: record whether the full text was actually assessed

Distinguishes "the full text was read and contained nothing relevant" from
"the full text was never retrieved/parsed" — the "tested negative" vs
"not tested" distinction. Without these columns the two collapse into the same
state (zero evidence units), and downstream coverage silently reads absence of
evidence as evidence of absence.

Additive only: two nullable columns, no data removed, no backfill.

Revision ID: 0034_task_paper_extraction_status
Revises: 0033_gap_audit_killer_coverage
Create Date: 2026-09-18
"""

import sqlalchemy as sa
from alembic import op

revision = "0034_task_paper_extraction_status"
down_revision = "0033_gap_audit_killer_coverage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("task_papers") as batch:
        batch.add_column(sa.Column("extraction_status", sa.String(), nullable=True))
        batch.add_column(sa.Column("extraction_gap_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("task_papers") as batch:
        batch.drop_column("extraction_gap_reason")
        batch.drop_column("extraction_status")

"""Add mining_input_version to gap_candidates.

Gap-mining evidence-sensitive idempotency: the phase was previously skipped
when a PhaseRun with the same input_version existed, but input_version only
hashed {contract, round, pipeline, policy}. After an O2 remediation round adds
evidence without advancing current_round, re-entry reused the old result and
the new evidence was never mined. Stamping each gap with the evidence
fingerprint it was mined from lets the existing-gap short-circuit bind to the
same fingerprint, so remediation can trigger a real re-mine.

legacy rows get NULL, which never matches a non-empty fingerprint, so resume
re-mines exactly once and dedup merges with the old rows.
"""

from alembic import op
import sqlalchemy as sa


revision = "0026_gap_mining_input_version"
down_revision = "0025_raw_retrieval_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = [c["name"] for c in inspector.get_columns("gap_candidates")]
    if "mining_input_version" not in cols:
        op.add_column("gap_candidates",
                      sa.Column("mining_input_version", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = [c["name"] for c in inspector.get_columns("gap_candidates")]
    if "mining_input_version" in cols:
        op.drop_column("gap_candidates", "mining_input_version")

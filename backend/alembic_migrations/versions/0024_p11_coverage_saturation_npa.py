"""P1-1: four-layer coverage decoupling + Top-K NPA + search saturation.

Adds the canonical-atomic-claim and neighbor-claim-coverage tables, and the
columns that carry mechanical Evidence Quality / Search Saturation / NPA
stability per RQ and per gap. All new columns are nullable so pre-existing
tasks keep working (metrics fall back to on-the-fly aggregation).
"""

from alembic import op
import sqlalchemy as sa


revision = "0024_p11_coverage_saturation_npa"
down_revision = "0023_phenomenon_grounding"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _has_column(bind, table: str, column: str) -> bool:
    return any(col["name"] == column for col in sa.inspect(bind).get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()

    if not _has_table(bind, "gap_atomic_claims"):
        op.create_table(
            "gap_atomic_claims",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("task_id", sa.String(), nullable=False),
            sa.Column("gap_id", sa.String(), nullable=False),
            sa.Column("claim_index", sa.Integer(), nullable=False),
            sa.Column("claim_text", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("idx_gac_gap", "gap_atomic_claims", ["gap_id"])
        op.create_index("idx_gac_task_gap", "gap_atomic_claims", ["task_id", "gap_id"])

    if not _has_table(bind, "neighbor_claim_coverage"):
        op.create_table(
            "neighbor_claim_coverage",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("task_id", sa.String(), nullable=False),
            sa.Column("gap_id", sa.String(), nullable=False),
            sa.Column("neighbor_paper_id", sa.String(), nullable=False),
            sa.Column("claim_id", sa.String(), nullable=False),
            sa.Column("coverage", sa.Text(), nullable=False),
            sa.Column("rationale", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index("idx_ncc_gap", "neighbor_claim_coverage", ["gap_id"])
        op.create_index("idx_ncc_claim", "neighbor_claim_coverage", ["claim_id"])
        op.create_index("idx_ncc_unique", "neighbor_claim_coverage",
                        ["gap_id", "neighbor_paper_id", "claim_id"], unique=True)

    # Add columns to existing tables (nullable, idempotent).
    if _has_table(bind, "neighbor_comparisons"):
        with op.batch_alter_table("neighbor_comparisons") as b:
            if not _has_column(bind, "neighbor_comparisons", "why_not_closed"):
                b.add_column(sa.Column("why_not_closed", sa.Text(), nullable=True))

    if _has_table(bind, "gap_candidates"):
        with op.batch_alter_table("gap_candidates") as b:
            if not _has_column(bind, "gap_candidates", "npa_stability"):
                b.add_column(sa.Column("npa_stability", sa.Float(), nullable=True))
            if not _has_column(bind, "gap_candidates", "family_coverage"):
                b.add_column(sa.Column("family_coverage", sa.Float(), nullable=True))
            if not _has_column(bind, "gap_candidates", "residual_claim_ids_json"):
                b.add_column(sa.Column("residual_claim_ids_json", sa.Text(), nullable=True))

    if _has_table(bind, "coverage_records"):
        with op.batch_alter_table("coverage_records") as b:
            for col in ("evidence_quality", "fulltext_ratio", "directness"):
                if not _has_column(bind, "coverage_records", col):
                    b.add_column(sa.Column(col, sa.Float(), nullable=True))
            if not _has_column(bind, "coverage_records", "search_saturation"):
                b.add_column(sa.Column("search_saturation", sa.Text(), nullable=True))
            for col in ("last_round_marginal_papers", "last_round_marginal_evidence"):
                if not _has_column(bind, "coverage_records", col):
                    b.add_column(sa.Column(col, sa.Integer(), nullable=True))

    if _has_table(bind, "evidence_units"):
        with op.batch_alter_table("evidence_units") as b:
            if not _has_column(bind, "evidence_units", "discovered_round"):
                b.add_column(sa.Column("discovered_round", sa.Integer(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()

    for table in ("neighbor_claim_coverage", "gap_atomic_claims"):
        if _has_table(bind, table):
            op.drop_table(table)

    for table, cols in {
        "neighbor_comparisons": ["why_not_closed"],
        "gap_candidates": ["npa_stability", "family_coverage", "residual_claim_ids_json"],
        "coverage_records": ["evidence_quality", "fulltext_ratio", "directness",
                             "search_saturation", "last_round_marginal_papers",
                             "last_round_marginal_evidence"],
        "evidence_units": ["discovered_round"],
    }.items():
        if _has_table(bind, table):
            with op.batch_alter_table(table) as b:
                for col in cols:
                    if _has_column(bind, table, col):
                        b.drop_column(col)

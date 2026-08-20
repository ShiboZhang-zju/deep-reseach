"""Add GapPaperRelevance for gap-specific prior-art screening.

Gap-specific scoring (false-novelty audit, fd688ba6): the existing
TaskPaper.final_score measures a paper's relevance to the whole task / RQ, but
the NPA audit needs "how close is this paper to THIS gap". Audit-recalled
papers were stored with final_score=NULL and then out-ranked by broad surveys
in neighbor selection, so direct prior art never reached the NPA pool. This
table stores a paper's structured relevance to a specific gap (many-to-many),
scored cheaply on title+abstract before the deep NPA audit.
"""

from alembic import op
import sqlalchemy as sa


revision = "0027_gap_paper_relevance"
down_revision = "0026_gap_mining_input_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "gap_paper_relevance" not in inspector.get_table_names():
        op.create_table(
            "gap_paper_relevance",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("gap_id", sa.String(), nullable=False),
            sa.Column("paper_id", sa.String(), nullable=False),
            sa.Column("task_id", sa.String(), nullable=False),
            sa.Column("relevance_score", sa.Float(), nullable=False, default=0.0),
            sa.Column("problem_overlap", sa.String(), nullable=True),
            sa.Column("mechanism_overlap", sa.String(), nullable=True),
            sa.Column("evaluation_overlap", sa.String(), nullable=True),
            sa.Column("claim_overlap", sa.String(), nullable=True),
            sa.Column("addresses_claim_ids_json", sa.Text(), nullable=True, default="[]"),
            sa.Column("rationale", sa.Text(), nullable=True),
            sa.Column("scoring_version", sa.String(), nullable=True, default="gap-rel-v1"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("gap_id", "paper_id", name="uq_gpr_gap_paper"),
        )
        op.create_index("idx_gpr_gap", "gap_paper_relevance", ["gap_id"])
        op.create_index("idx_gpr_paper", "gap_paper_relevance", ["paper_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "gap_paper_relevance" in inspector.get_table_names():
        op.drop_table("gap_paper_relevance")

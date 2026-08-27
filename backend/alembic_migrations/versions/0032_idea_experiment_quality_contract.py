"""Add deterministic Idea and experiment quality contract fields.

Revision ID: 0032_idea_experiment_quality_contract
Revises: 0031_gap_claim_salvage
Create Date: 2026-08-26
"""
from alembic import op
import sqlalchemy as sa

revision = "0032_idea_experiment_quality_contract"
down_revision = "0031_gap_claim_salvage"
branch_labels = None
depends_on = None


def _add_if_missing(bind, table: str, column: sa.Column) -> None:
    if table not in set(sa.inspect(bind).get_table_names()):
        return
    columns = {item["name"] for item in sa.inspect(bind).get_columns(table)}
    if column.name not in columns:
        op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    _add_if_missing(bind, "research_ideas", sa.Column("quality_reason_codes_json", sa.Text(), nullable=True, server_default="[]"))
    for name in ("model_spec", "dataset_provenance", "oracle", "statistical_analysis", "resource_budget"):
        _add_if_missing(bind, "experiment_plans", sa.Column(name, sa.Text(), nullable=True))
    _add_if_missing(bind, "experiment_plans", sa.Column("scenario_atoms_json", sa.Text(), nullable=True, server_default="[]"))


def downgrade() -> None:
    bind = op.get_bind()
    for table, names in {
        "experiment_plans": ["scenario_atoms_json", "resource_budget", "statistical_analysis", "oracle", "dataset_provenance", "model_spec"],
        "research_ideas": ["quality_reason_codes_json"],
    }.items():
        if table not in set(sa.inspect(bind).get_table_names()):
            continue
        columns = {item["name"] for item in sa.inspect(bind).get_columns(table)}
        for name in names:
            if name in columns:
                op.drop_column(table, name)

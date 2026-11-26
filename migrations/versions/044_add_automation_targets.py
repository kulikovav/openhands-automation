"""Add automation execution targets.

Revision ID: 044
Revises: 043
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "044"
down_revision: str = "043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    # The pre-1.36 fork added these columns under revision 042. A database that
    # ran that release already has them, so add only what is missing.
    if "target_id" not in _existing_columns("automations"):
        op.add_column(
            "automations",
            sa.Column("target_id", sa.String(length=64), nullable=True),
        )
    run_columns = _existing_columns("automation_runs")
    if "target_id" not in run_columns:
        op.add_column(
            "automation_runs",
            sa.Column("target_id", sa.String(length=64), nullable=True),
        )
    # The target's configured name at run creation; the run keeps this
    # snapshot when the automation's target later changes.
    if "target_name" not in run_columns:
        op.add_column(
            "automation_runs",
            sa.Column("target_name", sa.String(length=255), nullable=True),
        )
    # The governed-run execution claim: {target_id, target_url,
    # conversation_id, claimed_at}. NULL on every legacy row.
    if "execution_claim" not in run_columns:
        op.add_column(
            "automation_runs",
            sa.Column("execution_claim", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("automation_runs", "execution_claim")
    op.drop_column("automation_runs", "target_name")
    op.drop_column("automation_runs", "target_id")
    op.drop_column("automations", "target_id")

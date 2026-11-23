"""Add automation execution targets.

Revision ID: 042
Revises: 041
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "042"
down_revision: str = "041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "automations",
        sa.Column("target_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "automation_runs",
        sa.Column("target_id", sa.String(length=64), nullable=True),
    )
    # The target's configured name at run creation; the run keeps this
    # snapshot when the automation's target later changes.
    op.add_column(
        "automation_runs",
        sa.Column("target_name", sa.String(length=255), nullable=True),
    )
    # The governed-run execution claim: {target_id, target_url,
    # conversation_id, claimed_at}. NULL on every legacy row.
    op.add_column(
        "automation_runs",
        sa.Column("execution_claim", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("automation_runs", "execution_claim")
    op.drop_column("automation_runs", "target_name")
    op.drop_column("automation_runs", "target_id")
    op.drop_column("automations", "target_id")

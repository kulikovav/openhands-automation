"""Keep a governed run's conversation and transcript after its sandbox stops.

Revision ID: 043
Revises: 042
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "043"
down_revision: str = "042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing automations keep their transcripts: the retention redesign
    # changes the old delete-at-terminal behavior into the opt-out.
    op.add_column(
        "automations",
        sa.Column(
            "retain_transcript",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("automations", "retain_transcript")

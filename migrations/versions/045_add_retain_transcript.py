"""Keep a governed run's conversation and transcript after its sandbox stops.

Revision ID: 045
Revises: 044
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "045"
down_revision: str = "044"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The pre-1.36 fork added this column under revision 043. A database that
    # ran that release already has it, so add only what is missing.
    existing = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("automations")
    }
    if "retain_transcript" in existing:
        return
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

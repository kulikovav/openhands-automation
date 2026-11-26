"""Repair the columns the pre-1.36 fork's revision 043 shadowed.

The pre-1.36 fork stamped its database 043, and upstream 1.36.0 uses the
same revision numbers for its outcome migrations. Alembic therefore treated
them as applied and skipped them. Apply them once, only where their columns
are missing.

Revision ID: 046
Revises: 045
"""

import importlib.util
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from alembic import op


revision: str = "046"
down_revision: str = "045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _revision_module(filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(f"repair_{filename[:-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    factory_columns = {column["name"] for column in inspector.get_columns("factories")}
    automation_columns = {
        column["name"] for column in inspector.get_columns("automations")
    }
    if "outcome_keys" not in factory_columns:
        _revision_module("042_add_automation_outcome_values.py").upgrade()
    if "outcome_parsers" not in automation_columns:
        _revision_module("043_add_automation_outcome_parsers.py").upgrade()


def downgrade() -> None:
    # The repaired schema belongs to upstream revisions 042 and 043, which a
    # legacy database still records as applied. Dropping it here would leave
    # that database inconsistent with its own version table.
    pass

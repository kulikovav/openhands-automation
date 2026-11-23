"""Migration 042: automation targets and the run execution claim."""

import os
import sqlite3

from sqlalchemy import create_engine, inspect


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alembic(db_url: str, *args: str) -> str:
    env = os.environ.copy()
    env["AUTOMATION_DB_URL"] = db_url
    import subprocess

    result = subprocess.run(
        ["uv", "run", "alembic", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"alembic {args} failed: {result.stderr}"
    return result.stdout + result.stderr


def test_upgrade_adds_target_columns_on_an_empty_database(tmp_path):
    db_path = str(tmp_path / "targets.db")
    db_url = f"sqlite:///{db_path}"
    _alembic(db_url, "upgrade", "head")

    assert _alembic(db_url, "heads").count("(head)") == 1
    engine = create_engine(db_url)
    inspector = inspect(engine)
    automation_columns = {c["name"] for c in inspector.get_columns("automations")}
    run_columns = {c["name"] for c in inspector.get_columns("automation_runs")}
    engine.dispose()

    assert "target_id" in automation_columns
    assert {"target_id", "target_name", "execution_claim"} <= run_columns

    # A pre-existing row keeps null targets: the deployment default.
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO automations (id, user_id, created_by_user_id, org_id, name,"
            " trigger, tarball_path, entrypoint, enabled)"
            " VALUES (?,?,?,?,?,?,?,?,1)",
            (
                "a" * 32,
                "b" * 32,
                "c" * 32,
                "d" * 32,
                "legacy",
                "{}",
                "s3://bucket/code.tar.gz",
                "run",
            ),
        )
        row = con.execute(
            "SELECT target_id FROM automations WHERE name = 'legacy'"
        ).fetchone()
    assert row == (None,)


def test_downgrade_removes_target_columns(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'down.db'}"
    _alembic(db_url, "upgrade", "head")
    _alembic(db_url, "downgrade", "041")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    automation_columns = {c["name"] for c in inspector.get_columns("automations")}
    run_columns = {c["name"] for c in inspector.get_columns("automation_runs")}
    engine.dispose()

    assert "target_id" not in automation_columns
    assert not {"target_id", "target_name", "execution_claim"} & run_columns

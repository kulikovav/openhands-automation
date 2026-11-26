"""Migration 045: the per-automation governed transcript retention flag."""

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


def test_upgrade_adds_retention_on_an_empty_database(tmp_path):
    db_path = str(tmp_path / "retention.db")
    db_url = f"sqlite:///{db_path}"
    _alembic(db_url, "upgrade", "head")

    assert _alembic(db_url, "heads").count("(head)") == 1
    engine = create_engine(db_url)
    inspector = inspect(engine)
    automation_columns = {c["name"] for c in inspector.get_columns("automations")}
    engine.dispose()

    assert "retain_transcript" in automation_columns

    # An automation created before the flag keeps its transcript.
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
            "SELECT retain_transcript FROM automations WHERE name = 'legacy'"
        ).fetchone()
    assert row == (1,)


def test_downgrade_removes_retention(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'down.db'}"
    _alembic(db_url, "upgrade", "head")
    _alembic(db_url, "downgrade", "044")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    automation_columns = {c["name"] for c in inspector.get_columns("automations")}
    engine.dispose()

    assert "retain_transcript" not in automation_columns

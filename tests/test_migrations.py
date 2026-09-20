from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The revision a pre-pivot deployment is stamped with.
LEGACY_REVISION = "6741599e8157"
HEAD_REVISION = "a1b2c3d4e5f6"


def _upgrade(db_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the migration command with the current platform environment."""
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_path}"

    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _version(db_path: Path) -> str | None:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def _tables(db_path: Path) -> set[str]:
    con = sqlite3.connect(db_path)
    try:
        return {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        con.close()


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    """A database as a pre-pivot deployment left it."""
    db = tmp_path / "legacy.db"

    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE alembic_version "
        "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
    )
    con.execute(
        "INSERT INTO alembic_version VALUES (?)",
        (LEGACY_REVISION,),
    )
    con.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, note TEXT)"
    )
    con.execute(
        "INSERT INTO sessions VALUES "
        "('s1', 'a real interview record')"
    )
    con.commit()
    con.close()

    return db


def test_a_legacy_database_upgrades(legacy_db: Path) -> None:
    """The regression: this exited 255 and took the whole service down."""
    result = _upgrade(legacy_db)

    assert result.returncode == 0, result.stderr
    assert _version(legacy_db) == HEAD_REVISION
    assert "candidates" in _tables(legacy_db)


def test_upgrading_a_legacy_database_destroys_nothing(
    legacy_db: Path,
) -> None:
    """Old interview records must not be deleted."""
    assert _upgrade(legacy_db).returncode == 0

    con = sqlite3.connect(legacy_db)
    try:
        assert (
            con.execute("SELECT note FROM sessions").fetchone()[0]
            == "a real interview record"
        )
    finally:
        con.close()


def test_a_fresh_database_upgrades(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"

    result = _upgrade(db)

    assert result.returncode == 0, result.stderr
    assert _version(db) == HEAD_REVISION


def test_a_fresh_database_gets_no_dead_legacy_tables(
    tmp_path: Path,
) -> None:
    """The legacy revision resolves without recreating obsolete tables."""
    db = tmp_path / "fresh.db"

    assert _upgrade(db).returncode == 0
    assert _tables(db) == {"alembic_version", "candidates"}


def test_upgrade_is_idempotent(legacy_db: Path) -> None:
    """Every container start runs this, not just the first."""
    assert _upgrade(legacy_db).returncode == 0
    assert _upgrade(legacy_db).returncode == 0
    assert _version(legacy_db) == HEAD_REVISION
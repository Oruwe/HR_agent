"""SQLAlchemy engine and session factory.

``DATABASE_URL`` selects the backend. With nothing configured we default to a
file-backed SQLite database under ``./data`` -- consistent with the rest of
the project's "zero infrastructure required" default. Point it at Postgres
(``postgresql+psycopg://...``) in production; nothing else in this module
needs to change.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def _default_sqlite_url() -> str:
    data_dir = Path(os.environ.get("HRTE_DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'app.db'}"


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip() or _default_sqlite_url()


DATABASE_URL = _database_url()

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create tables if they don't exist yet.

    Used by tests and by ``make demo``/local dev so the app runs with zero
    setup. Real deployments should prefer ``alembic upgrade head`` (see
    ``app/db/migrations``) so schema changes are tracked and reversible.
    """
    from app.db import models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["DATABASE_URL", "Base", "SessionLocal", "engine", "get_db", "init_db"]

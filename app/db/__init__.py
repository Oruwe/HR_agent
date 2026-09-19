"""Persistence layer: the Primary DB in the architecture diagram.

SQLite by default (zero infrastructure, matches the rest of the project's
"works with nothing configured" philosophy); point ``DATABASE_URL`` at Postgres
in production. SQLAlchemy 2.0 models, Alembic migrations under
``app/db/migrations``.
"""

from app.db.engine import Base, SessionLocal, engine, get_db, init_db

__all__ = ["Base", "SessionLocal", "engine", "get_db", "init_db"]

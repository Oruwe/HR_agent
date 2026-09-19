"""Fixtures for the FastAPI layer.

Each test gets its own in-memory SQLite database via a dependency override on
``get_db``, rather than by touching the process-wide engine, so tests never
see another test's candidates and can run in any order.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    from app.api.app import app
    from app.api.routes_candidates import reset_index
    from app.db import models  # noqa: F401  registers tables on Base.metadata
    from app.db.engine import Base, get_db

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def _override_get_db() -> Iterator:
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # The pool index is process-global so Moss is loaded once, which means
    # it would otherwise leak documents between tests.
    reset_index()
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        reset_index()
        engine.dispose()


@pytest.fixture
def imported(api_client: TestClient):
    """Import records and return the resulting pool."""

    def _import(records: list[dict]) -> list[dict]:
        response = api_client.post("/api/candidates/import", json={"candidates": records})
        assert response.status_code == 201, response.text
        return api_client.get("/api/candidates").json()

    return _import

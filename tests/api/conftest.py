"""Fixtures for the FastAPI layer.

Each test gets its own in-memory SQLite database (via a FastAPI dependency
override on ``get_db``, not by touching the process-wide engine) and a
freshly reset in-process session store, so tests never see another test's
sessions and can run safely in any order.
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
    from app.api.session_store import reset_session_store
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

    reset_session_store()
    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        reset_session_store()
        engine.dispose()


@pytest.fixture
def create_session(api_client: TestClient):
    def _create(
        resume: str = "Backend engineer. Kubernetes, gRPC, Postgres, distributed systems.",
    ) -> dict:
        response = api_client.post("/api/sessions", json={"resume_text": resume})
        assert response.status_code == 201, response.text
        return response.json()

    return _create

"""Tests for the Primary DB layer: models, relationships, migrations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.engine import Base
from app.db.models import EvaluationRecord, SessionRecord, TurnRecord


def _memory_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)()


def test_session_turn_evaluation_relationships() -> None:
    db = _memory_session()
    session = SessionRecord(
        session_id="s1",
        candidate_id="c1",
        sanitized_name="Candidate_001",
        role="AI_ML_SYSTEMS_ENGINEER",
        token_hash="deadbeef",
        retrieval_backend="embedded fallback",
    )
    db.add(session)
    db.add(TurnRecord(session_id="s1", speaker="agent", text="Hello.", offset_ms=0.0))
    db.add(TurnRecord(session_id="s1", speaker="candidate", text="Hi.", offset_ms=100.0))
    db.commit()

    fetched = db.get(SessionRecord, "s1")
    assert fetched is not None
    assert len(fetched.turns) == 2
    assert fetched.turns[0].offset_ms == 0.0
    assert fetched.evaluation is None

    db.add(
        EvaluationRecord(
            session_id="s1",
            target_role="AI_ML_SYSTEMS_ENGINEER",
            rubric_fit_index=0.75,
            recommendation="ADVANCE",
            competency_scores={"distributed_training": 0.8},
        )
    )
    db.commit()
    db.refresh(fetched)
    assert fetched.evaluation is not None
    assert fetched.evaluation.recommendation == "ADVANCE"


def test_session_to_summary_shape() -> None:
    db = _memory_session()
    session = SessionRecord(
        session_id="s2",
        candidate_id="c2",
        sanitized_name="Candidate_002",
        role="FRONTEND_PLATFORM_ENGINEER",
        status="active",
        token_hash="abc123",
        retrieval_backend="embedded fallback",
    )
    db.add(session)
    db.commit()
    summary = session.to_summary()
    assert summary["session_id"] == "s2"
    assert summary["status"] == "active"
    assert summary["turns"] == 0
    assert summary["has_evaluation"] is False


def test_alembic_migration_applies_cleanly(tmp_path: Path) -> None:
    """The migration this project ships must actually run against a fresh DB,
    not merely exist. Regenerating the schema from the model (as
    Base.metadata.create_all does in the fixtures above) would not catch a
    broken revision file."""
    import os

    db_file = tmp_path / "migration_check.db"
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{db_file}"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert db_file.exists()

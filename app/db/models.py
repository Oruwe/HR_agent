"""ORM models for the Primary DB.

Three tables, deliberately narrow:

* ``sessions``    -- one row per screening call, created at intake.
* ``turns``        -- the persisted transcript, already PII-scrubbed by the
                       time it reaches this layer (see app/schemas/candidate.py).
* ``evaluations``  -- the dossier produced at close(), one-to-one with a
                       session.

Nothing here stores a raw identifier: every string that lands in these tables
has already passed through ``app.security.pii_scrubber`` upstream, in the
Pydantic models that build them. The database is an additional persistence
boundary, not a redaction boundary -- redaction happens once, at ingest.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.engine import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    candidate_id: Mapped[str] = mapped_column(String(36), index=True)
    sanitized_name: Mapped[str] = mapped_column(String(64))
    #: SHA-256 of the bearer token handed to the Candidate App at creation.
    #: Only the hash is ever persisted -- the token itself exists solely in
    #: the create-session HTTP response and the client's memory.
    token_hash: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    retrieval_backend: Mapped[str] = mapped_column(String(32), default="embedded")
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    updated_at: Mapped[int] = mapped_column(
        Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time())
    )

    turns: Mapped[list[TurnRecord]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="TurnRecord.offset_ms"
    )
    evaluation: Mapped[EvaluationRecord | None] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )

    def to_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sanitized_name": self.sanitized_name,
            "role": self.role,
            "status": self.status,
            "retrieval_backend": self.retrieval_backend,
            "turns": len(self.turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_evaluation": self.evaluation is not None,
            "rubric_fit_index": self.evaluation.rubric_fit_index if self.evaluation else None,
            "recommendation": self.evaluation.recommendation if self.evaluation else None,
        }


class TurnRecord(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    speaker: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    offset_ms: Mapped[float] = mapped_column(Float, default=0.0)
    turnaround_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    within_budget: Mapped[bool | None] = mapped_column(nullable=True)

    session: Mapped[SessionRecord] = relationship(back_populates="turns")

    __table_args__ = (Index("ix_turns_session_offset", "session_id", "offset_ms"),)


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), primary_key=True)
    target_role: Mapped[str] = mapped_column(String(64))
    rubric_fit_index: Mapped[float] = mapped_column(Float)
    routing_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation: Mapped[str] = mapped_column(String(32), index=True)
    turns_completed: Mapped[int] = mapped_column(Integer, default=0)
    latency_compliance: Mapped[float] = mapped_column(Float, default=1.0)
    competency_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    eliminations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    flagged_limitations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))

    session: Mapped[SessionRecord] = relationship(back_populates="evaluation")


__all__ = ["EvaluationRecord", "SessionRecord", "TurnRecord"]

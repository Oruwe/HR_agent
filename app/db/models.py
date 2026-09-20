"""ORM models. One table: the candidate pool.

Candidates arrive as scraped JSON from an external tool, so the record shape
is not ours to define and will change without warning. The full payload is
therefore stored verbatim (after PII scrubbing) in ``source`` rather than
shredded into columns, and the handful of columns that do exist are a
best-effort display/ranking cache derived from it. A scraper that starts
emitting a new field keeps working; it just isn't promoted to a column until
someone decides it should be.

Nothing here stores raw PII: every string lands here having already passed
through ``app.security.pii_scrubber`` at the import boundary.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import JSON, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.engine import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    #: Best-effort display name pulled from the scraped record. Never a legal
    #: identifier -- it is whatever the source called them, scrubbed.
    name: Mapped[str] = mapped_column(String(200), default="Unknown candidate")
    #: One-line summary (title, headline, current role) when the source has one.
    headline: Mapped[str] = mapped_column(String(400), default="")
    #: The scraped record, verbatim and scrubbed. The analyst reads this.
    source: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    imported_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))

    # -- the analyst's verdict, null until an analysis run has covered them ---
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyzed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "headline": self.headline,
            "score": self.score,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "imported_at": self.imported_at,
            "analyzed_at": self.analyzed_at,
        }

    def to_detail(self) -> dict[str, Any]:
        return {**self.to_summary(), "source": self.source}


__all__ = ["Candidate"]

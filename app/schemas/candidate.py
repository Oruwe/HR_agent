"""Candidate-facing models. Every one of them is a *post-redaction* structure.

There is deliberately no model in this package that can hold a raw identifier.
The type system is the cheapest place to enforce the zero-leak invariant: if
the only way to construct a :class:`CandidateProfile` is through
:meth:`CandidateProfile.from_resume`, which scrubs on the way in, then no code
path downstream can accidentally persist an Aadhaar number.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.roles import EngineeringRole
from app.security.pii_scrubber import ScrubResult, scrub


class Speaker(StrEnum):
    CANDIDATE = "candidate"
    AGENT = "agent"


def pseudonym_for(candidate_id: str) -> str:
    """Stable display pseudonym, e.g. ``Candidate_491``.

    Derived from a BLAKE2 digest of the candidate id, so it is consistent
    across processes and replays without a lookup table. It is a *display*
    label only -- uniqueness is carried by ``candidate_id``, and the three-digit
    suffix is expected to collide in a large cohort.
    """
    digest = hashlib.blake2b(candidate_id.encode("utf-8"), digest_size=4).digest()
    return f"Candidate_{int.from_bytes(digest, 'big') % 1000:03d}"


class TranscriptTurn(BaseModel):
    """One conversational turn, scrubbed at construction time."""

    model_config = ConfigDict(frozen=True)

    speaker: Speaker
    text: str
    #: Milliseconds since session start, not wall-clock: a session recording
    #: timestamped with wall-clock time is itself weakly identifying.
    offset_ms: float = Field(ge=0.0)
    #: Measured turnaround for agent turns; None for candidate turns.
    turnaround_ms: float | None = None
    redaction_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _must_be_scrubbed(cls, v: str) -> str:
        # Belt and braces: even if a caller bypasses `sanitised`, the model
        # itself will not hold raw PII.
        return scrub(v).text

    @classmethod
    def sanitised(
        cls,
        speaker: Speaker,
        raw_text: str,
        offset_ms: float,
        turnaround_ms: float | None = None,
    ) -> Self:
        result: ScrubResult = scrub(raw_text)
        return cls(
            speaker=speaker,
            text=result.text,
            offset_ms=offset_ms,
            turnaround_ms=turnaround_ms,
            redaction_counts=result.counts(),
        )


class CandidateProfile(BaseModel):
    """A screening subject, reduced to what the rubric actually needs.

    Note what is absent: legal name, contact details, address, date of birth,
    photograph, employer references. None of it improves a technical fit score,
    and all of it is liability. The agent screens competencies, not people.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sanitized_name: str = ""
    target_role: EngineeringRole | None = None
    #: Fully scrubbed resume text, safe for embedding and LLM egress.
    resume_text: str = ""
    #: Self-declared years of professional experience, if stated.
    years_experience: float | None = Field(default=None, ge=0.0, le=60.0)
    #: Per-class counts of what was removed on ingest. Audit trail, no values.
    redaction_counts: dict[str, int] = Field(default_factory=dict)
    ingested_at: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("resume_text")
    @classmethod
    def _resume_must_be_scrubbed(cls, v: str) -> str:
        return scrub(v).text

    def model_post_init(self, _context: object) -> None:
        if not self.sanitized_name:
            # frozen model: assign through __dict__, the documented pydantic
            # escape hatch for post-init derivation.
            object.__setattr__(self, "sanitized_name", pseudonym_for(self.candidate_id))

    @classmethod
    def from_resume(
        cls,
        raw_resume: str,
        *,
        candidate_id: str | None = None,
        target_role: EngineeringRole | None = None,
        years_experience: float | None = None,
    ) -> Self:
        """The only sanctioned ingest path. Scrubs before anything else runs."""
        result = scrub(raw_resume)
        cid = candidate_id or str(uuid.uuid4())
        return cls(
            candidate_id=cid,
            sanitized_name=pseudonym_for(cid),
            target_role=target_role,
            resume_text=result.text,
            years_experience=years_experience,
            redaction_counts=result.counts(),
        )

    @property
    def was_redacted(self) -> bool:
        return bool(self.redaction_counts)


class ScreeningSession(BaseModel):
    """Live session state. Held in memory for the duration of the interview."""

    model_config = ConfigDict(frozen=False)

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate: CandidateProfile
    role: EngineeringRole
    room_name: str = ""
    started_at: int = Field(default_factory=lambda: int(time.time()))
    transcript: list[TranscriptTurn] = Field(default_factory=list)

    def record(self, turn: TranscriptTurn) -> None:
        self.transcript.append(turn)

    def candidate_evidence(self) -> str:
        """All candidate speech plus the resume: the evidence the rubric sees.

        Agent turns are excluded on purpose. They contain the rubric's own
        vocabulary -- scoring against them would let the interviewer's questions
        inflate the candidate's score, which is the classic self-confirming
        evaluation bug.
        """
        spoken = " ".join(t.text for t in self.transcript if t.speaker is Speaker.CANDIDATE)
        return f"{self.candidate.resume_text} {spoken}".strip()

    @property
    def duration_ms(self) -> float:
        return self.transcript[-1].offset_ms if self.transcript else 0.0


__all__ = [
    "CandidateProfile",
    "ScreeningSession",
    "Speaker",
    "TranscriptTurn",
    "pseudonym_for",
]

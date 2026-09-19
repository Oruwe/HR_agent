"""Evaluation outputs: scores, gates, recommendations, and the stored payload.

The dossier is the artefact a human recruiter reads and that an auditor may
later have to defend. It therefore records not just *what* was decided but
which rubric line drove it, so that "why was this candidate not advanced?"
always has a numbered answer.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.roles import (
    EngineeringRole,
    RoleMatch,
    rubric_for,
)
from app.security.pii_scrubber import scrub


class Recommendation(StrEnum):
    """Screening outcome. Never an offer -- see SOUL.md Boundaries."""

    ADVANCE = "ADVANCE"
    ADVANCE_WITH_RESERVATIONS = "ADVANCE_WITH_RESERVATIONS"
    HOLD_FOR_HUMAN_REVIEW = "HOLD_FOR_HUMAN_REVIEW"
    DO_NOT_ADVANCE = "DO_NOT_ADVANCE"
    INCOMPLETE = "INCOMPLETE"


class EvidenceSource(StrEnum):
    RESUME = "resume"
    LIVE_INTERVIEW = "live_interview"
    COMBINED = "combined"


class CompetencyScore(BaseModel):
    """One scored rubric dimension with its supporting rationale."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str = ""
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, gt=0.0, le=1.0)
    source: EvidenceSource = EvidenceSource.COMBINED
    rationale: str = ""

    @field_validator("rationale")
    @classmethod
    def _scrub_rationale(cls, v: str) -> str:
        # The rationale is model-generated free text that quotes the candidate.
        # It is the single most likely place for a stray phone number.
        return scrub(v).text


class EliminationFlag(BaseModel):
    """A triggered hard gate, traced to its rubric line."""

    model_config = ConfigDict(frozen=True)

    key: str
    description: str
    competency_key: str
    observed_score: float = Field(ge=0.0, le=1.0)
    floor: float = Field(ge=0.0, le=1.0)


class RoleFit(BaseModel):
    """How well the candidate fits one role, as a storable record."""

    model_config = ConfigDict(frozen=True)

    role: EngineeringRole
    fit_index: float = Field(ge=0.0, le=1.0)
    competency_scores: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_match(cls, match: RoleMatch) -> Self:
        return cls(
            role=match.role,
            fit_index=round(min(1.0, max(0.0, match.score)), 4),
            competency_scores=match.competency_scores(),
        )


class CandidateEvaluation(BaseModel):
    """The dossier. This is what gets stored, traced, and shown to a recruiter."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    sanitized_name: str
    session_id: str
    target_role: EngineeringRole
    rubric_fit_index: float = Field(ge=0.0, le=1.0)
    #: Separation between the top role and the runner-up; low values mean the
    #: evidence did not commit to a track.
    routing_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    competency_scores: list[CompetencyScore] = Field(default_factory=list)
    alternate_fits: list[RoleFit] = Field(default_factory=list)
    eliminations: list[EliminationFlag] = Field(default_factory=list)
    flagged_limitations: list[str] = Field(default_factory=list)
    recommendation: Recommendation = Recommendation.INCOMPLETE
    turns_completed: int = Field(default=0, ge=0)
    #: Fraction of agent turns that landed inside the latency budget.
    latency_compliance: float = Field(default=1.0, ge=0.0, le=1.0)
    interview_timestamp: int = Field(default_factory=lambda: int(time.time()))

    @field_validator("flagged_limitations")
    @classmethod
    def _scrub_limitations(cls, v: list[str]) -> list[str]:
        return [scrub(item).text for item in v]

    def score_map(self) -> dict[str, float]:
        return {c.key: round(c.score, 4) for c in self.competency_scores}

    def to_payload(self) -> dict[str, object]:
        """The exact Qdrant payload shape. Contains no identifiers by design."""
        return {
            "candidate_id": self.candidate_id,
            "sanitized_name": self.sanitized_name,
            "target_role": self.target_role.value,
            "competency_scores": self.score_map(),
            "rubric_fit_index": round(self.rubric_fit_index, 4),
            "flagged_limitations": list(self.flagged_limitations),
            "interview_timestamp": self.interview_timestamp,
        }


def derive_recommendation(
    role: EngineeringRole,
    fit_index: float,
    scores: dict[str, float],
    *,
    turns_completed: int = 0,
    min_turns: int = 3,
) -> tuple[Recommendation, list[EliminationFlag]]:
    """Map scores to an outcome. Pure, total, and auditable.

    The ordering of the gates is the policy, and it is deliberate:

    1. **Insufficient evidence wins over everything.** A call that dropped after
       one question is INCOMPLETE, never DO_NOT_ADVANCE. Rejecting a candidate
       because the network failed is the worst outcome this system can produce.
    2. **Elimination gates beat a high aggregate.** A candidate can be strong
       across three dimensions and still fail the one that defines the role.
    3. Only then does the weighted threshold decide.
    """
    rubric = rubric_for(role)

    flags: list[EliminationFlag] = []
    for criterion in rubric.eliminations:
        observed = scores.get(criterion.competency_key, 0.0)
        if observed <= criterion.floor:
            flags.append(
                EliminationFlag(
                    key=criterion.key,
                    description=criterion.description,
                    competency_key=criterion.competency_key,
                    observed_score=round(observed, 4),
                    floor=criterion.floor,
                )
            )

    if turns_completed < min_turns:
        return Recommendation.INCOMPLETE, flags

    if flags:
        # A strong candidate who tripped a gate is a human's call, not a
        # machine's -- the gate may simply have been probed badly.
        if fit_index >= rubric.advance_threshold:
            return Recommendation.HOLD_FOR_HUMAN_REVIEW, flags
        return Recommendation.DO_NOT_ADVANCE, flags

    if fit_index >= rubric.advance_threshold:
        return Recommendation.ADVANCE, flags
    if fit_index >= rubric.advance_threshold * 0.75:
        return Recommendation.ADVANCE_WITH_RESERVATIONS, flags
    return Recommendation.DO_NOT_ADVANCE, flags


__all__ = [
    "CandidateEvaluation",
    "CompetencyScore",
    "EliminationFlag",
    "EvidenceSource",
    "Recommendation",
    "RoleFit",
    "derive_recommendation",
]

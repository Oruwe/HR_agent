"""Typed contracts for candidates, roles and evaluations."""

from app.schemas.candidate import (
    CandidateProfile,
    ScreeningSession,
    Speaker,
    TranscriptTurn,
    pseudonym_for,
)
from app.schemas.evaluation import (
    CandidateEvaluation,
    CompetencyScore,
    EliminationFlag,
    EvidenceSource,
    Recommendation,
    RoleFit,
    derive_recommendation,
)
from app.schemas.roles import (
    ROLE_RUBRICS,
    Competency,
    CompetencyMatch,
    EliminationCriterion,
    EngineeringRole,
    RoleMatch,
    RoleRubric,
    best_role,
    match_confidence,
    rank_roles,
    rubric_for,
    score_role,
)

__all__ = [
    "ROLE_RUBRICS",
    "CandidateEvaluation",
    "CandidateProfile",
    "Competency",
    "CompetencyMatch",
    "CompetencyScore",
    "EliminationCriterion",
    "EliminationFlag",
    "EngineeringRole",
    "EvidenceSource",
    "Recommendation",
    "RoleFit",
    "RoleMatch",
    "RoleRubric",
    "ScreeningSession",
    "Speaker",
    "TranscriptTurn",
    "best_role",
    "derive_recommendation",
    "match_confidence",
    "pseudonym_for",
    "rank_roles",
    "rubric_for",
    "score_role",
]

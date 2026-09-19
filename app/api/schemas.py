"""Request/response models for the HTTP API. Kept separate from
``app/schemas`` on purpose: those are domain models with PII-scrubbing
validators baked in; these are wire shapes for one specific transport.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.roles import EngineeringRole


class CreateSessionRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=20_000)
    target_role: EngineeringRole | None = None
    years_experience: float | None = Field(default=None, ge=0, le=60)


class CreateSessionResponse(BaseModel):
    session_id: str
    session_token: str
    sanitized_name: str
    role: EngineeringRole
    redacted: bool
    greeting_text: str
    greeting_audio_b64: str | None = None
    retrieval_backend: str


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)


class TurnResponse(BaseModel):
    agent_text: str
    audio_b64: str | None = None
    turnaround_ms: float
    within_budget: bool
    speculation_hit: bool
    turns_completed: int
    finished: bool
    stt_used: bool = False
    stt_provider: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


class CompetencyScoreOut(BaseModel):
    key: str
    label: str
    score: float
    source: str
    rationale: str


class EvaluationResponse(BaseModel):
    candidate_id: str
    sanitized_name: str
    session_id: str
    target_role: EngineeringRole
    rubric_fit_index: float
    routing_confidence: float
    recommendation: str
    competency_scores: list[CompetencyScoreOut]
    flagged_limitations: list[str]
    turns_completed: int
    latency_compliance: float


class SessionSummary(BaseModel):
    session_id: str
    sanitized_name: str
    role: str
    status: str
    retrieval_backend: str
    turns: int
    created_at: int
    updated_at: int
    has_evaluation: bool
    #: Populated once close() has run; None for an active session. Lets the
    #: admin dashboard rank candidates without a per-row evaluation fetch.
    rubric_fit_index: float | None = None
    recommendation: str | None = None


class TranscriptTurnOut(BaseModel):
    speaker: str
    text: str
    offset_ms: float
    turnaround_ms: float | None = None


class LatencyStagesOut(BaseModel):
    stage: str
    budget_ms: float
    p95_ms: float | None = None


class SystemStatusResponse(BaseModel):
    environment: str
    offline: bool
    #: "An API key is set" -- NOT "that key works". See cognition_degraded.
    cognition_configured: bool
    #: True once a live cognition call has failed and been served by the
    #: offline fallback instead. A configured-but-degraded deployment answers
    #: every candidate with the same canned sentence, and without this it
    #: looks identical to a healthy one.
    cognition_degraded: bool = False
    cognition_fallbacks: int = 0
    moss_configured: bool
    qdrant_configured: bool
    transport_configured: bool
    telemetry_configured: bool
    stt_configured: bool
    session_state_backend: str
    database_url_scheme: str
    active_sessions: int
    latency_budget_ms: float
    stage_budgets: list[LatencyStagesOut]


__all__ = [
    "CompetencyScoreOut",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "ErrorResponse",
    "EvaluationResponse",
    "LatencyStagesOut",
    "SessionSummary",
    "SystemStatusResponse",
    "TranscriptTurnOut",
    "TurnRequest",
    "TurnResponse",
]

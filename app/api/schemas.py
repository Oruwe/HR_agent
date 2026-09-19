"""Wire shapes for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ImportRequest(BaseModel):
    """A batch of scraped candidate records.

    ``candidates`` is a list of arbitrary JSON objects -- whatever the scraper
    produced. No schema is imposed beyond "it's an object", because the shape
    belongs to the scraper and imposing one here would mean rejecting records
    the moment it changes.
    """

    candidates: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


class ImportResponse(BaseModel):
    imported: int
    redacted: int
    total_in_pool: int


class CandidateSummary(BaseModel):
    id: str
    name: str
    headline: str
    score: float | None = None
    recommendation: str | None = None
    rationale: str | None = None
    imported_at: int
    analyzed_at: int | None = None


class CandidateDetail(CandidateSummary):
    source: dict[str, Any]


class AnalyzeResponse(BaseModel):
    analyzed: int
    skipped: int
    offline: bool


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|model)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length alone lets "   " through, and a whitespace question costs
        # a model call to answer with nothing.
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value.strip()


class CandidateRef(BaseModel):
    """A candidate an answer was grounded in."""

    id: str
    name: str


class ChatResponse(BaseModel):
    reply: str
    #: How many records the analyst actually read -- not the pool size. An
    #: answer grounded in 6 of 800 is a different claim from one that saw
    #: everything, and the manager deserves to know which they are reading.
    candidates_considered: int
    pool_size: int = 0
    sources: list[CandidateRef] = Field(default_factory=list)
    retrieval_backend: str = ""
    retrieval_ms: float = 0.0


class StatusResponse(BaseModel):
    environment: str
    offline: bool
    #: "A key is set" -- NOT "that key works". See `degraded`.
    model_configured: bool
    model: str
    #: True once a live call has failed and been served by the offline mock.
    degraded: bool = False
    fallbacks: int = 0
    candidates: int = 0
    analyzed: int = 0

    # -- retrieval ------------------------------------------------------------
    #: "Moss credentials are set" -- NOT "Moss works". See retrieval_degraded.
    moss_configured: bool = False
    #: What actually answered the last question: Moss, or the local index.
    retrieval_backend: str = ""
    #: True once a configured Moss has failed and the local index took over.
    retrieval_degraded: bool = False
    retrieval_fallbacks: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


__all__ = [
    "AnalyzeResponse",
    "CandidateDetail",
    "CandidateRef",
    "CandidateSummary",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "ImportRequest",
    "ImportResponse",
    "StatusResponse",
]

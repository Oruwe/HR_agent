"""Runtime configuration, read once from the environment and frozen.

Every field has a default that works with no infrastructure: with no
credentials the app runs OFFLINE, backed by a deterministic mock analyst
instead of a live model, which is what the test suite exercises and why it
needs no network.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field


class PiiMode(StrEnum):
    """How hard the egress guard fails when it finds unredacted PII.

    ``strict`` raises. ``permissive`` scrubs in place and continues, which is
    appropriate for a staging rollout and never for production.
    """

    STRICT = "strict"
    PERMISSIVE = "permissive"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    environment: str = "development"
    log_level: str = "INFO"

    # -- the analyst model ----------------------------------------------------
    google_api_key: str = ""
    #: A floating alias, not a pinned version, and deliberately so: this app
    #: previously defaulted to "gemini-2.0-flash" until that model was retired
    #: out from under the deployment, at which point every call 400'd and the
    #: product silently served canned answers. "gemini-2.5-flash" is already
    #: gone the same way. Pin via HRTE_MODEL if you need frozen phrasing.
    model: str = "gemini-flash-latest"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    #: Generous, because a hiring rationale over a whole pool is a real answer,
    #: not a one-line interview question.
    max_output_tokens: int = Field(default=2048, gt=0)
    #: Thinking tokens are billed out of max_output_tokens, so a reasoning
    #: model can burn the entire allowance before emitting a visible
    #: character -- an empty answer. 0 disables it; None omits the field for a
    #: model that rejects it.
    thinking_budget: int | None = Field(default=0, ge=0)

    # -- retrieval (Moss) -----------------------------------------------------
    #: Moss is a semantic search runtime (https://usemoss.dev). It backs the
    #: analyst's retrieval: a manager's question is matched against the pool
    #: and only the relevant records are put in front of the model, instead of
    #: every record in the database.
    #:
    #: Both values are needed. With either missing the local index is used,
    #: which is lexical rather than semantic -- a real fallback, but a weaker
    #: one, and /api/status says which is in play.
    moss_project_id: str = ""
    moss_project_key: str = ""
    moss_index: str = "candidate_pool"
    #: How many candidate records a retrieved answer is allowed to read.
    retrieval_top_k: int = Field(default=12, gt=0)

<<<<<<< HEAD
    # -- retrieval: Qdrant (cold archive) ------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "engineering_talent_rubrics"

    # -- speech-to-text ---------------------------------------------------------
    deepgram_api_key: str = ""
    
    # -- ElevenLabs voice AI ----------------------------------------------------
    elevenlabs_api_key: str = ""
    elevenlabs_agent_id: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_stability: float = Field(default=0.5, ge=0.0, le=1.0)
    elevenlabs_similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    elevenlabs_style: float = Field(default=0.0, ge=0.0, le=1.0)
    elevenlabs_speaker_boost: bool = True

    # -- telemetry ------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_endpoint: str = ""
=======
    # -- storage / transport --------------------------------------------------
    database_url: str = ""
    cors_origins: str = "*"
>>>>>>> a12cd5831229344cfe567d2f98949cf2442622d7

    # -- security -------------------------------------------------------------
    pii_mode: PiiMode = PiiMode.STRICT

    @property
    def model_configured(self) -> bool:
        """True when a real model is available. False means the mock analyst."""
        return bool(self.google_api_key)

    @property
    def offline(self) -> bool:
        return not self.model_configured

    @property
<<<<<<< HEAD
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def telemetry_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def stt_configured(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def speculation_window_ms(self) -> float:
        """Wall time speculation buys us, in ms."""
        return self.endpoint_silence_ms - self.speculative_silence_ms


def load_settings() -> Settings:
    """Build :class:`Settings` from the process environment."""
    return Settings(
        environment=_env("HRTE_ENV", "development"),
        log_level=_env("HRTE_LOG_LEVEL", "INFO").upper(),
        latency_budget_ms=_env_float("HRTE_LATENCY_BUDGET_MS", TOTAL_TURNAROUND_BUDGET_MS),
        endpoint_silence_ms=_env_float("HRTE_ENDPOINT_SILENCE_MS", 120.0),
        speculative_silence_ms=_env_float("HRTE_SPECULATIVE_SILENCE_MS", 40.0),
        barge_in_ms=_env_float("HRTE_BARGE_IN_MS", 30.0),
        livekit_url=_env("LIVEKIT_URL"),
        livekit_api_key=_env("LIVEKIT_API_KEY"),
        livekit_api_secret=_env("LIVEKIT_API_SECRET"),
        room_prefix=_env("HRTE_ROOM_PREFIX", "screening"),
        google_api_key=_env("GOOGLE_API_KEY"),
        cognition_model=_env("HRTE_COGNITION_MODEL", "gemini-2.0-flash"),
        cognition_temperature=_env_float("HRTE_COGNITION_TEMPERATURE", 0.2),
        cognition_max_tokens=_env_int("HRTE_COGNITION_MAX_TOKENS", 48),
        fast_path=_env("HRTE_FAST_PATH", "true").lower() not in {"0", "false", "no"},
        speech_engine=SpeechEngine(_env("HRTE_SPEECH_ENGINE", "mock") or "mock"),
        speech_ws_url=_env("HRTE_SPEECH_WS_URL"),
        speech_api_key=_env("HRTE_SPEECH_API_KEY"),
        moss_project_id=_env("MOSS_PROJECT_ID"),
        moss_project_key=_env("MOSS_PROJECT_KEY"),
        moss_index=_env("HRTE_MOSS_INDEX", "engineering_talent_rubrics"),
        qdrant_url=_env("QDRANT_URL"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        qdrant_collection=_env("HRTE_QDRANT_COLLECTION", "engineering_talent_rubrics"),
        deepgram_api_key=_env("DEEPGRAM_API_KEY"),
        elevenlabs_api_key=_env("ELEVENLABS_API_KEY"),
        elevenlabs_agent_id=_env("ELEVENLABS_AGENT_ID"),
        elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID"),
        elevenlabs_stability=_env_float("ELEVENLABS_STABILITY", 0.5),
        elevenlabs_similarity_boost=_env_float("ELEVENLABS_SIMILARITY_BOOST", 0.75),
        elevenlabs_style=_env_float("ELEVENLABS_STYLE", 0.0),
        elevenlabs_speaker_boost=_env("ELEVENLABS_SPEAKER_BOOST", "true").lower() not in {"0", "false", "no"},
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        otel_endpoint=_env("OTEL_EXPORTER_OTLP_ENDPOINT"),
        pii_mode=PiiMode(_env("HRTE_PII_MODE", "strict") or "strict"),
        database_url=_env("DATABASE_URL"),
        redis_url=_env("REDIS_URL"),
        cors_origins=_env("HRTE_CORS_ORIGINS", "*"),
        session_ttl_seconds=_env_int("HRTE_SESSION_TTL_SECONDS", 3600),
    )
=======
    def moss_configured(self) -> bool:
        """True when Moss *can* be reached. Not a claim that it works --
        see the retrieval backend reported by /api/status for that."""
        return bool(self.moss_project_id and self.moss_project_key)
>>>>>>> a12cd5831229344cfe567d2f98949cf2442622d7


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        environment=_env("HRTE_ENV", "development"),
        log_level=_env("HRTE_LOG_LEVEL", "INFO").upper(),
        google_api_key=_env("GOOGLE_API_KEY"),
        model=_env("HRTE_MODEL", "gemini-flash-latest"),
        temperature=_env_float("HRTE_TEMPERATURE", 0.4),
        max_output_tokens=_env_int("HRTE_MAX_OUTPUT_TOKENS", 2048),
        thinking_budget=_env_int("HRTE_THINKING_BUDGET", 0),
        moss_project_id=_env("MOSS_PROJECT_ID"),
        moss_project_key=_env("MOSS_PROJECT_KEY"),
        moss_index=_env("HRTE_MOSS_INDEX", "candidate_pool"),
        retrieval_top_k=_env_int("HRTE_RETRIEVAL_TOP_K", 12),
        database_url=_env("DATABASE_URL"),
        cors_origins=_env("HRTE_CORS_ORIGINS", "*"),
        pii_mode=PiiMode(_env("HRTE_PII_MODE", "strict") or "strict"),
    )


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment."""
    get_settings.cache_clear()


__all__ = ["PiiMode", "Settings", "get_settings", "reset_settings"]

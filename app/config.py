"""Runtime configuration and the latency budget that governs the whole pipeline.

Design notes
------------
Configuration is read from the process environment exactly once, at import of
:func:`get_settings`, and is then frozen. Nothing in the hot path may touch
``os.environ`` -- an environment lookup inside a 20ms audio frame callback is a
syscall-shaped foot-gun, and the budget below leaves no room for it.

Every field has a default that works with zero infrastructure. With no
credentials present the agent runs in OFFLINE mode: deterministic local
embeddings, an in-process vector index, mock cognition and mock synthesis. That
is what CI exercises, and it is why the test suite needs no network.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Latency budget
# =============================================================================

#: Hard ceiling, in milliseconds, for a conversational turn measured as
#: "candidate stops speaking" -> "first byte of agent audio handed to the
#: egress transport". Exceeding this is a defect, not a slow day.
TOTAL_TURNAROUND_BUDGET_MS: Final[float] = 150.0


class Stage(StrEnum):
    """The seven stages a turn passes through, in execution order.

    The names are also the span names emitted to Langfuse/OTel, so renaming one
    is a dashboard-breaking change.
    """

    TRANSPORT_INGRESS = "transport_ingress"
    VAD_ENDPOINT = "vad_endpoint"
    AUDIO_INGESTION = "audio_ingestion"
    VECTOR_MATCH = "vector_match"
    COGNITION_TTFT = "cognition_ttft"
    SPEECH_SYNTHESIS = "speech_synthesis"
    TRANSPORT_EGRESS = "transport_egress"


#: Per-stage ceilings. These sum to exactly TOTAL_TURNAROUND_BUDGET_MS, so a
#: stage that overruns has provably stolen headroom from a downstream stage.
STAGE_BUDGETS_MS: Final[dict[Stage, float]] = {
    Stage.TRANSPORT_INGRESS: 15.0,  # LiveKit WebRTC PeerConnection, edge -> SFU -> worker
    Stage.VAD_ENDPOINT: 20.0,  # Silero VAD v5 endpoint decision (ONNX, CPU)
    Stage.AUDIO_INGESTION: 10.0,  # dual-track 20ms ring buffer assembly
    Stage.VECTOR_MATCH: 5.0,  # in-memory rubric lookup
    Stage.COGNITION_TTFT: 45.0,  # streaming LLM time-to-first-token
    Stage.SPEECH_SYNTHESIS: 35.0,  # speech-to-speech first audio frame
    Stage.TRANSPORT_EGRESS: 20.0,  # jitter buffer + WebRTC publish
}

#: Stages whose cost is paid *concurrently with the candidate still speaking*
#: (prefetch/speculation), and therefore do not consume post-endpoint wall time.
#: Kept explicit so the scheduler and the benchmark agree on what "free" means.
PREFETCHABLE_STAGES: Final[frozenset[Stage]] = frozenset(
    {Stage.AUDIO_INGESTION, Stage.VECTOR_MATCH}
)

#: Audio format on the wire. 48kHz mono PCM16 in 20ms frames == 960 samples.
SAMPLE_RATE_HZ: Final[int] = 48_000
FRAME_DURATION_MS: Final[int] = 20
SAMPLES_PER_FRAME: Final[int] = SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000  # 960
BYTES_PER_SAMPLE: Final[int] = 2
CHANNELS: Final[int] = 1

#: Dense embedding width. Matches Gemini text-embedding output truncation.
EMBEDDING_DIM: Final[int] = 768


def assert_budget_is_coherent() -> None:
    """Fail loudly at import time if the stage budgets stop summing to the total.

    This is a real guard, not a formality: the budget table is the single place
    where an engineer is tempted to 'just add 5ms' during a deadline.
    """
    total = sum(STAGE_BUDGETS_MS.values())
    if abs(total - TOTAL_TURNAROUND_BUDGET_MS) > 1e-9:
        raise ValueError(
            f"Stage budgets sum to {total}ms but the total turnaround budget is "
            f"{TOTAL_TURNAROUND_BUDGET_MS}ms. Rebalance STAGE_BUDGETS_MS rather "
            f"than widening the contract."
        )


assert_budget_is_coherent()


# =============================================================================
# Settings
# =============================================================================


class PiiMode(StrEnum):
    STRICT = "strict"
    PERMISSIVE = "permissive"


class SpeechEngine(StrEnum):
    MOCK = "mock"
    MOSS = "moss"
    LIVEKIT = "livekit"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    return int(_env_float(key, float(default)))


class Settings(BaseModel):
    """Immutable, fully-defaulted runtime configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = "development"
    log_level: str = "INFO"

    # -- latency governance ---------------------------------------------------
    latency_budget_ms: float = Field(default=TOTAL_TURNAROUND_BUDGET_MS, gt=0)
    endpoint_silence_ms: float = Field(default=120.0, gt=0)
    speculative_silence_ms: float = Field(default=40.0, gt=0)
    barge_in_ms: float = Field(default=30.0, gt=0)

    # -- transport ------------------------------------------------------------
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    room_prefix: str = "screening"

    # -- cognition ------------------------------------------------------------
    google_api_key: str = ""
    #: A floating alias, not a pinned version, and deliberately so. This
    #: defaulted to "gemini-2.0-flash" until that model was retired out from
    #: under the deployment: every call 400'd, the interview fell back to the
    #: canned offline answer, and (before the degradation was made visible)
    #: nothing said so. "gemini-2.5-flash" is already gone the same way --
    #: the API now answers 404 "no longer available to new users" for it.
    #: A pin buys reproducibility only for as long as the pin exists, and
    #: nothing here needs it: role routing and competency scoring are
    #: deterministic in code, so a model rotation can change how a question
    #: is phrased but never what is asked or how it is scored. Pin a specific
    #: version via HRTE_COGNITION_MODEL if you want to freeze phrasing.
    cognition_model: str = "gemini-flash-latest"
    #: 0.7 rather than a near-zero value: the *decisions* here (which
    #: competency to probe, how to score one) are deterministic and made in
    #: code, never by the model (see app/agent/interview_flow.py) -- so the
    #: model has room to vary its phrasing turn to turn without touching
    #: anything that has to be reproducible or auditable.
    cognition_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    #: 150 (what .env.example documented, against a code default of 48) still
    #: truncated mid-sentence against the live API -- "Which specific NCCL
    #: environment variables" and then nothing. 250 leaves the model room to
    #: finish its own sentence; it stops naturally well before the cap, so
    #: this is a ceiling, not a target.
    cognition_max_tokens: int = Field(default=250, gt=0)
    #: Thinking tokens are billed out of max_output_tokens, so a reasoning
    #: model can burn the whole allowance before emitting a single visible
    #: character. 0 disables it, which is right for a real-time interview:
    #: the turn is one short spoken question. None omits the setting
    #: entirely, for a model that rejects it.
    cognition_thinking_budget: int | None = Field(default=0, ge=0)
    fast_path: bool = True

    # -- synthesis ------------------------------------------------------------
    # NOTE: these configure MOSS-Speech, the open speech-to-speech model. It is
    # unrelated to Moss (YC F25), the retrieval runtime configured below. The
    # two share a name and nothing else.
    speech_engine: SpeechEngine = SpeechEngine.MOCK
    speech_ws_url: str = ""
    speech_api_key: str = ""

    # -- retrieval: Moss (hot path) ------------------------------------------
    moss_project_id: str = ""
    moss_project_key: str = ""
    moss_index: str = "engineering_talent_rubrics"

    # -- retrieval: Qdrant (cold archive) ------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "engineering_talent_rubrics"

    # -- speech-to-text ---------------------------------------------------------
    deepgram_api_key: str = ""

    # -- telemetry ------------------------------------------------------------
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_endpoint: str = ""

    # -- security -------------------------------------------------------------
    pii_mode: PiiMode = PiiMode.STRICT

    # -- API / persistence layer -----------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    cors_origins: str = "*"
    session_ttl_seconds: int = Field(default=3600, gt=0)

    @field_validator("speculative_silence_ms")
    @classmethod
    def _speculation_precedes_commit(cls, v: float, info) -> float:
        commit = info.data.get("endpoint_silence_ms", 250.0)
        if v >= commit:
            raise ValueError(
                "speculative_silence_ms must be strictly less than "
                "endpoint_silence_ms, otherwise speculation buys no latency."
            )
        return v

    # -- derived capability flags --------------------------------------------

    @property
    def offline(self) -> bool:
        """True when no external service is configured; CI's normal state."""
        return not (
            self.google_api_key or self.livekit_url or self.qdrant_url or self.moss_configured
        )

    @property
    def moss_configured(self) -> bool:
        """True when a Moss project is available for hot-path retrieval."""
        return bool(self.moss_project_id and self.moss_project_key)

    @property
    def cognition_configured(self) -> bool:
        return bool(self.google_api_key)

    @property
    def transport_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
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
        cognition_model=_env("HRTE_COGNITION_MODEL", "gemini-flash-latest"),
        cognition_temperature=_env_float("HRTE_COGNITION_TEMPERATURE", 0.7),
        cognition_max_tokens=_env_int("HRTE_COGNITION_MAX_TOKENS", 250),
        cognition_thinking_budget=_env_int("HRTE_COGNITION_THINKING_BUDGET", 0),
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call :func:`reset_settings` in tests."""
    return load_settings()


def reset_settings() -> None:
    """Drop the cached settings so the next :func:`get_settings` re-reads env."""
    get_settings.cache_clear()


__all__ = [
    "BYTES_PER_SAMPLE",
    "CHANNELS",
    "EMBEDDING_DIM",
    "FRAME_DURATION_MS",
    "PREFETCHABLE_STAGES",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE_HZ",
    "STAGE_BUDGETS_MS",
    "TOTAL_TURNAROUND_BUDGET_MS",
    "PiiMode",
    "Settings",
    "SpeechEngine",
    "Stage",
    "assert_budget_is_coherent",
    "get_settings",
    "load_settings",
    "reset_settings",
]

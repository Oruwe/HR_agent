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

    # -- storage / transport --------------------------------------------------
    database_url: str = ""
    cors_origins: str = "*"

    # -- security -------------------------------------------------------------
    pii_mode: PiiMode = PiiMode.STRICT

    @property
    def model_configured(self) -> bool:
        """True when a real model is available. False means the mock analyst."""
        return bool(self.google_api_key)

    @property
    def offline(self) -> bool:
        return not self.model_configured


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
        database_url=_env("DATABASE_URL"),
        cors_origins=_env("HRTE_CORS_ORIGINS", "*"),
        pii_mode=PiiMode(_env("HRTE_PII_MODE", "strict") or "strict"),
    )


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment."""
    get_settings.cache_clear()


__all__ = ["PiiMode", "Settings", "get_settings", "reset_settings"]

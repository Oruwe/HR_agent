"""What GeminiCognition actually sends to the provider.

These exist because a live deployment answered every candidate with the same
canned sentence for hours, and two separate config mistakes were capable of
causing it -- both invisible from inside the app:

* a model name that no longer exists (every call 4xx'd, silently falling back
  to the offline mock), and
* thinking tokens billed out of ``max_output_tokens``, which let the model
  spend its whole allowance reasoning and return an empty string with
  ``finishReason: MAX_TOKENS``.

Neither is caught by any test that mocks the provider at a higher level, so
the request payload itself is asserted here.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.cognition import GeminiCognition, Message
from app.config import Settings
from tests.conftest import run_async


class _CapturingClient:
    """Stands in for google.genai.Client, recording the request config."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any):
        self.captured = {"model": model, "contents": contents, "config": config}

        async def _empty():
            return
            yield  # pragma: no cover - makes this an async generator

        return _empty()


def _capture(settings: Settings) -> dict[str, Any]:
    provider = GeminiCognition(settings)
    client = _CapturingClient()
    provider._client = client

    async def go() -> None:
        async for _ in provider.stream("system", [Message(role="user", content="hi")]):
            pass

    run_async(go)
    return client.captured


@pytest.fixture
def configured_settings() -> Settings:
    return Settings(google_api_key="test-key-not-real")


def test_thinking_is_disabled_by_default(configured_settings: Settings) -> None:
    """Left on, a reasoning model returns an empty answer at this budget."""
    config = _capture(configured_settings)["config"]
    assert config["thinking_config"] == {"thinking_budget": 0}


def test_thinking_budget_is_configurable(configured_settings: Settings) -> None:
    settings = configured_settings.model_copy(update={"cognition_thinking_budget": 512})
    config = _capture(settings)["config"]
    assert config["thinking_config"] == {"thinking_budget": 512}


def test_thinking_config_is_omitted_when_unset(configured_settings: Settings) -> None:
    """None means "don't send the field at all", for a model that rejects it."""
    settings = configured_settings.model_copy(update={"cognition_thinking_budget": None})
    assert "thinking_config" not in _capture(settings)["config"]


def test_token_ceiling_leaves_room_to_finish_a_sentence(
    configured_settings: Settings,
) -> None:
    """150 truncated mid-sentence against the live API; 48 was worse."""
    config = _capture(configured_settings)["config"]
    assert config["max_output_tokens"] >= 250


def test_default_model_is_not_a_known_retired_pin(configured_settings: Settings) -> None:
    """gemini-2.0-flash and gemini-2.5-flash are both gone; a pin that dies
    takes the whole interviewer down with it."""
    model = _capture(configured_settings)["model"]
    assert model not in {"gemini-2.0-flash", "gemini-2.5-flash"}

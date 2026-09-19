"""Streaming model interface, with an offline mock that actually works.

Two providers behind one protocol: Gemini for real deployments, and a
deterministic mock for CI, local development and any deployment with no
credentials. The mock is not a stub that returns an apology -- it produces
usable rankings and answers, so the whole product is demonstrable with zero
infrastructure and the test suite needs no network.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.security.pii_scrubber import scrub_text
from app.security.token_guard import redact_credentials

logger = logging.getLogger(__name__)

#: How many times a live model call has failed and been served by the mock
#: instead. `model_configured` only means "a key is set", not "that key
#: works": a deployment whose every call 4xx's looks identical, from the
#: outside, to a healthy one. It should be impossible to miss, so this is
#: surfaced through /api/status and the dashboard.
_fallbacks = 0


def record_fallback() -> None:
    global _fallbacks
    _fallbacks += 1


def fallback_count() -> int:
    return _fallbacks


def reset_fallbacks() -> None:
    """Test isolation only."""
    global _fallbacks
    _fallbacks = 0


class ChunkType(StrEnum):
    TEXT = "text"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class CognitionChunk:
    type: ChunkType
    text: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    role: str  # "user" | "model"
    content: str


@runtime_checkable
class CognitionProvider(Protocol):
    def stream(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> AsyncIterator[CognitionChunk]: ...


_CANDIDATE_RE = re.compile(r"### Candidate (\w+) \(([^)]*)\)")

#: The separator rendered by app.agent.analyst.render_candidate.
_RECORD_MARKER = "### Candidate "


@dataclass
class MockCognition:
    """Deterministic provider. No network, no credentials, still useful.

    For a ranking request it returns well-formed JSON covering exactly the
    candidates it was shown, so an offline deployment ranks its pool instead
    of failing. For a question it answers from the same records. The scoring
    is a transparent heuristic -- record richness -- and says so, because a
    number that looks like judgement but isn't is worse than no number.
    """

    responses: list[str] = field(default_factory=list)
    calls: int = field(default=0, init=False)

    def _scripted(self) -> str | None:
        if self.calls <= len(self.responses):
            return self.responses[self.calls - 1]
        return None

    @staticmethod
    def _seen_candidates(text: str) -> list[tuple[str, str]]:
        return _CANDIDATE_RE.findall(text)

    @staticmethod
    def _blocks(corpus: str) -> list[tuple[str, str, str]]:
        """Split the rendered pool into (handle, name, that candidate's text).

        Splitting on the record separator rather than slicing a fixed window
        from each match: a window runs into the *next* candidate's record, so
        every candidate looks equally detailed and the whole pool scores the
        same.
        """
        parts = corpus.split(_RECORD_MARKER)
        out: list[tuple[str, str, str]] = []
        for part in parts[1:]:
            header, _, body = part.partition("\n")
            handle, _, rest = header.partition(" ")
            name = rest.strip().strip("()")
            out.append((handle.strip(), name, body))
        return out

    def _rank_payload(self, corpus: str) -> str:
        rankings = []
        for handle, name, block in self._blocks(corpus):
            # Richness of the record as a stand-in for fit. Honest, cheap, and
            # explicitly labelled as offline so nobody mistakes it for judgement.
            richness = min(1.0, len(block.split()) / 220.0)
            verdict = "INTERVIEW" if richness > 0.66 else "MAYBE" if richness > 0.33 else "PASS"
            rankings.append(
                {
                    "id": handle,
                    "score": round(richness, 3),
                    "verdict": verdict,
                    "rationale": (
                        f"Offline analyst: no model is configured, so {name} is scored "
                        f"by how much detail the scraped record carries, not by fit. "
                        f"Set GOOGLE_API_KEY for a real assessment."
                    ),
                }
            )
        return json.dumps({"rankings": rankings})

    async def stream(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> AsyncIterator[CognitionChunk]:
        self.calls += 1
        scripted = self._scripted()
        corpus = "\n".join(m.content for m in messages) + "\n" + system_prompt

        if scripted is not None:
            text = scripted
        elif '"rankings"' in system_prompt:
            text = self._rank_payload(corpus)
        else:
            names = [name for _, name, _ in self._blocks(corpus)]
            roster = ", ".join(names[:8]) if names else "no candidates yet"
            text = (
                "No model is configured, so I can't analyse the pool for real. "
                f"I can see {len(names)} candidate(s) imported: {roster}. "
                "Set GOOGLE_API_KEY to get genuine answers here."
            )

        for word in text.split(" "):
            yield CognitionChunk(type=ChunkType.TEXT, text=word + " ")
        yield CognitionChunk(type=ChunkType.DONE)


@dataclass
class GeminiCognition:
    """Google GenAI streaming provider, degrading to the mock on failure."""

    settings: Settings
    fallback: MockCognition = field(default_factory=MockCognition)
    _client: Any | None = field(default=None, init=False)

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - requires the optional dependency
            from google import genai  # type: ignore[import-not-found]

            self._client = genai.Client(api_key=self.settings.google_api_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini client unavailable (%s: %s)", type(exc).__name__, exc)
            self._client = None
        return self._client

    async def stream(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> AsyncIterator[CognitionChunk]:
        client = self._ensure_client() if self.settings.model_configured else None
        if client is None:
            record_fallback()
            async for chunk in self.fallback.stream(system_prompt, messages):
                yield chunk
            return

        # Last gate before the payload leaves the process.
        safe_system = scrub_text(redact_credentials(system_prompt))
        contents = [{"role": m.role, "parts": [{"text": scrub_text(m.content)}]} for m in messages]

        config: dict[str, Any] = {
            "system_instruction": safe_system,
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        # Thinking tokens come out of max_output_tokens, so a reasoning model
        # can spend the whole allowance thinking and return an EMPTY string
        # with finishReason MAX_TOKENS. Verified against the live API.
        if self.settings.thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": self.settings.thinking_budget}

        try:  # pragma: no cover - network path
            stream = await client.aio.models.generate_content_stream(
                model=self.settings.model, contents=contents, config=config
            )
            async for event in stream:
                for candidate in getattr(event, "candidates", None) or []:
                    for part in getattr(candidate.content, "parts", None) or []:
                        if getattr(part, "text", None):
                            yield CognitionChunk(type=ChunkType.TEXT, text=part.text)
            yield CognitionChunk(type=ChunkType.DONE)
        except Exception as exc:  # pragma: no cover - degradation path
            record_fallback()
            # The message, not just the class name: "ClientError" alone cannot
            # distinguish a bad key from a retired model name, and both have
            # happened here. Redacted, because provider errors sometimes echo
            # the request URL with the key in it.
            logger.warning(
                "Gemini call failed (%s: %s); answering from the offline mock instead.",
                type(exc).__name__,
                redact_credentials(str(exc))[:500],
            )
            async for chunk in self.fallback.stream(system_prompt, messages):
                yield chunk


def build_cognition(settings: Settings | None = None) -> CognitionProvider:
    settings = settings or get_settings()
    if settings.model_configured:
        return GeminiCognition(settings)
    return MockCognition()


__all__ = [
    "ChunkType",
    "CognitionChunk",
    "CognitionProvider",
    "GeminiCognition",
    "Message",
    "MockCognition",
    "build_cognition",
    "fallback_count",
    "record_fallback",
    "reset_fallbacks",
]

"""Streaming LLM interface with tool calling and mid-generation abort.

Everything here is built around two constraints the rest of the system imposes:

* **Streaming is mandatory, not an optimisation.** The budget allows 45ms for
  time-to-first-token. A non-streaming call cannot participate at all, because
  its first byte arrives only after its last one.
* **Abort must be immediate.** When a candidate interrupts, the generation they
  interrupted is worthless. Continuing to stream it wastes tokens and, worse,
  risks the tail of an abandoned answer being spoken over their new question.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.security.pii_scrubber import scrub_text
from app.security.token_guard import redact_credentials

logger = logging.getLogger(__name__)

#: How many times a real cognition call has failed and silently handed the
#: turn to the offline mock. This exists because `cognition_configured` only
#: means "an API key is set", not "that key works" -- without a counter, an
#: interview where *every* answer came from the canned fallback is
#: indistinguishable, in the admin dashboard and in the API, from a healthy
#: one. A deployment answering every candidate with the same hardcoded
#: sentence should be impossible to miss.
_cognition_fallbacks = 0


def record_cognition_fallback() -> None:
    global _cognition_fallbacks
    _cognition_fallbacks += 1


def cognition_fallback_count() -> int:
    """Number of live cognition calls that fell back since process start."""
    return _cognition_fallbacks


def reset_cognition_fallbacks() -> None:
    """Test isolation only."""
    global _cognition_fallbacks
    _cognition_fallbacks = 0


class ChunkType(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CognitionChunk:
    type: ChunkType
    text: str = ""
    tool_call: ToolCall | None = None


@dataclass(frozen=True, slots=True)
class Message:
    role: str  # "user" | "model"
    content: str


#: The tool surface exposed to the interviewer model. Deliberately tiny: every
#: additional tool is another thing the model can spend a turn deciding about,
#: and this one has 45ms.
TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "record_candidate_competency",
        "description": (
            "Record an evidence-backed score for one rubric competency after "
            "the candidate has demonstrated or failed to demonstrate it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Rubric competency key."},
                "score": {
                    "type": "number",
                    "description": "Demonstrated proficiency from 0.0 to 1.0.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence citing what the candidate actually said.",
                },
            },
            "required": ["skill", "score", "rationale"],
        },
    },
    {
        "name": "trigger_role_transition",
        "description": (
            "Switch the active rubric when the candidate's evidence clearly "
            "belongs to a different engineering track."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "next_scenario": {"type": "string", "description": "Target role key."},
                "reason": {"type": "string"},
            },
            "required": ["next_scenario"],
        },
    },
    {
        "name": "terminate_screening_session",
        "description": (
            "End the screening. Use when the rubric is covered, the candidate "
            "withdraws, or continuing would serve no evaluative purpose."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
)


@runtime_checkable
class CognitionProvider(Protocol):
    """Streaming text generation with tool calls."""

    async def stream(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> AsyncIterator[CognitionChunk]: ...

    def abort(self) -> None: ...

    @property
    def aborted(self) -> bool: ...


@dataclass
class MockCognition:
    """Deterministic provider for CI, benchmarks and the offline demo.

    Responses are drawn from a scripted queue rather than generated, which is
    what makes latency tests meaningful: with a real model the measurement is
    dominated by network variance and tells you nothing about whether *your*
    pipeline regressed. ``ttft_ms`` models the provider's time-to-first-token so
    the harness still measures a realistic total.
    """

    ttft_ms: float = 30.0
    inter_token_ms: float = 0.4
    responses: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    default_response: str = (
        "Understood. Walk me through the trade-off you made there, and what "
        "the numbers looked like afterwards."
    )
    calls: int = field(default=0, init=False)
    _aborted: bool = field(default=False, init=False)

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True

    async def stream(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> AsyncIterator[CognitionChunk]:
        self._aborted = False
        self.calls += 1

        text = self.responses.pop(0) if self.responses else self.default_response
        await asyncio.sleep(self.ttft_ms / 1000.0)

        for word in text.split(" "):
            if self._aborted:
                return
            yield CognitionChunk(type=ChunkType.TEXT, text=word + " ")
            if self.inter_token_ms:
                await asyncio.sleep(self.inter_token_ms / 1000.0)

        if self.tool_calls and not self._aborted:
            yield CognitionChunk(type=ChunkType.TOOL_CALL, tool_call=self.tool_calls.pop(0))
        if not self._aborted:
            yield CognitionChunk(type=ChunkType.DONE)


class GeminiCognition:
    """Google GenAI streaming provider.

    Two settings do the heavy lifting and are not arbitrary:

    * ``temperature=0.2`` -- an interviewer that rephrases the same probe
      differently on every run cannot be compared across candidates, which is
      the whole point of a rubric.
    * ``max_output_tokens=150`` -- this is a *latency* control as much as a
      style one. Spoken responses longer than a couple of sentences are both
      unnatural in conversation and expensive to synthesise.

    Any failure degrades to :class:`MockCognition` rather than dropping the
    call. A candidate mid-interview should never be abandoned because a
    provider returned a 503.
    """

    def __init__(
        self, settings: Settings | None = None, fallback: CognitionProvider | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._fallback = fallback or MockCognition()
        self._aborted = False
        self._client: Any | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True
        self._fallback.abort()

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - requires the optional dependency
            from google import genai  # type: ignore[import-not-found]

            self._client = genai.Client(api_key=self.settings.google_api_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini client unavailable (%s)", type(exc).__name__)
            self._client = None
        return self._client

    async def stream(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> AsyncIterator[CognitionChunk]:
        self._aborted = False
        client = self._ensure_client() if self.settings.cognition_configured else None
        if client is None:
            async for chunk in self._fallback.stream(system_prompt, messages, tools):
                yield chunk
            return

        # Last gate before the payload leaves the process. Transcripts are
        # already scrubbed upstream, but concatenation is exactly where a
        # redacted history gets accidentally rebuilt from a raw source.
        safe_system = scrub_text(redact_credentials(system_prompt))
        contents = [{"role": m.role, "parts": [{"text": scrub_text(m.content)}]} for m in messages]

        try:  # pragma: no cover - network path
            config: dict[str, Any] = {
                "system_instruction": safe_system,
                "temperature": self.settings.cognition_temperature,
                "max_output_tokens": self.settings.cognition_max_tokens,
            }
            if tools:
                config["tools"] = [{"function_declarations": list(tools)}]

            stream = await client.aio.models.generate_content_stream(
                model=self.settings.cognition_model, contents=contents, config=config
            )
            async for event in stream:
                if self._aborted:
                    return
                for candidate in getattr(event, "candidates", None) or []:
                    for part in getattr(candidate.content, "parts", None) or []:
                        call = getattr(part, "function_call", None)
                        if call is not None:
                            yield CognitionChunk(
                                type=ChunkType.TOOL_CALL,
                                tool_call=ToolCall(name=call.name, arguments=dict(call.args or {})),
                            )
                        elif getattr(part, "text", None):
                            yield CognitionChunk(type=ChunkType.TEXT, text=part.text)
            if not self._aborted:
                yield CognitionChunk(type=ChunkType.DONE)
        except Exception as exc:  # pragma: no cover - degradation path
            record_cognition_fallback()
            # The exception *message* is the difference between "your key is
            # wrong" and "that model name doesn't exist" -- logging only the
            # class name (as this did) makes a total cognition outage look
            # identical to a transient blip, and leaves an operator with
            # nothing to act on. Redacted because provider errors sometimes
            # echo the request URL, key and all.
            logger.warning(
                "Gemini stream failed (%s: %s); continuing the interview on the "
                "fallback provider. Every answer from here is the canned offline "
                "response, not a real one.",
                type(exc).__name__,
                redact_credentials(str(exc))[:500],
            )
            async for chunk in self._fallback.stream(system_prompt, messages, tools):
                yield chunk


def build_cognition(settings: Settings | None = None) -> CognitionProvider:
    settings = settings or get_settings()
    if settings.cognition_configured:
        return GeminiCognition(settings)
    return MockCognition()


__all__ = [
    "TOOL_SCHEMAS",
    "ChunkType",
    "CognitionChunk",
    "CognitionProvider",
    "GeminiCognition",
    "Message",
    "MockCognition",
    "ToolCall",
    "build_cognition",
    "cognition_fallback_count",
    "record_cognition_fallback",
    "reset_cognition_fallbacks",
]

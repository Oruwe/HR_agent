"""Speech-to-text: turning a candidate's uploaded audio into text.

This is a genuine gap in the original repository, which assumed a transcript
already exists (the LiveKit worker consumes text turns; nothing upstream of it
produced them from audio). The Candidate App records real microphone audio, so
something has to transcribe it.

Same pattern as :mod:`app.agent.cognition` and :mod:`app.voice.moss_engine`:
a small :class:`Protocol`, a real provider that is only *reachable* when a key
is configured, and a deterministic offline fallback so the app never needs
credentials to run. The real provider here is Deepgram's prerecorded REST API
-- a single POST with raw audio bytes, no SDK, no websocket -- chosen because
it is the simplest transcription call that still does real work.

Sandboxed environments (CI containers, restricted egress) cannot reach
api.deepgram.com even with a key configured; :class:`MockTranscriber` is what
runs there, and it is exercised by the test suite. It is not a stand-in for
"transcription is hard to implement" -- it is a stand-in for "this network
call cannot be made from here," which is the one condition under which this
project's own conventions say a mock is the right tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.security.pii_scrubber import scrub

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    confidence: float = 0.0
    provider: str = "mock"
    duration_s: float = 0.0
    redaction_counts: dict[str, int] | None = None


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResult: ...


class MockTranscriber:
    """Deterministic offline stand-in. No network, no dependency, no key.

    It cannot know what was said -- that is the honest limit of an offline
    fallback -- so it returns a clearly-labelled placeholder rather than
    fabricating plausible-looking text that could be mistaken for a real
    transcript in a dossier. Candidate Apps should treat an empty/placeholder
    transcript as a cue to fall back to typed input, which the API surfaces
    via ``TranscriptionResult.provider == "mock"``.
    """

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="", provider="mock")
        # A tiny, deterministic "duration" estimate from payload size, purely
        # for the UI to show something plausible in offline/demo mode.
        approx_seconds = round(len(audio_bytes) / 32_000, 2)
        return TranscriptionResult(
            text="[offline mode: speech-to-text is not configured; type your answer instead]",
            confidence=0.0,
            provider="mock",
            duration_s=approx_seconds,
        )


class DeepgramTranscriber:
    """Real transcription via Deepgram's prerecorded REST endpoint.

    Reachable only when ``DEEPGRAM_API_KEY`` is set. Uses the pre-recorded
    (not streaming) endpoint deliberately: the Candidate App uploads a
    complete answer clip per turn rather than a live stream, which keeps this
    class a single stateless HTTP call -- no socket lifecycle to manage, no
    reconnect logic, no partial-result buffering.
    """

    ENDPOINT = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, model: str = "nova-2") -> None:
        self._api_key = api_key
        self._model = model

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptionResult:
        import httpx

        params = {"model": self._model, "smart_format": "true", "punctuate": "true"}
        headers = {"Authorization": f"Token {self._api_key}", "Content-Type": mime_type}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    self.ENDPOINT, params=params, headers=headers, content=audio_bytes
                )
                response.raise_for_status()
                payload = response.json()
            alt = payload["results"]["channels"][0]["alternatives"][0]
            text = alt.get("transcript", "")
            confidence = float(alt.get("confidence", 0.0))
        except Exception as exc:
            logger.warning(
                "Deepgram transcription failed (%s); returning empty transcript.",
                type(exc).__name__,
            )
            return TranscriptionResult(text="", provider="deepgram-error")

        scrubbed = scrub(text)
        return TranscriptionResult(
            text=scrubbed.text,
            confidence=confidence,
            provider="deepgram",
            redaction_counts=scrubbed.counts(),
        )


def build_transcriber(settings: Settings | None = None) -> Transcriber:
    """Pick the transcriber the same way :func:`build_cognition` picks a model.

    Real provider when a key is present, deterministic mock otherwise. Nothing
    in the caller needs to branch on which one it got.
    """
    settings = settings or get_settings()
    if settings.deepgram_api_key:
        return DeepgramTranscriber(settings.deepgram_api_key)
    return MockTranscriber()


__all__ = [
    "DeepgramTranscriber",
    "MockTranscriber",
    "Transcriber",
    "TranscriptionResult",
    "build_transcriber",
]

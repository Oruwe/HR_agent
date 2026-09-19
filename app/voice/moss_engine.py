"""Streaming speech synthesis with abort support.

Naming caution: the ``MOSS`` here is **MOSS-Speech**, the open speech-to-speech
model. It has nothing to do with Moss (YC F25), the retrieval runtime in
:mod:`app.storage.retrieval`. Configuration for this module lives under
``HRTE_SPEECH_*``; the retrieval runtime uses ``MOSS_PROJECT_*``.

The rule that governs this module: **never wait for a complete sentence you
could have started speaking.** A synthesiser that buffers the LLM's full
response before emitting audio adds the entire generation time to perceived
latency, which would blow the budget several times over no matter how fast the
model is. So text is chunked at the first clause boundary that yields something
speakable, and audio starts flowing while the model is still writing.

The second rule: **every stream must be abortable mid-frame.** Barge-in is
worthless if the synthesiser keeps producing audio for 400ms after the
candidate starts talking. Abort is cooperative and epoch-tagged, matching
:mod:`app.voice.ring_buffer`, so a frame generated before the abort cannot be
published after it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

import numpy as np

from app.config import FRAME_DURATION_MS, SAMPLE_RATE_HZ, Settings, get_settings
from app.voice.ring_buffer import AudioFrame, synth_tone

logger = logging.getLogger(__name__)

#: Boundaries at which a partial response becomes speakable. Sentence-final
#: punctuation first, then clause punctuation -- a comma is enough to start
#: talking and buys roughly a sentence of generation time.
_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|(?<=[,;:])\s+")

#: Do not emit a chunk shorter than this; sub-word fragments make the voice
#: stutter and the round trip is not worth it.
MIN_CHUNK_CHARS: Final[int] = 12


async def sentence_chunks(
    tokens: AsyncIterator[str], min_chars: int = MIN_CHUNK_CHARS
) -> AsyncIterator[str]:
    """Regroup a token stream into the earliest speakable chunks.

    The first chunk is deliberately allowed to be short: getting *some* audio
    out fast is what the candidate perceives as responsiveness, and by the time
    it has been spoken the model is comfortably ahead.
    """
    buffer = ""
    first = True
    async for token in tokens:
        buffer += token
        while True:
            match = _BOUNDARY_RE.search(buffer)
            if not match:
                break
            head, tail = buffer[: match.end()].strip(), buffer[match.end() :]
            threshold = 1 if first else min_chars
            if len(head) < threshold:
                break
            buffer = tail
            first = False
            yield head
    if buffer.strip():
        yield buffer.strip()


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Streaming text-to-audio with cooperative abort."""

    async def stream(
        self, text_chunks: AsyncIterator[str], epoch: int = 0
    ) -> AsyncIterator[AudioFrame]: ...

    def abort(self) -> None: ...

    @property
    def aborted(self) -> bool: ...


@dataclass
class MockSpeechEngine:
    """Deterministic synthesiser used by CI, benchmarks and the offline demo.

    It produces real PCM frames -- a voiced tone whose duration tracks the
    text's syllable count -- so the whole downstream path (ring buffer, epoch
    invalidation, transport, barge-in) is exercised for real rather than
    stubbed. Only the acoustic model is fake.

    ``first_frame_latency_ms`` models a production engine's time-to-first-audio
    so the latency harness measures a realistic pipeline rather than an
    instantaneous one.
    """

    first_frame_latency_ms: float = 12.0
    #: Speaking rate. 14 characters per 100ms is close to natural English pace.
    chars_per_100ms: float = 14.0
    #: When True, pace emission to wall clock. Off by default because a real
    #: synthesiser generates faster than realtime and the transport is what
    #: paces playback -- but the interruption tests need a stream that is still
    #: running when the barge-in arrives.
    realtime: bool = False
    _aborted: bool = field(default=False, init=False)

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True

    def reset(self) -> None:
        self._aborted = False

    async def stream(
        self, text_chunks: AsyncIterator[str], epoch: int = 0
    ) -> AsyncIterator[AudioFrame]:
        self._aborted = False
        emitted_ms = 0.0
        first = True
        async for chunk in text_chunks:
            if self._aborted:
                return
            if first:
                await asyncio.sleep(self.first_frame_latency_ms / 1000.0)
                first = False
            duration_ms = max(
                FRAME_DURATION_MS,
                round(len(chunk) / self.chars_per_100ms * 100.0 / FRAME_DURATION_MS)
                * FRAME_DURATION_MS,
            )
            n_frames = int(duration_ms // FRAME_DURATION_MS)
            for i in range(n_frames):
                if self._aborted:
                    return
                # Slight pitch drift keeps successive frames distinguishable in
                # test assertions without changing timing.
                pcm = synth_tone(FRAME_DURATION_MS, freq_hz=170.0 + (i % 5) * 4.0)
                yield AudioFrame(samples=pcm, timestamp_ms=emitted_ms, epoch=epoch)
                emitted_ms += FRAME_DURATION_MS
                # Yield control so an abort from another task is observed
                # promptly; without this the loop would run to completion.
                await asyncio.sleep(FRAME_DURATION_MS / 1000.0 if self.realtime else 0)


class SpeechToSpeechEngine:
    """Speech-to-speech bridge over a streaming WebSocket (MOSS-Speech et al).

    Kept behind the same protocol as the mock so the orchestrator is unaware of
    which is in use. On any connection failure it degrades to
    :class:`MockSpeechEngine` rather than ending the call -- the candidate
    hears a synthetic voice instead of silence, and the session is flagged in
    telemetry for review.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._aborted = False
        self._fallback = MockSpeechEngine()
        self._connection: object | None = None

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True
        self._fallback.abort()

    @property
    def configured(self) -> bool:
        return bool(self.settings.speech_ws_url and self.settings.speech_api_key)

    async def stream(
        self, text_chunks: AsyncIterator[str], epoch: int = 0
    ) -> AsyncIterator[AudioFrame]:
        self._aborted = False
        if not self.configured:
            async for frame in self._fallback.stream(text_chunks, epoch):
                yield frame
            return
        try:  # pragma: no cover - requires a live speech endpoint
            async for frame in self._stream_remote(text_chunks, epoch):
                yield frame
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning(
                "Speech engine failed mid-turn (%s); continuing on the local fallback voice.",
                type(exc).__name__,
            )
            async for frame in self._fallback.stream(text_chunks, epoch):
                yield frame

    async def _stream_remote(  # pragma: no cover - requires a live endpoint
        self, text_chunks: AsyncIterator[str], epoch: int
    ) -> AsyncIterator[AudioFrame]:
        import websockets  # type: ignore[import-not-found]

        async with websockets.connect(
            self.settings.speech_ws_url,
            additional_headers={"Authorization": f"Bearer {self.settings.speech_api_key}"},
            # No compression: it adds a buffering stage, and PCM at these frame
            # sizes barely compresses anyway.
            compression=None,
            max_queue=8,
        ) as socket:

            async def pump() -> None:
                async for chunk in text_chunks:
                    if self._aborted:
                        break
                    await socket.send(chunk)
                await socket.send("")  # end-of-utterance sentinel

            pump_task = asyncio.create_task(pump())
            try:
                emitted_ms = 0.0
                async for message in socket:
                    if self._aborted:
                        break
                    if not isinstance(message, (bytes, bytearray)):
                        continue
                    pcm = np.frombuffer(message, dtype=np.int16)
                    for i in range(0, len(pcm) - 959, 960):
                        yield AudioFrame(
                            samples=pcm[i : i + 960].copy(),
                            sample_rate=SAMPLE_RATE_HZ,
                            timestamp_ms=emitted_ms,
                            epoch=epoch,
                        )
                        emitted_ms += FRAME_DURATION_MS
            finally:
                pump_task.cancel()


def build_synthesizer(settings: Settings | None = None) -> SpeechSynthesizer:
    """Select a synthesiser from configuration, always returning a working one."""
    settings = settings or get_settings()
    from app.config import SpeechEngine

    if settings.speech_engine is SpeechEngine.MOSS:
        return SpeechToSpeechEngine(settings)
    return MockSpeechEngine()


#: Backwards-compatible alias for the pre-rename class name.
MossStreamingEngine = SpeechToSpeechEngine

__all__ = [
    "MIN_CHUNK_CHARS",
    "MockSpeechEngine",
    "MossStreamingEngine",
    "SpeechSynthesizer",
    "SpeechToSpeechEngine",
    "build_synthesizer",
    "sentence_chunks",
]

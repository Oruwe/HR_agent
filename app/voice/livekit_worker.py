"""WebRTC transport and the session worker that binds VAD to the agent loop.

The transport is behind a :class:`Transport` protocol for a practical reason,
not an architectural-purity one: LiveKit needs a server, a room, credentials
and a second participant before it will move a single frame. Making transport
swappable means the entire turn-taking path -- endpointing, speculation, the
barge-in drain, the latency accounting -- is exercised end to end in CI, in
milliseconds, with no infrastructure. The LiveKit implementation then has one
job (move bytes) and nothing else to get wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.config import FRAME_DURATION_MS, SAMPLE_RATE_HZ, Settings, get_settings
from app.voice.moss_engine import SpeechSynthesizer, build_synthesizer
from app.voice.ring_buffer import AudioFrame, DualTrackBuffer, now_ms
from app.voice.vad_stream import VadEvent, VadEventType, VadStream, build_detector

logger = logging.getLogger(__name__)


@runtime_checkable
class Transport(Protocol):
    """Bidirectional 20ms audio transport."""

    async def subscribe(self) -> AsyncIterator[AudioFrame]: ...

    async def publish(self, frame: AudioFrame) -> None: ...

    async def clear_egress(self) -> None: ...

    async def close(self) -> None: ...


@dataclass
class InProcessTransport:
    """In-memory transport for tests, benchmarks and the offline demo.

    ``published`` retains every frame that actually reached the wire, which is
    what the interruption test asserts against: the question is never "did we
    stop generating" but "did the candidate stop hearing us".
    """

    incoming: list[AudioFrame] = field(default_factory=list)
    published: list[AudioFrame] = field(default_factory=list)
    #: Pace subscription to wall clock, as a real SFU would.
    realtime: bool = False
    _closed: bool = field(default=False, init=False)
    clear_count: int = field(default=0, init=False)

    async def subscribe(self) -> AsyncIterator[AudioFrame]:
        for frame in self.incoming:
            if self._closed:
                return
            yield frame
            await asyncio.sleep(FRAME_DURATION_MS / 1000.0 if self.realtime else 0)

    async def publish(self, frame: AudioFrame) -> None:
        self.published.append(frame)

    async def clear_egress(self) -> None:
        self.clear_count += 1

    async def close(self) -> None:
        self._closed = True

    @property
    def published_ms(self) -> float:
        return sum(f.duration_ms for f in self.published)


class LiveKitTransport:  # pragma: no cover - requires a live LiveKit server
    """LiveKit WebRTC transport.

    Notes that matter for the 20ms ingress and 10ms egress budgets:

    * ``AudioSource`` is constructed with a small queue. LiveKit will happily
      buffer a second of audio for you, and every buffered millisecond is
      latency a barge-in has to flush through.
    * We publish raw PCM frames rather than re-encoding. Opus encode/decode
      would add roughly 20-40ms of algorithmic delay in each direction, which
      is a quarter of the entire budget spent on a codec we do not need
      server-side.
    """

    def __init__(self, settings: Settings | None = None, room_name: str = "") -> None:
        self.settings = settings or get_settings()
        self.room_name = room_name or f"{self.settings.room_prefix}-default"
        self._room: object | None = None
        self._source: object | None = None
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=8)
        # asyncio keeps only a weak reference to running tasks, so a fire-and-
        # forget pump can be garbage collected mid-call. Holding them here is
        # what keeps candidate audio flowing.
        self._tasks: set[asyncio.Task[None]] = set()

    async def connect(self, token: str) -> None:
        from livekit import rtc  # type: ignore[import-not-found]

        room = rtc.Room()

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            task = asyncio.create_task(self._pump_track(rtc.AudioStream(track)))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        await room.connect(self.settings.livekit_url, token)
        self._room = room

        source = rtc.AudioSource(SAMPLE_RATE_HZ, 1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self._source = source

    async def _pump_track(self, stream: object) -> None:
        import numpy as np

        started = now_ms()
        async for event in stream:  # type: ignore[attr-defined]
            data = np.frombuffer(event.frame.data, dtype=np.int16)
            frame = AudioFrame(samples=data.copy(), timestamp_ms=now_ms() - started)
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                # Dropping the oldest frame is correct under load: stale
                # candidate audio is worth less than a current endpoint decision.
                _ = self._queue.get_nowait()
                self._queue.put_nowait(frame)

    async def subscribe(self) -> AsyncIterator[AudioFrame]:
        while True:
            yield await self._queue.get()

    async def publish(self, frame: AudioFrame) -> None:
        if self._source is None:
            return
        from livekit import rtc  # type: ignore[import-not-found]

        await self._source.capture_frame(  # type: ignore[attr-defined]
            rtc.AudioFrame(
                data=frame.to_bytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.channels,
                samples_per_channel=len(frame.samples),
            )
        )

    async def clear_egress(self) -> None:
        if self._source is not None:
            await self._source.clear_queue()  # type: ignore[attr-defined]

    async def close(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()
        if self._room is not None:
            await self._room.disconnect()  # type: ignore[attr-defined]
            self._room = None


@dataclass
class InterruptionMetrics:
    """What the barge-in path actually did, for assertions and dashboards."""

    barge_ins: int = 0
    frames_dropped: int = 0
    last_drain_ms: float = 0.0
    worst_drain_ms: float = 0.0

    def record(self, dropped: int, elapsed_ms: float) -> None:
        self.barge_ins += 1
        self.frames_dropped += dropped
        self.last_drain_ms = elapsed_ms
        self.worst_drain_ms = max(self.worst_drain_ms, elapsed_ms)


class VoiceWorker:
    """Owns one screening session's audio loop.

    Responsibilities, in priority order:

    1. **Interrupt fast.** Barge-in is handled inline in the ingress loop, ahead
       of any other work. It is the only thing in this class with a hard
       deadline measured in single-digit milliseconds.
    2. **Commit turns on time**, honouring the speculation window.
    3. **Publish agent audio** without letting the egress buffer grow into
       latency.
    """

    def __init__(
        self,
        transport: Transport,
        settings: Settings | None = None,
        synthesizer: SpeechSynthesizer | None = None,
        vad: VadStream | None = None,
        on_speculate: Callable[[VadEvent], Awaitable[None]] | None = None,
        on_commit: Callable[[VadEvent], Awaitable[AsyncIterator[str] | None]] | None = None,
        on_cancel: Callable[[VadEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport
        self.synthesizer = synthesizer or build_synthesizer(self.settings)
        self.vad = vad or VadStream(settings=self.settings, detector=build_detector(self.settings))
        self.buffers = DualTrackBuffer()
        self.metrics = InterruptionMetrics()
        self._on_speculate = on_speculate
        self._on_commit = on_commit
        self._on_cancel = on_cancel
        self._speak_task: asyncio.Task[None] | None = None
        self._events: list[VadEvent] = []

    @property
    def events(self) -> tuple[VadEvent, ...]:
        return tuple(self._events)

    # -- interruption --------------------------------------------------------

    async def interrupt(self) -> float:
        """Stop the agent talking. Returns how long the drain took, in ms.

        The order is load-bearing. The epoch is bumped first, which
        retroactively invalidates every frame already in flight; only then do we
        cancel the producer and flush the transport. Cancelling first would
        leave a window in which the synthesiser's final frame lands in a
        freshly-cleared buffer and gets published after the interrupt.
        """
        started = now_ms()

        dropped = self.buffers.interrupt()  # 1. invalidate (O(1))
        self.synthesizer.abort()  # 2. stop the producer
        if self._speak_task is not None and not self._speak_task.done():
            self._speak_task.cancel()
        await self.transport.clear_egress()  # 3. flush the wire

        self.vad.end_agent_turn()
        elapsed = now_ms() - started
        self.metrics.record(dropped, elapsed)
        return elapsed

    # -- egress --------------------------------------------------------------

    async def speak(self, text_chunks: AsyncIterator[str]) -> None:
        """Synthesise into the egress buffer and publish from it, paced.

        Production and publication are separate tasks on purpose. A speech
        engine generates several times faster than realtime while the wire
        consumes exactly one 20ms frame per 20ms, so audio *will* queue between
        them -- and that queue is precisely what a barge-in has to throw away.
        Collapsing the two into one loop would hide the queue, make the drain
        look free, and ship a system whose interruption behaviour was never
        actually tested.
        """
        epoch = self.buffers.egress.epoch
        self.vad.begin_agent_turn()
        producer = asyncio.create_task(self._produce(text_chunks, epoch))
        try:
            await self._publish_paced(epoch, producer)
        except asyncio.CancelledError:
            raise
        finally:
            if not producer.done():
                producer.cancel()
            if self.vad.agent_is_speaking:
                self.vad.end_agent_turn()

    async def _produce(self, text_chunks: AsyncIterator[str], epoch: int) -> None:
        """Synthesiser -> egress buffer. Applies backpressure when full."""
        async for frame in self.synthesizer.stream(text_chunks, epoch=epoch):
            if epoch != self.buffers.egress.epoch:
                return  # interrupted while this frame was being made
            while not self.buffers.egress.push(frame):
                if epoch != self.buffers.egress.epoch:
                    return
                # Buffer full: the wire is the bottleneck, which is the healthy
                # state. Wait one frame rather than dropping audio.
                await asyncio.sleep(FRAME_DURATION_MS / 2000.0)

    async def _publish_paced(self, epoch: int, producer: asyncio.Task[None]) -> None:
        """Egress buffer -> wire, at one frame per frame-duration."""
        published = 0
        started = now_ms()
        while True:
            if epoch != self.buffers.egress.epoch:
                return  # interrupted
            frame = self.buffers.egress.pop()
            if frame is None:
                if producer.done():
                    return
                await asyncio.sleep(FRAME_DURATION_MS / 4000.0)
                continue
            await self.transport.publish(frame)
            published += 1
            # Absolute deadlines rather than a fixed sleep, so publish latency
            # does not accumulate into drift over a long answer.
            deadline = started + published * FRAME_DURATION_MS
            lag = deadline - now_ms()
            if lag > 0:
                await asyncio.sleep(lag / 1000.0)

    # -- ingress -------------------------------------------------------------

    async def run(self) -> None:
        """Consume candidate audio until the transport closes."""
        async for frame in self.transport.subscribe():
            self.buffers.ingress.push(
                self.buffers.ingress.current_frame(frame.samples, frame.timestamp_ms)
            )
            event = self.vad.process(frame)
            if event is None:
                continue
            self._events.append(event)
            await self._dispatch(event)

    async def _dispatch(self, event: VadEvent) -> None:
        if event.type is VadEventType.BARGE_IN:
            await self.interrupt()
            return
        if event.type is VadEventType.SPECULATE and self._on_speculate is not None:
            await self._on_speculate(event)
            return
        if event.type is VadEventType.SPECULATION_CANCELLED and self._on_cancel is not None:
            await self._on_cancel(event)
            return
        if event.type is VadEventType.TURN_COMMIT and self._on_commit is not None:
            chunks = await self._on_commit(event)
            if chunks is not None:
                self._speak_task = asyncio.create_task(self.speak(chunks))
            return

    async def drain_speaking(self) -> None:
        """Await the in-flight agent turn, if any."""
        if self._speak_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._speak_task
            self._speak_task = None

    async def close(self) -> None:
        await self.drain_speaking()
        await self.transport.close()


def build_access_token(  # pragma: no cover - requires livekit-api
    settings: Settings, room_name: str, identity: str
) -> str:
    """Mint a LiveKit room token scoped to exactly one room.

    Scoped deliberately: a screening token that can join any room is a
    cross-candidate data leak waiting for its first bug.
    """
    from livekit import api  # type: ignore[import-not-found]

    return (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=False,
            )
        )
        .to_jwt()
    )


__all__ = [
    "InProcessTransport",
    "InterruptionMetrics",
    "LiveKitTransport",
    "Transport",
    "VoiceWorker",
    "build_access_token",
]

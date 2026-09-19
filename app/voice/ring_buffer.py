"""Audio frames and the epoch-tagged ring buffer that makes barge-in O(1).

The hard requirement is that when a candidate interrupts, every already-queued
agent audio frame stops within 15ms -- including frames that a producer
coroutine is *about to* push, having decided to push them before the interrupt
arrived. Clearing a queue does not solve that: the producer wakes up afterwards
and pushes stale audio into the now-empty buffer, and the candidate hears the
agent talk over them for another few hundred milliseconds.

The fix is a monotonically increasing epoch. Draining bumps the epoch, and a
frame stamped with an older epoch is refused on arrival. Producers do not need
to check a flag, take a lock, or be cancellable at a safe point; correctness
comes from the data, not from coordination.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from app.config import (
    BYTES_PER_SAMPLE,
    CHANNELS,
    FRAME_DURATION_MS,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_FRAME,
)


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One 20ms mono PCM16 frame at 48kHz.

    Samples are held as ``int16`` because that is what LiveKit publishes and
    what Silero consumes; converting to float is done once, at the VAD boundary,
    rather than on every hop.
    """

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE_HZ
    channels: int = CHANNELS
    #: Milliseconds since session start.
    timestamp_ms: float = 0.0
    #: Buffer epoch this frame belongs to; see module docstring.
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"Expected a mono 1-D frame, got shape {self.samples.shape}")
        if self.samples.dtype != np.int16:
            raise ValueError(f"Expected int16 PCM, got {self.samples.dtype}")

    @property
    def duration_ms(self) -> float:
        return 1000.0 * len(self.samples) / self.sample_rate

    @property
    def nbytes(self) -> int:
        return len(self.samples) * BYTES_PER_SAMPLE

    def as_float32(self) -> np.ndarray:
        """Normalised float view in [-1, 1], which is what the VAD expects."""
        return self.samples.astype(np.float32) / 32768.0

    def rms(self) -> float:
        """Root-mean-square amplitude, normalised. Cheap voicing proxy."""
        if len(self.samples) == 0:
            return 0.0
        as_float = self.as_float32()
        return float(np.sqrt(np.mean(as_float * as_float)))

    def to_bytes(self) -> bytes:
        return self.samples.tobytes()

    @classmethod
    def silence(cls, timestamp_ms: float = 0.0, epoch: int = 0) -> AudioFrame:
        return cls(
            samples=np.zeros(SAMPLES_PER_FRAME, dtype=np.int16),
            timestamp_ms=timestamp_ms,
            epoch=epoch,
        )

    @classmethod
    def from_bytes(cls, raw: bytes, timestamp_ms: float = 0.0, epoch: int = 0) -> AudioFrame:
        return cls(
            samples=np.frombuffer(raw, dtype=np.int16).copy(),
            timestamp_ms=timestamp_ms,
            epoch=epoch,
        )


@dataclass(slots=True)
class RingBufferStats:
    pushed: int = 0
    popped: int = 0
    dropped_overflow: int = 0
    dropped_stale: int = 0
    drains: int = 0


@dataclass(slots=True)
class AudioRingBuffer:
    """Bounded FIFO of audio frames with epoch-based invalidation.

    ``capacity_ms`` bounds how much audio can sit queued ahead of the candidate.
    A deep buffer feels smooth right up to the moment someone interrupts, at
    which point every queued millisecond is latency the candidate perceives as
    the agent ignoring them. 400ms is about two sentences of runway.
    """

    capacity_ms: float = 400.0
    name: str = "egress"
    _frames: deque[AudioFrame] = field(default_factory=deque, init=False, repr=False)
    _epoch: int = field(default=0, init=False)
    _stats: RingBufferStats = field(default_factory=RingBufferStats, init=False)

    @property
    def capacity_frames(self) -> int:
        return max(1, int(self.capacity_ms // FRAME_DURATION_MS))

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stats(self) -> RingBufferStats:
        return self._stats

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[AudioFrame]:
        return iter(tuple(self._frames))

    @property
    def buffered_ms(self) -> float:
        return sum(f.duration_ms for f in self._frames)

    def current_frame(self, samples: np.ndarray, timestamp_ms: float = 0.0) -> AudioFrame:
        """Stamp a frame with the buffer's current epoch."""
        return AudioFrame(samples=samples, timestamp_ms=timestamp_ms, epoch=self._epoch)

    def push(self, frame: AudioFrame) -> bool:
        """Enqueue a frame. Returns False if it was refused.

        A frame is refused when it carries a stale epoch (its generation was
        interrupted) or when the buffer is full. Both are counted rather than
        raising: an audio producer must never be interrupted by an exception on
        a path that runs 50 times a second.
        """
        if frame.epoch != self._epoch:
            self._stats.dropped_stale += 1
            return False
        if len(self._frames) >= self.capacity_frames:
            self._stats.dropped_overflow += 1
            return False
        self._frames.append(frame)
        self._stats.pushed += 1
        return True

    def pop(self) -> AudioFrame | None:
        if not self._frames:
            return None
        self._stats.popped += 1
        return self._frames.popleft()

    def drain(self) -> int:
        """Invalidate the buffer. Returns how many queued frames were dropped.

        This is the barge-in primitive. It bumps the epoch *before* clearing so
        that a producer racing us cannot slip a frame in behind the clear.
        """
        self._epoch += 1
        dropped = len(self._frames)
        self._frames.clear()
        self._stats.drains += 1
        return dropped

    def snapshot(self) -> tuple[AudioFrame, ...]:
        return tuple(self._frames)


class DualTrackBuffer:
    """Candidate-in and agent-out buffers, kept side by side.

    Keeping both tracks in one object is what allows a barge-in decision to be
    made and enacted in a single step: the candidate frame that proves speech is
    in ``ingress`` at the moment ``egress`` needs draining, with no cross-task
    message in between.
    """

    def __init__(self, ingress_ms: float = 600.0, egress_ms: float = 400.0) -> None:
        self.ingress = AudioRingBuffer(capacity_ms=ingress_ms, name="ingress")
        self.egress = AudioRingBuffer(capacity_ms=egress_ms, name="egress")

    def interrupt(self) -> int:
        """Drain agent output. Candidate audio is never discarded."""
        return self.egress.drain()

    def reset(self) -> None:
        self.ingress.drain()
        self.egress.drain()


def frames_from_pcm(pcm: np.ndarray, start_ms: float = 0.0, epoch: int = 0) -> list[AudioFrame]:
    """Slice a PCM16 array into 20ms frames, dropping any partial tail."""
    if pcm.dtype != np.int16:
        raise ValueError(f"Expected int16 PCM, got {pcm.dtype}")
    out: list[AudioFrame] = []
    for i in range(0, len(pcm) - SAMPLES_PER_FRAME + 1, SAMPLES_PER_FRAME):
        out.append(
            AudioFrame(
                samples=pcm[i : i + SAMPLES_PER_FRAME],
                timestamp_ms=start_ms + (i / SAMPLES_PER_FRAME) * FRAME_DURATION_MS,
                epoch=epoch,
            )
        )
    return out


def synth_tone(
    duration_ms: float,
    freq_hz: float = 180.0,
    amplitude: float = 0.35,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Deterministic voiced-speech stand-in for tests and benchmarks.

    A sine at roughly male speaking F0 with a light second harmonic: enough
    periodic structure for an energy or spectral VAD to treat it as voiced,
    without pulling a speech corpus into the test suite.
    """
    n = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave = np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(2 * np.pi * freq_hz * 2 * t)
    wave = wave / np.max(np.abs(wave)) if n else wave
    return (wave * amplitude * 32767).astype(np.int16)


def synth_silence(duration_ms: float, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Digital silence with a trace of dither, as a real microphone produces."""
    n = int(sample_rate * duration_ms / 1000.0)
    rng = np.random.default_rng(seed=1337)  # fixed seed: reproducible tests
    return (rng.standard_normal(n) * 12).astype(np.int16)


def now_ms() -> float:
    """Monotonic milliseconds. Never wall-clock -- NTP steps would corrupt spans."""
    return time.perf_counter() * 1000.0


__all__ = [
    "AudioFrame",
    "AudioRingBuffer",
    "DualTrackBuffer",
    "RingBufferStats",
    "frames_from_pcm",
    "now_ms",
    "synth_silence",
    "synth_tone",
]

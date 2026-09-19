"""Barge-in: the candidate interrupts and the agent stops being heard.

The assertion that matters is not "did we stop generating audio" but "did the
candidate stop hearing us". Those differ by however much audio is already queued
between the synthesiser and the wire, which is exactly the gap a naive
implementation leaves open: it cancels the producer, the queue keeps draining,
and the agent talks over the candidate for another few hundred milliseconds.

Every test here therefore asserts against what actually reached the transport.
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from app.config import FRAME_DURATION_MS, Settings
from app.voice.livekit_worker import InProcessTransport, VoiceWorker
from app.voice.moss_engine import MockSpeechEngine, sentence_chunks
from app.voice.ring_buffer import (
    AudioFrame,
    AudioRingBuffer,
    DualTrackBuffer,
    frames_from_pcm,
    now_ms,
    synth_silence,
    synth_tone,
)
from app.voice.vad_stream import VadEventType, VadState, VadStream
from tests.conftest import run_async

#: The contract: queued agent audio must stop within this many milliseconds.
DRAIN_DEADLINE_MS = 15.0

LONG_ANSWER = (
    "That is a good question, and the honest answer is that we measured it "
    "before we decided, because the intuition pointed the other way, and then "
    "we rolled it out region by region over about three weeks."
)


async def _token_stream(text: str):
    for word in text.split(" "):
        yield word + " "


# =============================================================================
# Ring buffer primitives
# =============================================================================


def test_drain_clears_the_buffer_and_reports_the_loss() -> None:
    buffer = AudioRingBuffer(capacity_ms=400.0)
    for i in range(10):
        buffer.push(buffer.current_frame(synth_tone(FRAME_DURATION_MS), i * 20.0))
    assert len(buffer) == 10

    dropped = buffer.drain()
    assert dropped == 10
    assert len(buffer) == 0
    assert buffer.buffered_ms == 0.0


def test_drain_bumps_the_epoch_and_refuses_in_flight_frames() -> None:
    """The race a plain queue.clear() cannot close.

    A producer that decided to push *before* the interrupt will complete that
    push afterwards. Epoch tagging makes the frame invalid on arrival, so the
    producer needs no cancellation point and no lock.
    """
    buffer = AudioRingBuffer()
    in_flight = buffer.current_frame(synth_tone(FRAME_DURATION_MS), 0.0)

    buffer.drain()

    assert buffer.epoch == 1
    assert buffer.push(in_flight) is False
    assert len(buffer) == 0
    assert buffer.stats.dropped_stale == 1


def test_drain_is_fast_enough_to_be_called_inline() -> None:
    buffer = AudioRingBuffer(capacity_ms=2000.0)
    for i in range(buffer.capacity_frames):
        buffer.push(buffer.current_frame(synth_tone(FRAME_DURATION_MS), i * 20.0))

    start = now_ms()
    buffer.drain()
    elapsed = now_ms() - start

    assert elapsed <= DRAIN_DEADLINE_MS, f"drain took {elapsed:.3f}ms"


def test_buffer_is_bounded() -> None:
    """A deep buffer feels smooth until someone interrupts; then it is all latency."""
    buffer = AudioRingBuffer(capacity_ms=100.0)
    accepted = sum(
        buffer.push(buffer.current_frame(synth_tone(FRAME_DURATION_MS), i * 20.0))
        for i in range(50)
    )
    assert accepted == buffer.capacity_frames == 5
    assert buffer.stats.dropped_overflow == 45


def test_dual_track_interrupt_spares_candidate_audio() -> None:
    """Never discard what the candidate said -- it is the evidence."""
    buffers = DualTrackBuffer()
    buffers.ingress.push(buffers.ingress.current_frame(synth_tone(FRAME_DURATION_MS), 0.0))
    buffers.egress.push(buffers.egress.current_frame(synth_tone(FRAME_DURATION_MS), 0.0))

    dropped = buffers.interrupt()

    assert dropped == 1
    assert len(buffers.egress) == 0
    assert len(buffers.ingress) == 1


# =============================================================================
# VAD barge-in detection
# =============================================================================


def test_barge_in_fires_after_the_configured_speech_threshold() -> None:
    settings = Settings()
    vad = VadStream(settings=settings)
    vad.begin_agent_turn()
    assert vad.agent_is_speaking

    event = None
    for frame in frames_from_pcm(synth_tone(200), start_ms=1000.0):
        event = vad.process(frame)
        if event is not None:
            break

    assert event is not None and event.type is VadEventType.BARGE_IN
    assert event.speech_ms >= settings.barge_in_ms
    assert event.speech_ms <= settings.barge_in_ms + FRAME_DURATION_MS * 2
    assert vad.state is VadState.CANDIDATE_SPEAKING


def test_silence_during_agent_speech_does_not_barge_in() -> None:
    vad = VadStream()
    vad.begin_agent_turn()
    events = [vad.process(frame) for frame in frames_from_pcm(synth_silence(400), start_ms=1000.0)]
    assert all(e is None for e in events)
    assert vad.agent_is_speaking


def test_a_single_transient_does_not_barge_in() -> None:
    """A door slam is one frame. It must not stop the agent mid-sentence."""
    vad = VadStream()
    vad.begin_agent_turn()
    pcm = np.concatenate([synth_tone(20), synth_silence(300)])
    events = [e for e in (vad.process(f) for f in frames_from_pcm(pcm, 1000.0)) if e]
    assert not events
    assert vad.agent_is_speaking


# =============================================================================
# End-to-end interruption
# =============================================================================


def test_nothing_is_published_after_an_interrupt() -> None:
    """The load-bearing assertion of this whole file."""

    async def scenario() -> tuple[int, int, float, int]:
        transport = InProcessTransport()
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())
        task = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))

        await asyncio.sleep(0.10)  # let a few frames reach the wire
        published_before = len(transport.published)
        queued = len(worker.buffers.egress)

        elapsed = await worker.interrupt()
        await asyncio.sleep(0.20)  # ample time for a straggler to slip through

        with contextlib.suppress(asyncio.CancelledError):
            await task
        return published_before, len(transport.published), elapsed, queued

    before, after, elapsed, queued = run_async(scenario)

    assert before > 0, "the agent never started speaking; the test proves nothing"
    assert queued > 0, "nothing was queued; the drain was not actually exercised"
    assert after == before, f"{after - before} frame(s) published after the interrupt"
    assert elapsed <= DRAIN_DEADLINE_MS, f"drain took {elapsed:.3f}ms"


def test_interrupt_drops_the_queued_audio_and_records_it() -> None:
    async def scenario() -> VoiceWorker:
        transport = InProcessTransport()
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())
        task = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))
        await asyncio.sleep(0.10)
        await worker.interrupt()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return worker

    worker = run_async(scenario)
    assert worker.metrics.barge_ins == 1
    assert worker.metrics.frames_dropped > 0
    assert worker.metrics.worst_drain_ms <= DRAIN_DEADLINE_MS


def test_interrupt_flushes_the_transport() -> None:
    """Draining our own buffer is not enough; the wire has a queue too."""

    async def scenario() -> InProcessTransport:
        transport = InProcessTransport()
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())
        task = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))
        await asyncio.sleep(0.06)
        await worker.interrupt()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return transport

    assert run_async(scenario).clear_count == 1


def test_interrupt_aborts_the_synthesiser() -> None:
    async def scenario() -> MockSpeechEngine:
        transport = InProcessTransport()
        synth = MockSpeechEngine()
        worker = VoiceWorker(transport, synthesizer=synth)
        task = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))
        await asyncio.sleep(0.06)
        await worker.interrupt()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return synth

    assert run_async(scenario).aborted


def test_agent_can_speak_again_after_an_interrupt() -> None:
    """An interrupt must not wedge the worker; the interview continues."""

    async def scenario() -> int:
        transport = InProcessTransport()
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())

        first = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))
        await asyncio.sleep(0.06)
        await worker.interrupt()
        with contextlib.suppress(asyncio.CancelledError):
            await first

        published_after_interrupt = len(transport.published)
        worker.synthesizer = MockSpeechEngine()
        await worker.speak(sentence_chunks(_token_stream("Understood, go on.")))
        return len(transport.published) - published_after_interrupt

    assert run_async(scenario) > 0, "the worker never recovered the floor"


def test_barge_in_through_the_full_ingress_loop() -> None:
    """Drive the real path: candidate audio arrives, the VAD decides, audio stops.

    Two distinct quantities are checked, and conflating them is the easy mistake:

    * **Overspeak** -- audio published between the candidate opening their mouth
      and the VAD being convinced. This is non-zero by design. The detector
      requires ``barge_in_ms`` of sustained energy precisely so that a cough
      does not stop the agent, and the agent is legitimately still speaking
      during that window.
    * **Leakage** -- audio published *after* the interrupt fired. This must be
      exactly zero, and it is the actual contract.
    """

    async def scenario() -> tuple[int, int, int, VoiceWorker]:
        transport = InProcessTransport(
            incoming=frames_from_pcm(synth_tone(300), start_ms=1000.0), realtime=True
        )
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())

        speaking = asyncio.create_task(worker.speak(sentence_chunks(_token_stream(LONG_ANSWER))))
        await asyncio.sleep(0.08)
        at_candidate_onset = len(transport.published)

        await worker.run()  # candidate speaks over the agent
        at_interrupt = len(transport.published)

        await asyncio.sleep(0.20)  # ample time for a straggler to slip through
        with contextlib.suppress(asyncio.CancelledError):
            await speaking
        return at_candidate_onset, at_interrupt, len(transport.published), worker

    onset, at_interrupt, final, worker = run_async(scenario)

    assert worker.metrics.barge_ins >= 1, "the ingress loop never detected the barge-in"
    assert any(e.type is VadEventType.BARGE_IN for e in worker.events)

    assert final == at_interrupt, (
        f"{final - at_interrupt} frame(s) leaked after the interrupt fired"
    )

    overspeak_ms = (at_interrupt - onset) * FRAME_DURATION_MS
    detection_budget_ms = Settings().barge_in_ms + 2 * FRAME_DURATION_MS
    assert overspeak_ms <= detection_budget_ms, (
        f"agent overspoke the candidate by {overspeak_ms:.0f}ms; the detector "
        f"should have been convinced within {detection_budget_ms:.0f}ms"
    )


def test_frames_from_a_stale_epoch_are_never_published() -> None:
    """Directly exercise the invariant the epoch scheme exists to guarantee."""

    async def scenario() -> int:
        transport = InProcessTransport()
        worker = VoiceWorker(transport, synthesizer=MockSpeechEngine())
        stale = AudioFrame(samples=synth_tone(FRAME_DURATION_MS), epoch=0)

        worker.buffers.egress.drain()  # epoch -> 1
        accepted = worker.buffers.egress.push(stale)
        assert accepted is False

        await transport.close()
        return len(transport.published)

    assert run_async(scenario) == 0

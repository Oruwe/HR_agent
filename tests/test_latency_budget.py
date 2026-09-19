"""The 160ms turnaround budget, measured rather than asserted by construction.

What "turnaround" means here is precise and worth stating, because the number is
easy to make meaningless: it is the wall time from **endpoint commit** (the VAD
has decided the candidate finished) to **the first synthesised audio frame
reaching the transport**. It is not time-to-first-token, and it is not the time
to produce the whole answer.

Measurements are taken against the offline providers. That is the point, not a
compromise: with a live model the numbers would be dominated by network
variance and would tell you nothing about whether *this pipeline* regressed. The
mock providers model realistic component costs (30ms TTFT, 12ms to first audio),
so what is under test is the orchestration -- speculation, prefetch, chunking,
buffering -- which is the part this repository actually controls.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from app.agent.cognition import MockCognition
from app.agent.orchestrator import ScreeningOrchestrator, new_session
from app.config import (
    PREFETCHABLE_STAGES,
    STAGE_BUDGETS_MS,
    TOTAL_TURNAROUND_BUDGET_MS,
    Settings,
    Stage,
    assert_budget_is_coherent,
)
from app.schemas.roles import EngineeringRole
from app.storage.qdrant_client import HybridVectorStore
from app.telemetry.metrics import LatencyRecorder, StageTimer, TurnMetrics, percentile
from app.voice.moss_engine import MockSpeechEngine, sentence_chunks
from app.voice.ring_buffer import frames_from_pcm, now_ms, synth_silence, synth_tone
from app.voice.vad_stream import VadEventType, VadStream
from tests.conftest import ROLE_PROFILES, run_async

ANSWERS = [
    "We ran FSDP across 512 A100s and the collective ops were the bottleneck "
    "until we retuned NCCL and added gradient checkpointing.",
    "Tensor parallelism inside the node because NVLink has the bandwidth, "
    "pipeline parallelism across nodes where you are on InfiniBand.",
    "KV-cache is two times layers times heads times head dimension times "
    "sequence length times batch, in half precision.",
    "We tuned ef until recall at ten hit ninety eight percent with p99 under twelve milliseconds.",
    "The trade-off was index build time, which roughly tripled.",
    "We measured it over a week of production traffic before committing.",
]


# =============================================================================
# The budget contract itself
# =============================================================================


def test_stage_budgets_sum_to_the_total() -> None:
    assert_budget_is_coherent()
    assert sum(STAGE_BUDGETS_MS.values()) == pytest.approx(TOTAL_TURNAROUND_BUDGET_MS)


def test_every_stage_has_a_budget() -> None:
    assert set(STAGE_BUDGETS_MS) == set(Stage)
    assert all(v > 0 for v in STAGE_BUDGETS_MS.values())


def test_prefetchable_stages_are_declared() -> None:
    """Prefetched stages must be a strict subset -- speculation cannot hide TTFT."""
    assert set(Stage) > PREFETCHABLE_STAGES
    assert Stage.COGNITION_TTFT not in PREFETCHABLE_STAGES
    assert Stage.SPEECH_SYNTHESIS not in PREFETCHABLE_STAGES


def test_speculation_precedes_commit() -> None:
    settings = Settings()
    assert settings.speculative_silence_ms < settings.endpoint_silence_ms
    assert settings.speculation_window_ms > 0


def test_speculation_window_cannot_be_inverted() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        Settings(endpoint_silence_ms=100.0, speculative_silence_ms=250.0)


# =============================================================================
# Per-stage costs
# =============================================================================


def test_retrieval_stays_inside_its_budget(warm_store: HybridVectorStore) -> None:
    evidence = ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER]
    warm_store.search_rubrics(evidence, limit=3)  # exclude first-call warmth

    samples: list[float] = []
    for _ in range(50):
        start = time.perf_counter()
        warm_store.search_rubrics(evidence, limit=3)
        samples.append((time.perf_counter() - start) * 1000.0)

    p95 = percentile(samples, 95)
    assert p95 <= STAGE_BUDGETS_MS[Stage.VECTOR_MATCH], (
        f"retrieval p95 {p95:.2f}ms exceeds the {STAGE_BUDGETS_MS[Stage.VECTOR_MATCH]}ms budget"
    )


def test_vad_endpoint_decision_is_cheap() -> None:
    """The VAD runs on every 20ms frame; its per-frame cost must be negligible."""
    stream = VadStream()
    pcm = np.concatenate([synth_tone(400), synth_silence(400)])
    frames = frames_from_pcm(pcm)

    for frame in frames:  # warm the adaptive floor
        stream.process(frame)
    stream.reset()

    start = time.perf_counter()
    for frame in frames:
        stream.process(frame)
    per_frame_ms = (time.perf_counter() - start) * 1000.0 / len(frames)

    assert per_frame_ms <= STAGE_BUDGETS_MS[Stage.VAD_ENDPOINT] / 5, (
        f"VAD costs {per_frame_ms:.3f}ms per frame; it runs 50 times a second"
    )


def test_vad_commits_within_the_configured_silence_window() -> None:
    settings = Settings()
    stream = VadStream(settings=settings)
    pcm = np.concatenate([synth_tone(400), synth_silence(600)])

    commit_at: float | None = None
    speech_ended_at = 400.0
    for frame in frames_from_pcm(pcm):
        event = stream.process(frame)
        if event is not None and event.type is VadEventType.TURN_COMMIT:
            commit_at = event.timestamp_ms
            break

    assert commit_at is not None, "the VAD never committed the turn"
    overshoot = commit_at - speech_ended_at - settings.endpoint_silence_ms
    assert 0 <= overshoot <= 40, f"endpoint fired {overshoot:.0f}ms beyond the configured window"


# =============================================================================
# End-to-end turnaround
# =============================================================================


async def _measure_turn(
    orchestrator: ScreeningOrchestrator,
    synthesizer: MockSpeechEngine,
    answer: str,
    settings: Settings,
    *,
    speculate: bool,
) -> TurnMetrics:
    """One full turn, measured commit -> first audio frame."""
    orchestrator.observe_candidate(answer)

    if speculate:
        await orchestrator.speculate()
        await asyncio.sleep(settings.speculation_window_ms / 1000.0)

    commit_at = now_ms()
    stream, turn = await orchestrator.commit()

    first_audio_ms: float | None = None
    async for _frame in synthesizer.stream(sentence_chunks(stream)):
        if first_audio_ms is None:
            first_audio_ms = now_ms() - commit_at
    turn.perceived_ms = first_audio_ms if first_audio_ms is not None else 0.0
    return turn


async def _build(settings: Settings) -> tuple[ScreeningOrchestrator, MockSpeechEngine]:
    session = new_session(ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER])
    cognition = MockCognition(ttft_ms=30.0, inter_token_ms=0.3)
    orchestrator = ScreeningOrchestrator(session, settings=settings, cognition=cognition)
    await orchestrator.open()
    return orchestrator, MockSpeechEngine(first_frame_latency_ms=12.0)


def test_single_turn_is_within_budget(settings: Settings) -> None:
    async def scenario() -> TurnMetrics:
        orchestrator, synth = await _build(settings)
        return await _measure_turn(orchestrator, synth, ANSWERS[0], settings, speculate=True)

    turn = run_async(scenario)
    assert turn.perceived_ms <= TOTAL_TURNAROUND_BUDGET_MS, (
        f"turnaround {turn.perceived_ms:.1f}ms exceeds the {TOTAL_TURNAROUND_BUDGET_MS}ms budget"
    )
    assert not turn.latency_exceeded


def test_every_turn_of_a_full_interview_is_within_budget(settings: Settings) -> None:
    async def scenario() -> LatencyRecorder:
        orchestrator, synth = await _build(settings)
        for answer in ANSWERS:
            await _measure_turn(orchestrator, synth, answer, settings, speculate=True)
        return orchestrator.latency

    recorder = run_async(scenario)
    report = recorder.report()

    assert report.samples == len(ANSWERS)
    assert report.compliance == 1.0, f"{report.summary()}"
    assert report.passes, report.summary()
    assert report.p95_ms <= TOTAL_TURNAROUND_BUDGET_MS
    assert report.max_ms <= TOTAL_TURNAROUND_BUDGET_MS


def test_percentiles_hold_over_many_turns(settings: Settings) -> None:
    """p95 is the gate; a single lucky run proves nothing about the tail."""

    async def scenario() -> list[float]:
        orchestrator, synth = await _build(settings)
        out: list[float] = []
        for i in range(24):
            turn = await _measure_turn(
                orchestrator, synth, ANSWERS[i % len(ANSWERS)], settings, speculate=True
            )
            out.append(turn.perceived_ms)
        return out

    samples = run_async(scenario)
    assert percentile(samples, 95) <= TOTAL_TURNAROUND_BUDGET_MS
    assert max(samples) <= TOTAL_TURNAROUND_BUDGET_MS * 2, (
        f"tail turn of {max(samples):.1f}ms: speculation missed and nothing absorbed it"
    )


def test_speculation_is_what_buys_the_budget(settings: Settings) -> None:
    """Speculation must actually buffer cognition before turn commit.

    This test verifies the architectural contract directly instead of relying
    on a wall-clock millisecond threshold, which is inherently scheduler-
    dependent in CI and on developer machines.
    """

    async def scenario() -> tuple[bool, float, float]:
        orchestrator, synth = await _build(settings)
        eager = await _measure_turn(orchestrator, synth, ANSWERS[0], settings, speculate=True)

        cold_orchestrator, cold_synth = await _build(settings)
        cold = await _measure_turn(
            cold_orchestrator, cold_synth, ANSWERS[0], settings, speculate=False
        )

        return eager.speculation_hit, eager.perceived_ms, cold.perceived_ms

    speculation_hit, speculated, cold = run_async(scenario)

    assert speculation_hit, "speculation did not buffer a cognition token before commit"
    assert speculated < cold, f"speculation bought nothing: {speculated:.1f}ms vs {cold:.1f}ms cold"


def test_speculation_hit_rate_is_reported(settings: Settings) -> None:
    async def scenario() -> float:
        orchestrator, synth = await _build(settings)
        for answer in ANSWERS[:4]:
            await _measure_turn(orchestrator, synth, answer, settings, speculate=True)
        return orchestrator.speculation_hit_rate

    assert run_async(scenario) == 1.0


def test_cancelled_speculation_does_not_corrupt_the_next_turn(settings: Settings) -> None:
    """A candidate who pauses and resumes must not poison the committed turn."""

    async def scenario() -> TurnMetrics:
        orchestrator, synth = await _build(settings)
        orchestrator.observe_candidate(ANSWERS[0])
        await orchestrator.speculate()
        await asyncio.sleep(0.01)
        await orchestrator.cancel_speculation()
        return await _measure_turn(orchestrator, synth, ANSWERS[1], settings, speculate=True)

    turn = run_async(scenario)
    assert turn.perceived_ms <= TOTAL_TURNAROUND_BUDGET_MS


# =============================================================================
# Accounting honesty
# =============================================================================


def test_prefetched_stages_are_excluded_from_the_critical_path() -> None:
    turn = TurnMetrics()
    turn.add(Stage.VECTOR_MATCH, 8.0, prefetched=True)
    turn.add(Stage.COGNITION_TTFT, 40.0)
    assert turn.accounted_ms == pytest.approx(40.0)
    assert turn.prefetched_ms == pytest.approx(8.0)


def test_over_budget_stage_is_flagged() -> None:
    turn = TurnMetrics()
    span = turn.add(Stage.VECTOR_MATCH, STAGE_BUDGETS_MS[Stage.VECTOR_MATCH] + 5.0)
    assert span.over_budget
    assert turn.violations() == [span]


def test_prefetched_stage_is_never_over_budget() -> None:
    turn = TurnMetrics()
    span = turn.add(Stage.VECTOR_MATCH, 500.0, prefetched=True)
    assert not span.over_budget


def test_interrupted_turns_are_excluded_from_percentiles() -> None:
    """An interrupted turn is a candidate's choice, not a latency measurement."""
    recorder = LatencyRecorder()
    good = recorder.start_turn()
    good.perceived_ms = 100.0
    bad = recorder.start_turn()
    bad.perceived_ms = 5.0
    bad.interrupted = True
    report = recorder.report()
    assert report.samples == 1
    assert report.p50_ms == pytest.approx(100.0)


def test_latency_report_fails_on_a_pathological_tail() -> None:
    recorder = LatencyRecorder()
    for value in [50.0] * 40 + [900.0]:
        turn = recorder.start_turn()
        turn.perceived_ms = value
    report = recorder.report()
    assert not report.passes, "a 900ms turn must fail the gate"
    assert "FAIL" in report.summary()


def test_stage_timer_records_real_elapsed_time() -> None:
    turn = TurnMetrics()
    with StageTimer(turn, Stage.COGNITION_TTFT):
        time.sleep(0.005)
    assert turn.spans[0].duration_ms >= 4.0


def test_percentile_is_interpolated() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 99) == pytest.approx(99.01, abs=0.1)
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0

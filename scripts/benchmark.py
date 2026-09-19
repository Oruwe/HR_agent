#!/usr/bin/env python3
"""Latency benchmark: retrieval, turnaround, and what speculation actually buys.

Run with ``python scripts/benchmark.py``. No credentials, no services. Numbers
are measured on the offline providers, which model realistic component costs
(30ms cognition TTFT, 12ms to first audio frame) so that what is being measured
is the orchestration -- the part this repository controls -- rather than
somebody else's network.

Every number printed by this script is reproducible on a laptop, which is the
point: a latency claim you cannot re-run is a marketing claim.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.cognition import MockCognition
from app.agent.orchestrator import ScreeningOrchestrator, new_session
from app.config import (
    STAGE_BUDGETS_MS,
    TOTAL_TURNAROUND_BUDGET_MS,
    Settings,
    Stage,
)
from app.schemas.roles import EngineeringRole
from app.storage.retrieval import RETRIEVAL_BUDGET_MS, build_retriever
from app.telemetry.metrics import percentile
from app.voice.moss_engine import MockSpeechEngine, sentence_chunks
from app.voice.ring_buffer import now_ms

PROFILE = """
Six years on model training and serving infrastructure. Ran FSDP across 512
A100s with DeepSpeed ZeRO-3, retuned NCCL collectives and added gradient
checkpointing. Moved serving onto vLLM with paged attention and continuous
batching, wrote fused Triton kernels, used FlashAttention where shapes allowed.
Tuned an HNSW index over 200 million embeddings, trading recall against p99.
"""

ANSWERS = [
    "We ran FSDP across 512 A100s and the collectives were the bottleneck until we retuned NCCL.",
    "Tensor parallelism inside the node because NVLink has the bandwidth, pipeline across nodes.",
    "KV-cache is two times layers times heads times head dim times sequence times batch, in fp16.",
    "We tuned ef until recall at ten hit ninety eight percent with p99 under twelve milliseconds.",
    "The trade-off was index build time, which roughly tripled on every rebuild.",
    "We measured it over a week of production traffic before committing to the rollout.",
]


def _row(label: str, samples: list[float], budget: float | None = None) -> str:
    p50 = percentile(samples, 50)
    p95 = percentile(samples, 95)
    p99 = percentile(samples, 99)
    verdict = ""
    if budget is not None:
        verdict = "  PASS" if p95 <= budget else "  FAIL"
    return (
        f"  {label:<34}{p50:8.2f}{p95:8.2f}{p99:8.2f}{max(samples):8.2f}"
        f"{(budget if budget else 0):8.0f}{verdict}"
    )


def _header(title: str) -> None:
    print(f"\n{title}\n{'-' * 78}")
    print(f"  {'metric':<34}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}{'budget':>8}")


async def bench_retrieval(settings: Settings, iterations: int) -> list[float]:
    retriever = build_retriever(settings)
    await retriever.warm()
    await retriever.search(PROFILE)  # discard the unwarmed first call

    samples: list[float] = []
    for _ in range(iterations):
        await retriever.search(PROFILE, limit=3)
        samples.append(retriever.last_latency_ms)
    return samples


async def bench_turns(settings: Settings, iterations: int, speculate: bool) -> list[float]:
    session = new_session(PROFILE, role=EngineeringRole.AI_ML_SYSTEMS_ENGINEER)
    orchestrator = ScreeningOrchestrator(
        session,
        settings=settings,
        cognition=MockCognition(ttft_ms=30.0, inter_token_ms=0.3),
        max_turns=iterations + 4,
    )
    await orchestrator.open()
    synth = MockSpeechEngine(first_frame_latency_ms=12.0)

    samples: list[float] = []
    for i in range(iterations):
        orchestrator.observe_candidate(ANSWERS[i % len(ANSWERS)])
        if speculate:
            await orchestrator.speculate()
            await asyncio.sleep(settings.speculation_window_ms / 1000.0)

        commit_at = now_ms()
        stream, turn = await orchestrator.commit()
        first_audio: float | None = None
        async for _frame in synth.stream(sentence_chunks(stream)):
            if first_audio is None:
                first_audio = now_ms() - commit_at
        turn.perceived_ms = first_audio or 0.0
        samples.append(turn.perceived_ms)
    return samples


async def main(iterations: int) -> int:
    settings = Settings()

    print("=" * 78)
    print("hr-talent-evaluator -- latency benchmark")
    print("=" * 78)
    backend = "moss" if settings.moss_configured else "embedded fallback"
    print(f"  retrieval backend   : {backend}")
    print(f"  turnaround budget   : {TOTAL_TURNAROUND_BUDGET_MS:.0f} ms")
    print(f"  retrieval budget    : {RETRIEVAL_BUDGET_MS:.0f} ms")
    print(f"  speculation window  : {settings.speculation_window_ms:.0f} ms")
    print(f"  iterations          : {iterations}")

    retrieval = await bench_retrieval(settings, iterations)
    _header("RETRIEVAL  (rubric lookup, milliseconds)")
    print(_row(f"rubric search [{backend}]", retrieval, RETRIEVAL_BUDGET_MS))

    speculated = await bench_turns(settings, iterations, speculate=True)
    cold = await bench_turns(settings, iterations, speculate=False)

    _header("TURNAROUND  (endpoint commit -> first audio byte, milliseconds)")
    print(_row("with speculative turn-taking", speculated, TOTAL_TURNAROUND_BUDGET_MS))
    print(_row("without (sequential baseline)", cold, TOTAL_TURNAROUND_BUDGET_MS))

    saved = statistics.median(cold) - statistics.median(speculated)
    print(f"\n  Speculation saves a median of {saved:.1f} ms per turn "
          f"({saved / max(statistics.median(cold), 1e-9):.0%} of the sequential cost).")

    _header("DECLARED STAGE BUDGET")
    for stage in Stage:
        print(f"  {stage.value:<34}{STAGE_BUDGETS_MS[stage]:>8.1f}")
    print(f"  {'TOTAL':<34}{sum(STAGE_BUDGETS_MS.values()):>8.1f}")

    passed = (
        percentile(retrieval, 95) <= RETRIEVAL_BUDGET_MS
        and percentile(speculated, 95) <= TOTAL_TURNAROUND_BUDGET_MS
    )
    print(f"\n{'=' * 78}\nRESULT: {'PASS' if passed else 'FAIL'}\n{'=' * 78}")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.iterations)))

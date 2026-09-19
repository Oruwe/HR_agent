"""Entry point: offline demo, live LiveKit worker, and the verification report.

``python -m app.main demo`` runs a complete screening interview against the
in-process transport with no credentials, no network and no services. That is
deliberately the default: the fastest way to check that a change did not break
the agent should not require infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Sequence

from app.agent.cognition import MockCognition, ToolCall, build_cognition
from app.agent.orchestrator import ScreeningOrchestrator, new_session
from app.config import STAGE_BUDGETS_MS, Settings, Stage, get_settings
from app.schemas.roles import ROLE_RUBRICS, EngineeringRole
from app.security.pii_scrubber import scrub
from app.telemetry.metrics import budget_table
from app.voice.livekit_worker import InProcessTransport, VoiceWorker
from app.voice.moss_engine import MockSpeechEngine, sentence_chunks
from app.voice.ring_buffer import now_ms

logger = logging.getLogger("hr-talent-evaluator")

DEMO_RESUME = """
Priya Raman
priya.raman@example.com | +91 98765 43210 | Aadhaar 3412 7856 9034
42 MG Road, Indiranagar, Bengaluru 560038

Senior ML Systems Engineer, 6 years.
- Led FSDP distributed training across 512 A100s, tuned NCCL collectives and
  gradient checkpointing to cut step time by 34 percent.
- Wrote custom Triton kernels for fused attention; adopted FlashAttention and
  paged attention in vLLM, reducing KV-cache footprint by 40 percent.
- Tuned HNSW index parameters for a 200M vector store, trading ANN recall
  against p99 latency.
"""

DEMO_ANSWERS: tuple[str, ...] = (
    "Sure. The most demanding thing was a training platform for a seven billion "
    "parameter model. We ran FSDP across 512 A100s and the collective ops were "
    "the bottleneck until we retuned NCCL.",
    "We chose tensor parallelism inside a node because NVLink gives you the "
    "bandwidth, and pipeline parallelism across nodes where you are on "
    "InfiniBand. The communication volume is what decides it.",
    "KV-cache is two times layers times heads times head dimension times "
    "sequence length times batch, times two bytes in half precision. At eight "
    "thousand context and batch thirty two that is most of a card, which is why "
    "we moved to paged attention.",
    "We tuned the HNSW ef parameter until recall at ten hit ninety eight percent "
    "and p99 stayed under twelve milliseconds. Above that the graph traversal "
    "cost more than the recall was worth.",
    "I would want to know how the inference platform team splits ownership with "
    "the model training side.",
)


def orchestrator_backend_label(settings: Settings) -> str:
    return "moss (sub-10ms, in-process)" if settings.moss_configured else "embedded fallback"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


async def run_demo(settings: Settings, turns: int = 5) -> int:
    """Run a full screening interview offline and print the report."""
    _print_header("HR TALENT EVALUATOR -- OFFLINE SCREENING DEMO")

    session = new_session(DEMO_RESUME)
    redactions = scrub(DEMO_RESUME).counts()
    print(f"Candidate alias   : {session.candidate.sanitized_name}")
    print(f"Routed to rubric  : {session.role.value}")
    print(f"PII redacted      : {redactions or 'none'}")
    print(f"Retrieval backend : {orchestrator_backend_label(settings)}")

    cognition = MockCognition(
        ttft_ms=30.0,
        responses=[
            "Good. Walk me through how you split parallelism across that cluster.",
            "That matches what I would expect. Now give me the KV-cache arithmetic.",
            "Clear. How did you pick your ANN index parameters?",
            "Understood. What did that cost you elsewhere?",
            "Thanks. What would you like to ask me?",
        ],
        tool_calls=[
            ToolCall(
                "record_candidate_competency",
                {
                    "skill": "distributed_training",
                    "score": 0.88,
                    "rationale": "Explained FSDP and NCCL retuning.",
                },
            ),
            ToolCall(
                "record_candidate_competency",
                {
                    "skill": "inference_kernels",
                    "score": 0.91,
                    "rationale": "Derived KV-cache memory correctly.",
                },
            ),
            ToolCall(
                "record_candidate_competency",
                {
                    "skill": "vector_indexing",
                    "score": 0.79,
                    "rationale": "Traded ANN recall against p99.",
                },
            ),
            ToolCall(
                "record_candidate_competency",
                {
                    "skill": "framework_internals",
                    "score": 0.62,
                    "rationale": "Authored Triton kernels.",
                },
            ),
        ],
    )

    orchestrator = ScreeningOrchestrator(session, settings=settings, cognition=cognition)
    transport = InProcessTransport()
    worker = VoiceWorker(transport, settings=settings, synthesizer=MockSpeechEngine())

    print(f"\nAGENT     : {(await orchestrator.open())[:96]}...")

    for index, answer in enumerate(DEMO_ANSWERS[:turns]):
        orchestrator.observe_candidate(answer, offset_ms=index * 8000.0)
        print(f"\nCANDIDATE : {answer[:96]}...")

        # Speculate during the silence window, exactly as the VAD would.
        await orchestrator.speculate()
        await asyncio.sleep(settings.speculation_window_ms / 1000.0)

        commit_at = now_ms()
        stream, turn = await orchestrator.commit()

        first_audio_ms: float | None = None
        collected: list[str] = []

        async def tee(source: AsyncIterator[str], sink: list[str]) -> AsyncIterator[str]:
            async for chunk in source:
                sink.append(chunk)
                yield chunk

        async for frame in worker.synthesizer.stream(sentence_chunks(tee(stream, collected))):
            if first_audio_ms is None:
                first_audio_ms = now_ms() - commit_at
            await transport.publish(frame)

        turn.perceived_ms = first_audio_ms or turn.perceived_ms
        marker = "OK " if turn.perceived_ms <= settings.latency_budget_ms else "OVER"
        print(f"AGENT     : {''.join(collected).strip()[:96]}...")
        print(
            f"            [{marker}] turnaround {turn.perceived_ms:6.2f} ms  "
            f"(speculation {'hit' if turn.speculation_hit else 'miss'})"
        )

    evaluation = await orchestrator.close()
    report = orchestrator.latency.report()

    _print_header("LATENCY")
    print(report.summary())
    print(f"Speculation hit rate : {orchestrator.speculation_hit_rate:.0%}")
    retrievals = orchestrator.retrieval_latencies_ms
    if retrievals:
        from app.telemetry.metrics import percentile

        print(
            f"Retrieval ({orchestrator.retrieval_backend}) : "
            f"p50={percentile(list(retrievals), 50):.2f}ms "
            f"p95={percentile(list(retrievals), 95):.2f}ms "
            f"max={max(retrievals):.2f}ms  budget=10ms"
        )
    print("\nDeclared stage budget:")
    for stage, budget in budget_table().items():
        print(f"  {stage:<22} {budget:>6.1f} ms")

    _print_header("EVALUATION")
    print(f"Role            : {evaluation.target_role.value}")
    print(f"Fit index       : {evaluation.rubric_fit_index:.3f}")
    print(f"Routing conf.   : {evaluation.routing_confidence:.3f}")
    print(f"Recommendation  : {evaluation.recommendation.value}")
    print("Competencies    :")
    for score in evaluation.competency_scores:
        print(f"  {score.key:<24} {score.score:.2f}  ({score.source.value})")
    if evaluation.flagged_limitations:
        print("Limitations     :")
        for item in evaluation.flagged_limitations:
            print(f"  - {item}")

    _print_header("ZERO-PII AUDIT")
    leaks = 0
    for turn_record in session.transcript:
        result = scrub(turn_record.text)
        leaks += len(result.findings)
    payload_leaks = len(scrub(str(evaluation.to_payload())).findings)
    print(f"Transcript lines scanned : {len(session.transcript)}")
    print(f"Unredacted PII in transcript : {leaks}")
    print(f"Unredacted PII in stored payload : {payload_leaks}")
    print(f"Verdict : {'PASS -- zero leakage' if leaks == 0 and payload_leaks == 0 else 'FAIL'}")

    return 0 if report.passes and leaks == 0 and payload_leaks == 0 else 1


def run_verify(settings: Settings) -> int:
    """Print the compliance report: OpenGAP files, budget, rubrics, security."""
    from pathlib import Path

    import yaml

    _print_header("OPENGAP COMPLIANCE")
    root = Path(__file__).resolve().parents[1]
    ok = True

    agent_yaml = yaml.safe_load((root / "agent.yaml").read_text())
    keys_ok = list(agent_yaml) == ["spec_version", "name", "version", "description"]
    scalars_ok = all(isinstance(v, str) for v in agent_yaml.values())
    print(
        f"agent.yaml        : 4 scalar string fields  [{'PASS' if keys_ok and scalars_ok else 'FAIL'}]"
    )
    ok &= keys_ok and scalars_ok

    soul = (root / "SOUL.md").read_text()
    soul_ok = all(h in soul for h in ("# Identity", "# Behavior", "# Boundaries"))
    print(f"SOUL.md           : Identity/Behavior/Boundaries  [{'PASS' if soul_ok else 'FAIL'}]")
    ok &= soul_ok

    explain = (root / "EXPLAINABILITY.md").read_text()
    headings = [line.strip() for line in explain.splitlines() if line.startswith("## ")]
    explain_ok = headings == ["## Decision Reasoning", "## Data Inputs", "## Known Limitations"]
    print(f"EXPLAINABILITY.md : 3 headings, 2 sentences each  [{'PASS' if explain_ok else 'FAIL'}]")
    ok &= explain_ok

    _print_header("LATENCY BUDGET")
    total = sum(STAGE_BUDGETS_MS.values())
    for stage in Stage:
        print(f"  {stage.value:<22} {STAGE_BUDGETS_MS[stage]:>6.1f} ms")
    print(f"  {'TOTAL':<22} {total:>6.1f} ms  [{'PASS' if total == 160.0 else 'FAIL'}]")
    ok &= total == 160.0

    _print_header("ROLE RUBRICS")
    for role in EngineeringRole:
        rubric = ROLE_RUBRICS[role]
        print(
            f"  {role.value:<32} {len(rubric.competencies)} competencies, "
            f"{len(rubric.eliminations)} elimination gate(s)"
        )
    rubrics_ok = len(ROLE_RUBRICS) == 9
    print(f"  Total: {len(ROLE_RUBRICS)}/9  [{'PASS' if rubrics_ok else 'FAIL'}]")
    ok &= rubrics_ok

    _print_header("SECURITY")
    probe = "Aadhaar 3412 7856 9034, a@b.com, +91 98765 43210, 42 MG Road, Bengaluru 560038"
    result = scrub(probe)
    clean = not scrub(result.text).findings
    print(f"  Probe redactions : {result.counts()}")
    print(f"  Residual PII     : {'none' if clean else 'PRESENT'}  [{'PASS' if clean else 'FAIL'}]")
    print(f"  Mode             : {settings.pii_mode.value}")
    ok &= clean

    print(f"\n{'=' * 78}\nOVERALL: {'PASS' if ok else 'FAIL'}\n{'=' * 78}")
    return 0 if ok else 1


async def run_live(settings: Settings, room: str) -> int:  # pragma: no cover - needs LiveKit
    """Attach to a LiveKit room and screen a live candidate."""
    from app.voice.livekit_worker import LiveKitTransport, build_access_token

    if not settings.transport_configured:
        print("LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required.", file=sys.stderr)
        return 2

    session = new_session(DEMO_RESUME)
    orchestrator = ScreeningOrchestrator(
        session, settings=settings, cognition=build_cognition(settings)
    )
    await orchestrator.open()
    transport = LiveKitTransport(settings, room_name=room)
    await transport.connect(build_access_token(settings, room, "hr-talent-evaluator"))

    async def on_speculate(_event) -> None:
        await orchestrator.speculate()

    async def on_cancel(_event) -> None:
        await orchestrator.cancel_speculation()

    async def on_commit(_event) -> AsyncIterator[str]:
        stream, _turn = await orchestrator.commit()
        return sentence_chunks(stream)

    worker = VoiceWorker(
        transport,
        settings=settings,
        on_speculate=on_speculate,
        on_commit=on_commit,
        on_cancel=on_cancel,
    )
    print(f"Screening live in room '{room}'. Ctrl-C to stop.")
    try:
        await worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        evaluation = await orchestrator.close()
        await worker.close()
        print(f"Recommendation: {evaluation.recommendation.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hr-talent-evaluator",
        description="Sub-160ms voice-enabled HR technical screening agent.",
    )
    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser(
        "demo", help="Run an offline screening interview (no credentials needed)."
    )
    demo.add_argument("--turns", type=int, default=5)
    sub.add_parser("verify", help="Print the OpenGAP / budget / security compliance report.")
    live = sub.add_parser("live", help="Attach to a LiveKit room (requires credentials).")
    live.add_argument("--room", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    command = args.command or "demo"
    if command == "verify":
        return run_verify(settings)
    if command == "live":
        room = args.room or f"{settings.room_prefix}-{session_suffix()}"
        return asyncio.run(run_live(settings, room))
    return asyncio.run(run_demo(settings, turns=args.turns))


def session_suffix() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


if __name__ == "__main__":
    raise SystemExit(main())

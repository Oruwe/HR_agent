"""Session tracing with a hard PII gate on every payload.

Observability is an egress boundary like any other -- arguably the most
dangerous one, because tracing payloads are verbose by design, retained for
months, and replicated to a third party. Every string that reaches a span here
goes through the scrubber first, without exception and without a bypass flag.

Tracing is always on locally. The in-process :class:`SpanRecorder` runs whether
or not Langfuse is configured, so latency assertions and post-call audits work
identically in CI, in a local demo, and in production.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from app.config import TOTAL_TURNAROUND_BUDGET_MS, Settings, Stage, get_settings
from app.security.pii_scrubber import sanitize_for_egress
from app.security.token_guard import redact_credentials
from app.telemetry.metrics import LatencyRecorder, TurnMetrics

logger = logging.getLogger(__name__)


def _safe(payload: Any) -> Any:
    """Scrub credentials then PII. Order matters -- see token_guard.sanitize."""
    if isinstance(payload, str):
        return sanitize_for_egress(redact_credentials(payload))
    if isinstance(payload, Mapping):
        return {_safe(k): _safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_safe(v) for v in payload]
    return payload


@dataclass(slots=True)
class Span:
    name: str
    started_ms: float
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"

    @property
    def latency_exceeded(self) -> bool:
        return bool(self.metadata.get("latency_exceeded", False))


@dataclass(slots=True)
class SpanRecorder:
    """In-process span log. Always active; the source of truth for assertions."""

    session_id: str
    spans: list[Span] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def record(
        self, name: str, duration_ms: float, metadata: Mapping[str, Any] | None = None
    ) -> Span:
        span = Span(
            name=name,
            started_ms=(time.perf_counter() - self.started_at) * 1000.0,
            duration_ms=duration_ms,
            metadata=dict(_safe(dict(metadata or {}))),
        )
        self.spans.append(span)
        return span

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def violations(self) -> list[Span]:
        return [s for s in self.spans if s.latency_exceeded]


class LangfuseTracer:
    """Root trace per screening session, one span per pipeline stage.

    The Langfuse client is optional and failures are swallowed on purpose. A
    telemetry outage is not a reason to drop a candidate's interview, and an
    exception raised from a tracing call inside a 20ms audio callback would do
    exactly that.
    """

    #: Span names, fixed to the pipeline stages so dashboards stay stable.
    SPAN_VAD = "vad_silence_detection"
    SPAN_SPEECH = "speech_to_speech_turnaround"
    SPAN_RETRIEVAL = "qdrant_similarity_lookup"
    SPAN_COGNITION = "llm_cognitive_generation"
    SPAN_EGRESS = "audio_egress_push"

    STAGE_SPANS: ClassVar[dict[Stage, str]] = {
        Stage.VAD_ENDPOINT: SPAN_VAD,
        Stage.VECTOR_MATCH: SPAN_RETRIEVAL,
        Stage.COGNITION_TTFT: SPAN_COGNITION,
        Stage.SPEECH_SYNTHESIS: SPAN_SPEECH,
        Stage.TRANSPORT_EGRESS: SPAN_EGRESS,
    }

    def __init__(
        self,
        session_id: str | None = None,
        settings: Settings | None = None,
        recorder: SpanRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_id = session_id or str(uuid.uuid4())
        self.recorder = recorder or SpanRecorder(session_id=self.session_id)
        self._client: Any | None = None
        self._trace: Any | None = None
        if self.settings.telemetry_configured:
            self._connect()

    @property
    def remote_enabled(self) -> bool:
        return self._client is not None

    def _connect(self) -> None:
        try:  # pragma: no cover - requires the optional dependency
            from langfuse import Langfuse  # type: ignore[import-not-found]

            self._client = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Langfuse unavailable (%s); tracing locally only.", type(exc).__name__)
            self._client = None

    # -- trace lifecycle -----------------------------------------------------

    def start_session(self, role: str, candidate_alias: str) -> None:
        """Open the root trace. Only the pseudonym is ever sent."""
        metadata = _safe(
            {
                "role": role,
                "candidate": candidate_alias,
                "budget_ms": TOTAL_TURNAROUND_BUDGET_MS,
                "environment": self.settings.environment,
            }
        )
        self.recorder.record("session_start", 0.0, metadata)
        if self._client is None:
            return
        try:  # pragma: no cover - network path
            self._trace = self._client.trace(
                id=self.session_id, name="hr_screening_session", metadata=metadata
            )
        except Exception:
            self._trace = None

    def span(
        self, name: str, duration_ms: float, metadata: Mapping[str, Any] | None = None
    ) -> Span:
        payload = dict(metadata or {})
        if duration_ms > TOTAL_TURNAROUND_BUDGET_MS:
            payload["latency_exceeded"] = True
        span = self.recorder.record(name, duration_ms, payload)
        if self._trace is not None:  # pragma: no cover - network path
            # A telemetry failure must never surface inside a 20ms audio
            # callback; losing a span is strictly better than losing the call.
            with contextlib.suppress(Exception):
                self._trace.span(
                    name=name,
                    metadata=span.metadata,
                    start_time=None,
                    end_time=None,
                    level="WARNING" if span.latency_exceeded else "DEFAULT",
                )
        return span

    def record_turn(self, turn: TurnMetrics) -> None:
        """Emit one span per pipeline stage plus a turn-level rollup."""
        for stage_span in turn.spans:
            name = self.STAGE_SPANS.get(stage_span.stage, stage_span.stage.value)
            self.span(
                name,
                stage_span.duration_ms,
                {
                    "turn": turn.turn_index,
                    "budget_ms": stage_span.budget_ms,
                    "prefetched": stage_span.prefetched,
                    "over_budget": stage_span.over_budget,
                },
            )
        self.span(
            "turn",
            turn.perceived_ms,
            {
                "turn": turn.turn_index,
                "perceived_ms": round(turn.perceived_ms, 3),
                "speculation_hit": turn.speculation_hit,
                "interrupted": turn.interrupted,
                "latency_exceeded": turn.latency_exceeded,
                "breakdown": turn.breakdown(),
            },
        )

    def record_transcript(self, speaker: str, text: str, turn_index: int) -> Span:
        """Attach a transcript line. Scrubbed twice -- here and at construction."""
        return self.span(
            "transcript",
            0.0,
            {"speaker": speaker, "text": text, "turn": turn_index},
        )

    def end_session(self, recorder: LatencyRecorder, outcome: str) -> dict[str, Any]:
        report = recorder.report()
        summary = _safe(
            {
                "outcome": outcome,
                "turns": report.samples,
                "p50_ms": report.p50_ms,
                "p95_ms": report.p95_ms,
                "p99_ms": report.p99_ms,
                "max_ms": report.max_ms,
                "compliance": round(report.compliance, 4),
                "budget_ms": report.budget_ms,
                "passes_gate": report.passes,
            }
        )
        self.recorder.record("session_end", 0.0, summary)
        if self._client is not None:  # pragma: no cover - network path
            with contextlib.suppress(Exception):
                if self._trace is not None:
                    self._trace.update(output=summary)
                self._client.flush()
        return summary


__all__ = ["LangfuseTracer", "Span", "SpanRecorder"]

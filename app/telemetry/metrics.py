"""Latency accounting: per-stage spans, percentiles, and budget compliance.

A mean latency number is close to useless for conversational audio. The
experience is governed by the tail: a p50 of 90ms with a p99 of 700ms feels
broken, because the candidate notices the one turn in a hundred where the agent
appears to freeze, not the ninety-nine where it did not. Everything here is
therefore built around percentiles and explicit budget violations.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType

from app.config import (
    PREFETCHABLE_STAGES,
    STAGE_BUDGETS_MS,
    TOTAL_TURNAROUND_BUDGET_MS,
    Stage,
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 100].

    Implemented here rather than pulled from numpy so that the metrics module
    has no array dependency and can be imported by a lightweight sidecar.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (max(0.0, min(100.0, q)) / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[int(pos)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


@dataclass(slots=True)
class StageSpan:
    stage: Stage
    duration_ms: float
    #: True when this stage ran while the candidate was still speaking, so its
    #: cost is hidden from the user-perceived turnaround.
    prefetched: bool = False

    @property
    def budget_ms(self) -> float:
        return STAGE_BUDGETS_MS.get(self.stage, 0.0)

    @property
    def over_budget(self) -> bool:
        return not self.prefetched and self.duration_ms > self.budget_ms


@dataclass(slots=True)
class TurnMetrics:
    """One conversational turn's latency breakdown."""

    turn_index: int = 0
    spans: list[StageSpan] = field(default_factory=list)
    #: Wall time the candidate actually waited: endpoint commit -> first audio
    #: byte handed to the transport.
    perceived_ms: float = 0.0
    speculation_hit: bool = False
    interrupted: bool = False

    def add(self, stage: Stage, duration_ms: float, prefetched: bool = False) -> StageSpan:
        span = StageSpan(stage=stage, duration_ms=duration_ms, prefetched=prefetched)
        self.spans.append(span)
        return span

    @property
    def accounted_ms(self) -> float:
        """Sum of stages that are on the critical path."""
        return sum(s.duration_ms for s in self.spans if not s.prefetched)

    @property
    def prefetched_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.prefetched)

    @property
    def latency_exceeded(self) -> bool:
        return self.perceived_ms > TOTAL_TURNAROUND_BUDGET_MS

    def violations(self) -> list[StageSpan]:
        return [s for s in self.spans if s.over_budget]

    def breakdown(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for span in self.spans:
            out[span.stage.value] = round(out.get(span.stage.value, 0.0) + span.duration_ms, 4)
        return out


class StageTimer:
    """Context manager that records one stage into a :class:`TurnMetrics`.

    Uses ``perf_counter`` rather than wall-clock: an NTP step during a call
    would otherwise produce negative durations, and a negative latency in a
    dashboard destroys trust in every other number on it.
    """

    __slots__ = ("_prefetched", "_stage", "_start", "_turn", "span")

    def __init__(self, turn: TurnMetrics, stage: Stage, prefetched: bool | None = None) -> None:
        self._turn = turn
        self._stage = stage
        self._prefetched = stage in PREFETCHABLE_STAGES if prefetched is None else prefetched
        self._start = 0.0
        self.span: StageSpan | None = None

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000.0
        self.span = self._turn.add(self._stage, elapsed, prefetched=self._prefetched)


@dataclass(slots=True)
class LatencyReport:
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    budget_ms: float
    within_budget: int
    compliance: float
    stage_p95_ms: dict[str, float] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        """The release gate: p95 inside budget and no turn beyond 2x budget.

        p95 rather than p99 because a screening call has on the order of twenty
        turns -- p99 over twenty samples is a single observation and is mostly
        noise. The 2x hard ceiling is what catches the pathological tail that
        p95 would otherwise hide.
        """
        return self.p95_ms <= self.budget_ms and self.max_ms <= self.budget_ms * 2

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"[{verdict}] n={self.samples} p50={self.p50_ms:.1f}ms "
            f"p95={self.p95_ms:.1f}ms p99={self.p99_ms:.1f}ms "
            f"max={self.max_ms:.1f}ms budget={self.budget_ms:.0f}ms "
            f"compliance={self.compliance:.1%}"
        )


class LatencyRecorder:
    """Accumulates turns across a session and reports the distribution."""

    def __init__(self, budget_ms: float = TOTAL_TURNAROUND_BUDGET_MS) -> None:
        self.budget_ms = budget_ms
        self._turns: list[TurnMetrics] = []

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[TurnMetrics, ...]:
        return tuple(self._turns)

    def start_turn(self) -> TurnMetrics:
        turn = TurnMetrics(turn_index=len(self._turns))
        self._turns.append(turn)
        return turn

    def add(self, turn: TurnMetrics) -> None:
        self._turns.append(turn)

    def extend(self, turns: Iterable[TurnMetrics]) -> None:
        self._turns.extend(turns)

    def perceived(self) -> list[float]:
        # Interrupted turns are excluded: the candidate chose to stop them, so
        # their truncated duration is not a latency measurement at all and
        # including them would flatter the distribution.
        return [t.perceived_ms for t in self._turns if not t.interrupted]

    def report(self) -> LatencyReport:
        samples = self.perceived()
        stage_values: dict[str, list[float]] = defaultdict(list)
        for turn in self._turns:
            for stage, value in turn.breakdown().items():
                stage_values[stage].append(value)

        within = sum(1 for s in samples if s <= self.budget_ms)
        return LatencyReport(
            samples=len(samples),
            p50_ms=round(percentile(samples, 50), 4),
            p95_ms=round(percentile(samples, 95), 4),
            p99_ms=round(percentile(samples, 99), 4),
            max_ms=round(max(samples), 4) if samples else 0.0,
            mean_ms=round(sum(samples) / len(samples), 4) if samples else 0.0,
            budget_ms=self.budget_ms,
            within_budget=within,
            compliance=(within / len(samples)) if samples else 1.0,
            stage_p95_ms={
                stage: round(percentile(values, 95), 4)
                for stage, values in sorted(stage_values.items())
            },
        )

    def compliance(self) -> float:
        return self.report().compliance


def budget_table() -> Mapping[str, float]:
    """The declared budget, for printing in reports and README verification."""
    return {stage.value: budget for stage, budget in STAGE_BUDGETS_MS.items()}


__all__ = [
    "LatencyRecorder",
    "LatencyReport",
    "StageSpan",
    "StageTimer",
    "TurnMetrics",
    "budget_table",
    "percentile",
]

"""The cognitive turn loop: speculation, retrieval, generation, and scoring.

This is where the 160ms budget is actually won or lost, so it is worth stating
the mechanism plainly.

A conventional voice agent runs strictly in sequence once the candidate stops
speaking: wait out the endpoint silence, then retrieve, then generate, then
synthesise. Every one of those costs is paid *after* the candidate is already
waiting, which is why the usual number is over a second.

This orchestrator instead starts work during the silence it is still measuring.
At 120ms of silence the VAD emits SPECULATE and we begin retrieval and
generation on a draft turn. At 250ms it emits TURN_COMMIT and, in the common
case, the first tokens are already buffered -- so the candidate-perceived
latency is the time to hand the first synthesised frame to the transport, not
the time to produce the answer. If the candidate resumes talking in between,
the draft is aborted and we have lost nothing but some compute.

The honest accounting: speculation converts the 130ms speculation window into
free budget. It does not make a network round trip faster, and a turn whose
generation takes longer than the window still pays the difference. That is why
:class:`TurnMetrics` records both the perceived wait and the true stage costs --
a system that only measured the flattering number would hide its own
regressions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.agent.cognition import (
    TOOL_SCHEMAS,
    ChunkType,
    CognitionProvider,
    Message,
    ToolCall,
    build_cognition,
)
from app.agent.conversation_prompts import (
    build_fast_system_prompt,
    build_system_prompt,
    closing,
    greeting,
)
from app.agent.interview_flow import InterviewFlow
from app.config import Settings, Stage, get_settings
from app.schemas.candidate import (
    CandidateProfile,
    ScreeningSession,
    Speaker,
    TranscriptTurn,
)
from app.schemas.evaluation import (
    CandidateEvaluation,
    CompetencyScore,
    EvidenceSource,
    RoleFit,
    derive_recommendation,
)
from app.schemas.roles import (
    EngineeringRole,
    match_confidence,
    rank_roles,
    rubric_for,
    score_role,
)
from app.storage.qdrant_client import HybridVectorStore
from app.storage.retrieval import RETRIEVAL_BUDGET_MS, RubricRetriever, build_retriever
from app.telemetry.langfuse_tracer import LangfuseTracer
from app.telemetry.metrics import LatencyRecorder, StageTimer, TurnMetrics

logger = logging.getLogger(__name__)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class SpeculativeDraft:
    """A response being generated before the turn is confirmed.

    Tokens are pumped into a queue by a background task so that the moment
    TURN_COMMIT arrives, the consumer can start draining immediately rather
    than awaiting the provider's first byte.
    """

    provider: CognitionProvider
    queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None
    started_ms: float = field(default_factory=_now_ms)
    first_token_ms: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    cancelled: bool = False
    finished: bool = False

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_ms is None:
            return None
        return self.first_token_ms - self.started_ms

    @property
    def ready(self) -> bool:
        """True when at least one token is already buffered."""
        return not self.queue.empty()

    async def cancel(self) -> None:
        self.cancelled = True
        self.provider.abort()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            # Suppress everything: this is best-effort teardown of a draft the
            # candidate has already made irrelevant, and a failure here must not
            # propagate into the turn that replaced it.
            with contextlib.suppress(BaseException):
                await self.task


@dataclass(slots=True)
class TurnResult:
    """What one committed turn produced."""

    text: str
    metrics: TurnMetrics
    tool_calls: tuple[ToolCall, ...] = ()
    speculation_hit: bool = False


class ScreeningOrchestrator:
    """Drives one screening session end to end."""

    def __init__(
        self,
        session: ScreeningSession,
        settings: Settings | None = None,
        store: HybridVectorStore | None = None,
        cognition: CognitionProvider | None = None,
        tracer: LangfuseTracer | None = None,
        flow: InterviewFlow | None = None,
        retriever: RubricRetriever | None = None,
        max_turns: int = 12,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session
        self.store = store or HybridVectorStore(self.settings)
        self.retriever = retriever or build_retriever(self.settings, self.store)
        self.cognition = cognition or build_cognition(self.settings)
        self.tracer = tracer or LangfuseTracer(session.session_id, self.settings)
        self.flow = flow or InterviewFlow(role=session.role, max_turns=max_turns)
        self.latency = LatencyRecorder(self.settings.latency_budget_ms)

        self._draft: SpeculativeDraft | None = None
        self._pending_context: str = ""
        self._history: list[Message] = []
        self._speculation_hits = 0
        self._speculation_attempts = 0
        self._terminate_reason: str = ""
        self._retrieval_samples: list[float] = []

    # -- lifecycle -----------------------------------------------------------

    async def open(self) -> str:
        """Warm every index and return the greeting.

        Warming happens here, behind a fixed greeting, so the first real turn
        never pays for index construction. The greeting itself is a constant
        rather than a generated line for the same reason: it is the only turn
        with no prior audio to hide latency behind, so generating it would make
        the interview's first impression its slowest response.
        """
        self.store.ensure_collection()
        await self.retriever.warm()
        self.tracer.start_session(self.session.role.value, self.session.candidate.sanitized_name)
        text = greeting(self.session.role)
        self._append(Speaker.AGENT, text, 0.0)
        self.flow.advance()
        return text

    def observe_candidate(self, text: str, offset_ms: float = 0.0) -> None:
        """Record what the candidate just said. Scrubbed on the way in."""
        self._append(Speaker.CANDIDATE, text, offset_ms)
        self._history.append(Message(role="user", content=self.session.transcript[-1].text))

    def _append(
        self, speaker: Speaker, text: str, offset_ms: float, turnaround: float | None = None
    ) -> None:
        turn = TranscriptTurn.sanitised(speaker, text, offset_ms, turnaround)
        self.session.record(turn)
        self.tracer.record_transcript(speaker.value, turn.text, len(self.session.transcript))

    # -- speculation ---------------------------------------------------------

    async def speculate(self) -> SpeculativeDraft | None:
        """Start a draft response while the candidate may still be speaking."""
        if self.flow.finished:
            return None
        await self._discard_draft()
        self._speculation_attempts += 1

        turn = TurnMetrics(turn_index=len(self.latency))
        # Retrieval happens here, inside the speculation window, which is why it
        # is marked prefetched: its cost never reaches the candidate. It is
        # still measured against the 10ms stage budget -- prefetched is not a
        # licence to be slow, it is a statement about who waits.
        with StageTimer(turn, Stage.VECTOR_MATCH, prefetched=True):
            self._pending_context = await self._retrieve_context()

        draft = SpeculativeDraft(provider=self.cognition)
        draft.task = asyncio.create_task(self._pump(draft))
        self._draft = draft
        self._draft_turn = turn
        return draft

    async def _retrieve_context(self) -> str:
        """Ask the retrieval runtime which competency this answer evidences.

        This is the call the 10ms budget is written for, and it is why Moss is
        on the hot path: an in-process search runtime answers in under ten
        milliseconds, whereas a round trip to a hosted vector database spends
        that much on the network before it has looked at anything.
        """
        if self.settings.fast_path:
            return ""
        evidence = self.session.candidate_evidence()
        if not evidence.strip():
            return ""
        try:
            hits = await self.retriever.search(evidence, limit=2)
        except Exception as exc:  # pragma: no cover - retrieval must never block a turn
            logger.warning("Retrieval failed (%s); continuing without context.", type(exc).__name__)
            return ""

        self._retrieval_samples.append(self.retriever.last_latency_ms)
        if self.retriever.last_latency_ms > RETRIEVAL_BUDGET_MS:
            logger.warning(
                "Retrieval took %.2fms against a %.0fms budget (backend=%s).",
                self.retriever.last_latency_ms,
                RETRIEVAL_BUDGET_MS,
                self.retriever.backend,
            )
        if not hits:
            return ""
        labels = [hit.competency_label for hit in hits if hit.competency_label]
        if not labels:
            return ""
        return "Their answer currently reads as evidence for: " + "; ".join(labels) + "."

    async def _pump(self, draft: SpeculativeDraft) -> None:
        """Drain the provider into the draft queue, recording time-to-first-token."""
        prompt_builder = build_fast_system_prompt if self.settings.fast_path else build_system_prompt
        system = prompt_builder(
            self.flow.role,
            candidate_alias=self.session.candidate.sanitized_name,
            covered=self.flow.covered(),
            remaining_turns=self.flow.remaining_turns,
        )
        directive = self.flow.directive()
        if self._pending_context:
            directive = f"{self._pending_context}\n{directive}"

        history = self._history[-2:] if self.settings.fast_path else self._history
        messages = [*history, Message(role="user", content=directive)]
        try:
            tools = () if self.settings.fast_path else TOOL_SCHEMAS
            async for chunk in draft.provider.stream(system, messages, tools):
                if draft.cancelled:
                    return
                if chunk.type is ChunkType.TEXT and chunk.text:
                    if draft.first_token_ms is None:
                        draft.first_token_ms = _now_ms()
                    await draft.queue.put(chunk.text)
                elif chunk.type is ChunkType.TOOL_CALL and chunk.tool_call is not None:
                    draft.tool_calls.append(chunk.tool_call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning("Generation failed (%s)", type(exc).__name__)
        finally:
            draft.finished = True
            await draft.queue.put(None)

    async def cancel_speculation(self) -> None:
        """The candidate resumed. Throw the draft away."""
        await self._discard_draft()

    async def _discard_draft(self) -> None:
        if self._draft is not None:
            await self._draft.cancel()
            self._draft = None

    # -- commit --------------------------------------------------------------

    async def commit(self) -> tuple[AsyncIterator[str], TurnMetrics]:
        """Confirm the turn and return a text stream for synthesis.

        Returns immediately with an async iterator so the caller can begin
        synthesising the first chunk while the rest is still generating. The
        returned :class:`TurnMetrics` is filled in as the stream is consumed;
        ``perceived_ms`` is final once the first chunk has been yielded.
        """
        commit_at = _now_ms()
        turn = getattr(self, "_draft_turn", None) or TurnMetrics(turn_index=len(self.latency))
        turn.turn_index = len(self.latency)

        draft = self._draft
        if draft is None or draft.cancelled:
            # No usable draft: pay the full generation cost now. This is the
            # slow path, and it is measured as such rather than hidden.
            draft = SpeculativeDraft(provider=self.cognition)
            draft.task = asyncio.create_task(self._pump(draft))
            self._draft = draft
            turn.speculation_hit = False
        else:
            turn.speculation_hit = draft.ready
            if draft.ready:
                self._speculation_hits += 1

        self.latency.add(turn)
        stream = self._drain(draft, turn, commit_at)
        return stream, turn

    async def _drain(
        self, draft: SpeculativeDraft, turn: TurnMetrics, commit_at: float
    ) -> AsyncIterator[str]:
        """Yield text chunks, closing out the turn's latency accounting."""
        collected: list[str] = []
        first = True
        while True:
            chunk = await draft.queue.get()
            if chunk is None:
                break
            if first:
                # The number that matters: commit -> first speakable text.
                wait_ms = _now_ms() - commit_at
                turn.add(Stage.COGNITION_TTFT, wait_ms)
                turn.perceived_ms = wait_ms
                first = False
            collected.append(chunk)
            yield chunk

        text = "".join(collected).strip()
        self._finish_turn(turn, draft, text, commit_at)

    def _finish_turn(
        self, turn: TurnMetrics, draft: SpeculativeDraft, text: str, commit_at: float
    ) -> None:
        for call in draft.tool_calls:
            self._apply_tool_call(call)

        if text:
            self._append(Speaker.AGENT, text, commit_at, turn.perceived_ms)
            self._history.append(Message(role="model", content=text))

        self.flow.advance()
        self.tracer.record_turn(turn)
        self._draft = None
        self._draft_turn = None

    # -- tools ---------------------------------------------------------------

    def _apply_tool_call(self, call: ToolCall) -> None:
        """Apply a model tool call. Unknown or malformed calls are ignored.

        Tolerant by design: a malformed tool call is a model glitch, and raising
        here would end a live interview over a bad float.
        """
        try:
            if call.name == "record_candidate_competency":
                skill = str(call.arguments.get("skill", "")).strip()
                if skill:
                    self.flow.record(skill, float(call.arguments.get("score", 0.0)))
            elif call.name == "trigger_role_transition":
                target = str(call.arguments.get("next_scenario", "")).strip()
                if target in EngineeringRole.__members__:
                    self.flow.switch_role(
                        EngineeringRole[target], str(call.arguments.get("reason", ""))
                    )
                    self.session.role = self.flow.role
            elif call.name == "terminate_screening_session":
                reason = str(call.arguments.get("reason", "unspecified"))
                self._terminate_reason = reason
                self.flow.terminate(reason)
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring malformed tool call %s (%s)", call.name, exc)

    # -- evaluation ----------------------------------------------------------

    def build_evaluation(self) -> CandidateEvaluation:
        """Produce the dossier.

        Two independent evidence channels are combined:

        * the **deterministic rubric match** over the resume plus everything the
          candidate said, which is reproducible and auditable; and
        * the **interviewer's recorded scores**, which capture depth that
          keyword evidence cannot -- whether they could actually explain the
          thing they listed.

        The interviewer's score wins where it exists, because a live
        demonstration is stronger evidence than a mention. The matcher fills the
        gaps so a competency that was never reached still carries its resume
        signal rather than scoring zero.
        """
        evidence = self.session.candidate_evidence()
        ranked = rank_roles(evidence) if evidence.strip() else ()
        role = self.flow.role
        rubric = rubric_for(role)
        match = score_role(evidence, role)
        matched = {c.key: c.score for c in match.competencies}

        scores: list[CompetencyScore] = []
        weighted = 0.0
        for competency in rubric.competencies:
            live = self.flow.scores.get(competency.key)
            derived = matched.get(competency.key, 0.0)
            if live is not None:
                value, source = live, EvidenceSource.LIVE_INTERVIEW
            else:
                value, source = derived, EvidenceSource.RESUME
            weighted += value * competency.weight
            scores.append(
                CompetencyScore(
                    key=competency.key,
                    label=competency.label,
                    score=round(value, 4),
                    weight=competency.weight,
                    source=source,
                    rationale=(
                        f"{len(match.competencies[0].matched_signals) if match.competencies else 0} "
                        f"rubric signals observed"
                        if source is EvidenceSource.RESUME
                        else "Scored from the candidate's live explanation."
                    ),
                )
            )

        fit = weighted / rubric.total_weight if rubric.total_weight else 0.0
        score_map = {s.key: s.score for s in scores}
        recommendation, eliminations = derive_recommendation(
            role, fit, score_map, turns_completed=self.flow.turns
        )

        limitations = [e.description for e in eliminations]
        for competency in rubric.competencies:
            if score_map.get(competency.key, 0.0) < 0.3:
                limitations.append(f"Limited evidence for {competency.label.lower()}.")

        report = self.latency.report()
        return CandidateEvaluation(
            candidate_id=self.session.candidate.candidate_id,
            sanitized_name=self.session.candidate.sanitized_name,
            session_id=self.session.session_id,
            target_role=role,
            rubric_fit_index=round(min(1.0, max(0.0, fit)), 4),
            routing_confidence=round(match_confidence(ranked), 4) if ranked else 0.0,
            competency_scores=scores,
            alternate_fits=[RoleFit.from_match(m) for m in ranked[:3]],
            eliminations=eliminations,
            flagged_limitations=limitations,
            recommendation=recommendation,
            turns_completed=self.flow.turns,
            latency_compliance=round(report.compliance, 4),
        )

    async def close(self) -> CandidateEvaluation:
        """Finish the session: persist the dossier and flush telemetry."""
        await self._discard_draft()
        evaluation = self.build_evaluation()
        try:
            self.store.upsert_candidate(evaluation.to_payload(), self.session.candidate_evidence())
        except Exception as exc:  # pragma: no cover - persistence must not lose the dossier
            logger.error(
                "Failed to persist evaluation for %s (%s). The dossier is still "
                "returned to the caller and must be replayed.",
                evaluation.sanitized_name,
                type(exc).__name__,
            )
        self.tracer.end_session(self.latency, evaluation.recommendation.value)
        return evaluation

    # -- introspection -------------------------------------------------------

    @property
    def retrieval_latencies_ms(self) -> tuple[float, ...]:
        """Every measured retrieval, for the benchmark and the latency report."""
        return tuple(self._retrieval_samples)

    @property
    def retrieval_backend(self) -> str:
        return self.retriever.backend

    @property
    def speculation_hit_rate(self) -> float:
        if not self._speculation_attempts:
            return 0.0
        return self._speculation_hits / self._speculation_attempts

    def closing_line(self) -> str:
        return closing()


def new_session(
    resume_text: str,
    role: EngineeringRole | None = None,
    candidate_id: str | None = None,
) -> ScreeningSession:
    """Build a session from a raw resume, scrubbing and routing deterministically."""
    profile = CandidateProfile.from_resume(resume_text, candidate_id=candidate_id)
    resolved = role
    if resolved is None:
        ranked = rank_roles(profile.resume_text)
        resolved = ranked[0].role if ranked else EngineeringRole.FULL_STACK_PRODUCT_ENGINEER
    return ScreeningSession(candidate=profile, role=resolved)


__all__ = [
    "ScreeningOrchestrator",
    "SpeculativeDraft",
    "TurnResult",
    "new_session",
]

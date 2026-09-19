"""Candidate-facing endpoints: create a session, exchange turns, close it out.

This is the HTTP surface over :class:`app.agent.orchestrator.ScreeningOrchestrator`.
Each turn reuses exactly the speculate -> wait -> commit sequence the offline
demo (`app.main.run_demo`) and the LiveKit worker both already use -- this
router does not reimplement the turn-taking logic, it drives the same one
over REST instead of over an audio VAD.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session as DbSession

from app.agent.orchestrator import ScreeningOrchestrator, new_session
from app.api.audio import frames_to_wav_b64
from app.api.routes_health import (
    ACTIVE_SESSIONS,
    SESSIONS_CLOSED_TOTAL,
    SESSIONS_CREATED_TOTAL,
    STT_ERRORS_TOTAL,
    STT_REQUESTS_TOTAL,
    TURN_TURNAROUND_MS,
    TURNS_OVER_BUDGET_TOTAL,
)
from app.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    EvaluationResponse,
    TurnRequest,
    TurnResponse,
)
from app.api.security import extract_bearer_token, hash_token, new_token, verify_token
from app.api.session_store import SessionStore, get_session_store
from app.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models import EvaluationRecord, SessionRecord, TurnRecord
from app.security.pii_scrubber import scrub
from app.voice.moss_engine import MockSpeechEngine, build_synthesizer, sentence_chunks
from app.voice.ring_buffer import now_ms
from app.voice.stt_engine import build_transcriber

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB: a few minutes of PCM/WebM, generous for one answer


async def _authorized_orchestrator(
    session_id: str,
    token: str,
    db: DbSession,
    store: SessionStore,
) -> ScreeningOrchestrator:
    record = db.get(SessionRecord, session_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such session.")
    if not verify_token(token, record.token_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session token.")
    if record.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, "Session is already closed.")
    orchestrator = await store.get(session_id)
    if orchestrator is None:
        # The session exists in the DB but this process never opened it (e.g.
        # a restart, or a second replica behind a load balancer). There is no
        # way to resume mid-generation state, so this is a clean, honest 410
        # rather than a silent restart that would confuse the candidate.
        raise HTTPException(
            status.HTTP_410_GONE,
            "This session is not live on this server (it may have restarted). Start a new session.",
        )
    return orchestrator


@router.post("", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    db: DbSession = Depends(get_db),
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> CreateSessionResponse:
    scrub_result = scrub(body.resume_text)
    session = new_session(
        body.resume_text, role=body.target_role
    )  # scrubs internally via CandidateProfile.from_resume
    orchestrator = ScreeningOrchestrator(session, settings=settings)
    greeting_text = await orchestrator.open()

    token = new_token()
    record = SessionRecord(
        session_id=session.session_id,
        candidate_id=session.candidate.candidate_id,
        sanitized_name=session.candidate.sanitized_name,
        role=session.role.value,
        status="active",
        token_hash=hash_token(token),
        retrieval_backend=orchestrator.retrieval_backend,
    )
    db.add(record)
    db.add(
        TurnRecord(
            session_id=session.session_id,
            speaker="agent",
            text=greeting_text,
            offset_ms=0.0,
        )
    )
    db.commit()
    await store.put(orchestrator)
    SESSIONS_CREATED_TOTAL.inc()
    ACTIVE_SESSIONS.set(len(store))

    audio_b64 = None
    try:
        synthesizer = MockSpeechEngine()

        async def _one_chunk():
            yield greeting_text

        audio_b64 = await frames_to_wav_b64(synthesizer.stream(sentence_chunks(_one_chunk())))
    except Exception:
        logger.warning("Greeting synthesis failed; continuing text-only.")

    return CreateSessionResponse(
        session_id=session.session_id,
        session_token=token,
        sanitized_name=session.candidate.sanitized_name,
        role=session.role,
        redacted=bool(scrub_result.counts()),
        greeting_text=greeting_text,
        greeting_audio_b64=audio_b64,
        retrieval_backend=orchestrator.retrieval_backend,
    )


async def _run_turn(
    orchestrator: ScreeningOrchestrator, candidate_text: str, settings: Settings
) -> tuple[str, float, bool, bool]:
    """Speculate -> wait the speculation window -> commit -> synthesize.

    Identical sequence to ``app.main.run_demo``, just without a terminal to
    print to. Returns (agent_text, turnaround_ms, within_budget, speculation_hit).
    """
    orchestrator.observe_candidate(candidate_text, offset_ms=now_ms())
    await orchestrator.speculate()
    await asyncio.sleep(settings.speculation_window_ms / 1000.0)

    stream, turn = await orchestrator.commit()

    collected: list[str] = []

    async def _tee():
        async for chunk in stream:
            collected.append(chunk)
            yield chunk

    synthesizer = build_synthesizer(settings)
    # Draining the synthesizer stream is what drives `_tee` (and therefore the
    # cognition stream) to completion and finalizes turn.perceived_ms.
    async for _frame in synthesizer.stream(sentence_chunks(_tee())):
        pass

    text = "".join(collected).strip()
    turnaround = turn.perceived_ms
    within_budget = turnaround <= settings.latency_budget_ms
    TURN_TURNAROUND_MS.observe(turnaround)
    if not within_budget:
        TURNS_OVER_BUDGET_TOTAL.inc()
    return text, turnaround, within_budget, turn.speculation_hit


@router.post("/{session_id}/turns/text", response_model=TurnResponse)
async def post_text_turn(
    session_id: str,
    body: TurnRequest,
    token: str = Depends(extract_bearer_token),
    db: DbSession = Depends(get_db),
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> TurnResponse:
    text = body.text
    if not text.strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Empty answer text.",
        )
    orchestrator = await _authorized_orchestrator(session_id, token, db, store)

    agent_text, turnaround_ms, within_budget, speculation_hit = await _run_turn(
        orchestrator, text, settings
    )

    db.add(
        TurnRecord(
            session_id=session_id, speaker="candidate", text=scrub(text).text, offset_ms=now_ms()
        )
    )
    if agent_text:
        db.add(
            TurnRecord(
                session_id=session_id,
                speaker="agent",
                text=agent_text,
                offset_ms=now_ms(),
                turnaround_ms=turnaround_ms,
                within_budget=within_budget,
            )
        )
    db.commit()
    await store.touch(orchestrator)

    audio_b64 = None
    if agent_text:
        try:
            synthesizer = build_synthesizer(settings)

            async def _one_chunk():
                yield agent_text

            audio_b64 = await frames_to_wav_b64(synthesizer.stream(sentence_chunks(_one_chunk())))
        except Exception:
            logger.warning("Turn audio synthesis failed; returning text only.")

    return TurnResponse(
        agent_text=agent_text,
        audio_b64=audio_b64,
        turnaround_ms=round(turnaround_ms, 2),
        within_budget=within_budget,
        speculation_hit=speculation_hit,
        turns_completed=orchestrator.flow.turns,
        finished=orchestrator.flow.finished,
    )


@router.post("/{session_id}/turns/audio", response_model=TurnResponse)
async def post_audio_turn(
    session_id: str,
    file: UploadFile,
    token: str = Depends(extract_bearer_token),
    db: DbSession = Depends(get_db),
    store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> TurnResponse:
    """Accept a recorded answer clip, transcribe it, then run the same turn.

    Gracefully degrades: an empty/failed transcription does not end the
    interview -- it is surfaced to the Candidate App as ``stt_used: false``
    so the UI can prompt the candidate to type the answer instead, exactly
    the "handle transcription failures ... gracefully" requirement.
    """
    orchestrator = await _authorized_orchestrator(session_id, token, db, store)

    audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "Audio upload is too large.",
        )

    if not audio_bytes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Empty audio upload.",
        )

    transcriber = build_transcriber(settings)
    provider_label = "deepgram" if settings.stt_configured else "mock"
    STT_REQUESTS_TOTAL.labels(provider=provider_label).inc()
    try:
        result = await transcriber.transcribe(audio_bytes, file.content_type or "audio/webm")
    except Exception as exc:
        logger.warning("Transcription failed (%s).", type(exc).__name__)
        STT_ERRORS_TOTAL.labels(provider=provider_label).inc()
        result = None

    if result is None or not result.text.strip():
        return TurnResponse(
            agent_text="",
            turnaround_ms=0.0,
            within_budget=True,
            speculation_hit=False,
            turns_completed=orchestrator.flow.turns,
            finished=orchestrator.flow.finished,
            stt_used=False,
            stt_provider=result.provider if result else "error",
        )

    agent_text, turnaround_ms, within_budget, speculation_hit = await _run_turn(
        orchestrator, result.text, settings
    )

    db.add(
        TurnRecord(session_id=session_id, speaker="candidate", text=result.text, offset_ms=now_ms())
    )
    if agent_text:
        db.add(
            TurnRecord(
                session_id=session_id,
                speaker="agent",
                text=agent_text,
                offset_ms=now_ms(),
                turnaround_ms=turnaround_ms,
                within_budget=within_budget,
            )
        )
    db.commit()
    await store.touch(orchestrator)

    audio_b64 = None
    if agent_text:
        try:
            synthesizer = build_synthesizer(settings)

            async def _one_chunk():
                yield agent_text

            audio_b64 = await frames_to_wav_b64(synthesizer.stream(sentence_chunks(_one_chunk())))
        except Exception:
            logger.warning("Turn audio synthesis failed; returning text only.")

    return TurnResponse(
        agent_text=agent_text,
        audio_b64=audio_b64,
        turnaround_ms=round(turnaround_ms, 2),
        within_budget=within_budget,
        speculation_hit=speculation_hit,
        turns_completed=orchestrator.flow.turns,
        finished=orchestrator.flow.finished,
        stt_used=result.provider not in ("mock", "deepgram-error", "error"),
        stt_provider=result.provider,
    )


@router.post("/{session_id}/close", response_model=EvaluationResponse)
async def close_session(
    session_id: str,
    token: str = Depends(extract_bearer_token),
    db: DbSession = Depends(get_db),
    store: SessionStore = Depends(get_session_store),
) -> EvaluationResponse:
    orchestrator = await _authorized_orchestrator(session_id, token, db, store)
    evaluation = await orchestrator.close()

    record = db.get(SessionRecord, session_id)
    record.status = "closed"
    db.add(
        EvaluationRecord(
            session_id=session_id,
            target_role=evaluation.target_role.value,
            rubric_fit_index=evaluation.rubric_fit_index,
            routing_confidence=evaluation.routing_confidence,
            recommendation=evaluation.recommendation.value,
            turns_completed=evaluation.turns_completed,
            latency_compliance=evaluation.latency_compliance,
            competency_scores=evaluation.score_map(),
            eliminations=[e.model_dump(mode="json") for e in evaluation.eliminations],
            flagged_limitations=evaluation.flagged_limitations,
            payload=evaluation.to_payload(),
        )
    )
    db.commit()
    await store.close(session_id)
    SESSIONS_CLOSED_TOTAL.labels(recommendation=evaluation.recommendation.value).inc()
    ACTIVE_SESSIONS.set(len(store))

    return EvaluationResponse(
        candidate_id=evaluation.candidate_id,
        sanitized_name=evaluation.sanitized_name,
        session_id=evaluation.session_id,
        target_role=evaluation.target_role,
        rubric_fit_index=evaluation.rubric_fit_index,
        routing_confidence=evaluation.routing_confidence,
        recommendation=evaluation.recommendation.value,
        competency_scores=[
            {
                "key": s.key,
                "label": s.label,
                "score": s.score,
                "source": s.source.value,
                "rationale": s.rationale,
            }
            for s in evaluation.competency_scores
        ],
        flagged_limitations=evaluation.flagged_limitations,
        turns_completed=evaluation.turns_completed,
        latency_compliance=evaluation.latency_compliance,
    )


@router.get("/{session_id}/evaluation", response_model=EvaluationResponse)
def get_evaluation(session_id: str, db: DbSession = Depends(get_db)) -> EvaluationResponse:
    record = db.get(EvaluationRecord, session_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No evaluation yet for this session.")
    return EvaluationResponse(
        candidate_id=record.session.candidate_id,
        sanitized_name=record.session.sanitized_name,
        session_id=record.session_id,
        target_role=record.target_role,
        rubric_fit_index=record.rubric_fit_index,
        routing_confidence=record.routing_confidence,
        recommendation=record.recommendation,
        competency_scores=[
            {
                "key": k,
                "label": k.replace("_", " ").title(),
                "score": v,
                "source": "combined",
                "rationale": "",
            }
            for k, v in record.competency_scores.items()
        ],
        flagged_limitations=record.flagged_limitations,
        turns_completed=record.turns_completed,
        latency_compliance=record.latency_compliance,
    )


__all__ = ["router"]

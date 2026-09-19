"""Admin-facing endpoints: session/candidate listing, system status, health.

No separate admin auth system is introduced here -- this repository has none
to build on, and inventing one (users, passwords, JWTs) would be exactly the
kind of unrequested new subsystem the brief warns against ("do not introduce
unnecessary databases/services"). In a real deployment this router sits behind
whatever perimeter auth the platform already provides (a reverse proxy, an
IdP-gated ingress); ``HRTE_ADMIN_TOKEN`` below is the minimum viable guard for
running this outside a fully trusted network, and is optional so local/dev use
is unaffected.
"""

from __future__ import annotations

import hmac
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.agent.cognition import cognition_fallback_count
from app.api.schemas import (
    EvaluationResponse,
    LatencyStagesOut,
    SessionSummary,
    SystemStatusResponse,
    TranscriptTurnOut,
)
from app.api.session_store import SessionStore, get_session_store
from app.config import STAGE_BUDGETS_MS, Settings, get_settings
from app.db.engine import DATABASE_URL, get_db
from app.db.models import EvaluationRecord, SessionRecord

router = APIRouter(prefix="/api/admin", tags=["admin"])

_START_TIME = time.time()


def _require_admin(
    settings: Settings = Depends(get_settings),
    x_admin_token: str | None = Header(default=None),
) -> None:
    import os

    expected = os.environ.get("HRTE_ADMIN_TOKEN", "").strip()
    if expected and not hmac.compare_digest(x_admin_token or "", expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin token.")


@router.get(
    "/sessions", response_model=list[SessionSummary], dependencies=[Depends(_require_admin)]
)
def list_sessions(
    status_filter: str | None = None,
    limit: int = 50,
    db: DbSession = Depends(get_db),
) -> list[SessionSummary]:
    stmt = select(SessionRecord).order_by(SessionRecord.created_at.desc()).limit(min(limit, 200))
    if status_filter:
        stmt = stmt.where(SessionRecord.status == status_filter)
    rows = db.execute(stmt).scalars().all()
    return [SessionSummary(**row.to_summary()) for row in rows]


@router.get(
    "/sessions/{session_id}/transcript",
    response_model=list[TranscriptTurnOut],
    dependencies=[Depends(_require_admin)],
)
def admin_get_transcript(
    session_id: str, db: DbSession = Depends(get_db)
) -> list[TranscriptTurnOut]:
    """The turn-by-turn conversation, already PII-scrubbed at ingest.

    Ordered by `offset_ms` via the `turns` relationship's own ordering
    (see SessionRecord.turns in app/db/models.py), so this is chronological
    without a separate ORDER BY here.
    """
    record = db.get(SessionRecord, session_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such session.")
    return [
        TranscriptTurnOut(
            speaker=turn.speaker,
            text=turn.text,
            offset_ms=turn.offset_ms,
            turnaround_ms=turn.turnaround_ms,
        )
        for turn in record.turns
    ]


@router.get(
    "/sessions/{session_id}/evaluation",
    response_model=EvaluationResponse,
    dependencies=[Depends(_require_admin)],
)
def admin_get_evaluation(session_id: str, db: DbSession = Depends(get_db)) -> EvaluationResponse:
    record = db.get(EvaluationRecord, session_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No evaluation for this session.")
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


@router.get("/status", response_model=SystemStatusResponse, dependencies=[Depends(_require_admin)])
async def system_status(
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
) -> SystemStatusResponse:
    fallbacks = cognition_fallback_count()
    return SystemStatusResponse(
        environment=settings.environment,
        offline=settings.offline,
        cognition_configured=settings.cognition_configured,
        cognition_degraded=settings.cognition_configured and fallbacks > 0,
        cognition_fallbacks=fallbacks,
        moss_configured=settings.moss_configured,
        qdrant_configured=settings.qdrant_configured,
        transport_configured=settings.transport_configured,
        telemetry_configured=settings.telemetry_configured,
        stt_configured=settings.stt_configured,
        session_state_backend="redis" if store.remote_enabled else "in-process",
        database_url_scheme=urlsplit(DATABASE_URL).scheme,
        active_sessions=len(store),
        latency_budget_ms=settings.latency_budget_ms,
        stage_budgets=[
            LatencyStagesOut(stage=stage.value, budget_ms=budget)
            for stage, budget in STAGE_BUDGETS_MS.items()
        ],
    )


__all__ = ["router"]

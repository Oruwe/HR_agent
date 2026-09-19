"""The whole product surface: import a pool, rank it, ask about it.

Every route is admin-gated by ``HRTE_ADMIN_TOKEN`` when it is set. There is
no candidate-facing side to this application -- it exists for one hiring
manager looking at scraped records.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from app.agent.analyst import Analyst
from app.agent.cognition import Message, fallback_count
from app.api.routes_health import (
    ANALYSIS_RUNS_TOTAL,
    CANDIDATES_IMPORTED_TOTAL,
    CHAT_MESSAGES_TOTAL,
)
from app.api.schemas import (
    AnalyzeResponse,
    CandidateDetail,
    CandidateSummary,
    ChatRequest,
    ChatResponse,
    ImportRequest,
    ImportResponse,
    StatusResponse,
)
from app.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models import Candidate
from app.demo_pool import DEMO_BASELINE, DEMO_CANDIDATES
from app.ingest import build_candidate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["candidates"])

#: How many records go into one analysis or chat prompt. A hiring manager's
#: shortlist is tens of people, not thousands; this is a guard against a
#: scraper dumping its entire database into one model call, not a paging
#: strategy.
POOL_LIMIT = 200


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.environ.get("HRTE_ADMIN_TOKEN", "").strip()
    if expected and not hmac.compare_digest(x_admin_token or "", expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin token.")


def _ingest(records: list[dict[str, Any]], db: DbSession) -> ImportResponse:
    redacted = 0
    for record in records:
        row, contained_pii = build_candidate(record)
        redacted += int(contained_pii)
        db.add(row)
    db.commit()
    CANDIDATES_IMPORTED_TOTAL.inc(len(records))
    total = db.execute(select(func.count()).select_from(Candidate)).scalar_one()
    return ImportResponse(imported=len(records), redacted=redacted, total_in_pool=int(total))


@router.post(
    "/candidates/import",
    response_model=ImportResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def import_candidates(body: ImportRequest, db: DbSession = Depends(get_db)) -> ImportResponse:
    """Ingest scraped records of any shape, scrubbing PII at the boundary."""
    return _ingest(body.candidates, db)


@router.post(
    "/candidates/demo",
    response_model=ImportResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def load_demo_pool(db: DbSession = Depends(get_db)) -> ImportResponse:
    """Load the bundled demo pool, pre-ranked, into an empty deployment.

    The baseline rankings come from ``DEMO_BASELINE`` -- hand-written, not
    model output -- so the board is populated immediately. Running an analysis
    overwrites every one of them with the model's own judgement.
    """
    existing = db.execute(select(func.count()).select_from(Candidate)).scalar_one()
    if int(existing):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The pool already has candidates. Clear it before loading the demo pool.",
        )

    result = _ingest(list(DEMO_CANDIDATES), db)
    apply_baseline(db.execute(select(Candidate)).scalars().all())
    db.commit()
    return result


def apply_baseline(rows: Sequence[Candidate]) -> int:
    """Stamp the hand-written demo rankings onto matching rows."""
    now = int(time.time())
    applied = 0
    for row in rows:
        baseline = DEMO_BASELINE.get(row.name)
        if baseline is None:
            continue
        row.score, row.recommendation, row.rationale = baseline
        row.analyzed_at = now
        applied += 1
    return applied


@router.get(
    "/candidates",
    response_model=list[CandidateSummary],
    dependencies=[Depends(require_admin)],
)
def list_candidates(db: DbSession = Depends(get_db)) -> list[CandidateSummary]:
    """The ranked pool: scored candidates first, best first, unscored last."""
    rows = db.execute(select(Candidate)).scalars().all()
    ordered = sorted(
        rows,
        key=lambda c: (c.score is None, -(c.score or 0.0), c.name.lower()),
    )
    return [CandidateSummary(**row.to_summary()) for row in ordered]


@router.get(
    "/candidates/{candidate_id}",
    response_model=CandidateDetail,
    dependencies=[Depends(require_admin)],
)
def get_candidate(candidate_id: str, db: DbSession = Depends(get_db)) -> CandidateDetail:
    row = db.get(Candidate, candidate_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such candidate.")
    return CandidateDetail(**row.to_detail())


@router.delete(
    "/candidates/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)],
)
def delete_candidate(candidate_id: str, db: DbSession = Depends(get_db)) -> None:
    row = db.get(Candidate, candidate_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such candidate.")
    db.delete(row)
    db.commit()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_admin)],
)
async def analyze_pool(
    db: DbSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """Score the whole pool in one pass.

    One call covering every candidate, rather than one call each: ranking is
    comparative, and a model shown the field can say "stronger than the other
    three on distributed systems" where a model shown one record cannot.
    """
    rows = db.execute(select(Candidate).limit(POOL_LIMIT)).scalars().all()
    if not rows:
        return AnalyzeResponse(analyzed=0, skipped=0, offline=settings.offline)

    ANALYSIS_RUNS_TOTAL.inc()
    rankings = await Analyst(settings).rank(rows)
    by_id = {r.candidate_id: r for r in rankings}

    now = int(time.time())
    analyzed = 0
    for row in rows:
        ranking = by_id.get(row.id)
        if ranking is None:
            continue
        row.score = ranking.score
        row.recommendation = ranking.verdict
        row.rationale = ranking.rationale
        row.analyzed_at = now
        analyzed += 1
    db.commit()

    return AnalyzeResponse(
        analyzed=analyzed, skipped=len(rows) - analyzed, offline=settings.offline
    )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_admin)])
async def chat(
    body: ChatRequest,
    db: DbSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Answer one manager question against the pool."""
    CHAT_MESSAGES_TOTAL.inc()
    rows = db.execute(select(Candidate).limit(POOL_LIMIT)).scalars().all()
    history = [Message(role=m.role, content=m.content) for m in body.history]
    reply = await Analyst(settings).answer(body.message, rows, history)
    return ChatResponse(reply=reply, candidates_considered=len(rows))


@router.get("/status", response_model=StatusResponse, dependencies=[Depends(require_admin)])
def system_status(
    db: DbSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StatusResponse:
    total = int(db.execute(select(func.count()).select_from(Candidate)).scalar_one())
    analyzed = int(
        db.execute(
            select(func.count()).select_from(Candidate).where(Candidate.analyzed_at.is_not(None))
        ).scalar_one()
    )
    fallbacks = fallback_count()
    return StatusResponse(
        environment=settings.environment,
        offline=settings.offline,
        model_configured=settings.model_configured,
        model=settings.model,
        degraded=settings.model_configured and fallbacks > 0,
        fallbacks=fallbacks,
        candidates=total,
        analyzed=analyzed,
    )


__all__ = ["POOL_LIMIT", "apply_baseline", "require_admin", "router"]

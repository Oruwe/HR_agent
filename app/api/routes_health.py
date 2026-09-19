"""Health, readiness, and Prometheus metrics.

Kept as a separate, unauthenticated, unprefixed router: these are the
endpoints an orchestrator (Kubernetes, a load balancer, Docker Compose's
``healthcheck``) polls constantly, and they must never depend on the app's
own DB/session-store health in a way that could create a restart loop from a
transient blip -- ``/health`` is "the process is alive", ``/ready`` is
"the process can serve a request right now".
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import text

from app.db.engine import SessionLocal

router = APIRouter(tags=["health"])

_START_TIME = time.time()

# -- Prometheus metrics -------------------------------------------------------
# Real counters wired from actual request handling (see app/api/app.py's
# middleware and app/api/routes_sessions.py), not placeholder values.

HTTP_REQUESTS_TOTAL = Counter(
    "hrte_http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "hrte_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
)
TURN_TURNAROUND_MS = Histogram(
    "hrte_turn_turnaround_ms",
    "Candidate-perceived turnaround per conversational turn",
    buckets=(10, 25, 50, 100, 160, 250, 500, 1000, 2000, 4000),
)
TURNS_OVER_BUDGET_TOTAL = Counter(
    "hrte_turns_over_budget_total", "Turns whose turnaround exceeded the 160ms budget"
)
ACTIVE_SESSIONS = Gauge("hrte_active_sessions", "Currently open screening sessions")
SESSIONS_CREATED_TOTAL = Counter("hrte_sessions_created_total", "Sessions created")
SESSIONS_CLOSED_TOTAL = Counter("hrte_sessions_closed_total", "Sessions closed", ["recommendation"])
STT_REQUESTS_TOTAL = Counter("hrte_stt_requests_total", "Speech-to-text calls", ["provider"])
STT_ERRORS_TOTAL = Counter("hrte_stt_errors_total", "Speech-to-text failures", ["provider"])


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "uptime_s": round(time.time() - _START_TIME, 1)}


@router.get("/ready")
def ready() -> Response:
    """True readiness: can this process actually reach its dependencies?

    Checks the one hard dependency this app cannot run without -- the
    database. External providers (LLM, STT, Moss, LiveKit) are intentionally
    excluded: they each have a documented offline fallback, so their absence
    degrades quality, not availability, and should never fail a readiness
    probe.
    """
    checks: dict[str, bool] = {}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    all_ok = all(checks.values())
    return Response(
        content=json.dumps({"ready": all_ok, "checks": checks}),
        media_type="application/json",
        status_code=200 if all_ok else 503,
    )


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = [
    "ACTIVE_SESSIONS",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "SESSIONS_CLOSED_TOTAL",
    "SESSIONS_CREATED_TOTAL",
    "STT_ERRORS_TOTAL",
    "STT_REQUESTS_TOTAL",
    "TURNS_OVER_BUDGET_TOTAL",
    "TURN_TURNAROUND_MS",
    "router",
]
